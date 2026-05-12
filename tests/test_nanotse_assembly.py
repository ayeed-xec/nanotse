"""Forward + overfit checks for the audio-only NanoTSE assembly (W2.4)."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from nanotse.data import SyntheticAVMixDataset
from nanotse.losses import negative_si_snr
from nanotse.models.nanotse import NanoTSE


def test_nanotse_forward_shape() -> None:
    m = NanoTSE()
    x = torch.randn(2, 16000)
    y = m(x)
    assert y.shape == (2, 16000)


def test_nanotse_smoke_4s_shape() -> None:
    m = NanoTSE()
    x = torch.randn(1, 16000 * 4)
    y = m(x)
    assert y.shape == (1, 16000 * 4)


def test_nanotse_overfit_4_clips_decreases_loss() -> None:
    """Same plumbing bar as TDSEBaseline: NanoTSE must train on 4 clips
    without NaNs and reduce loss by >= 1 dB.

    The PLAN W2 +10 dB target is for real speech once the VoxCeleb2-mix
    loader lands.
    """
    torch.manual_seed(0)
    ds = SyntheticAVMixDataset(num_clips=4, clip_seconds=1.0, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = NanoTSE(d_model=64, n_heads=2, n_layers=1, cache_len=50)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for _ in range(60):
        for batch in loader:
            est = model(batch["mix"])
            loss = negative_si_snr(est, batch["target"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

    assert all(math.isfinite(loss_val) for loss_val in losses), "non-finite loss"
    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    assert final < initial - 1.0, (
        f"loss did not decrease >=1 dB: initial {initial:.2f} -> final {final:.2f}"
    )
