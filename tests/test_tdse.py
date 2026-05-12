"""Forward shape + 8-clip overfit plumbing for the W2 TDSE baseline."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from nanotse.data import SyntheticAVMixDataset
from nanotse.losses import negative_si_snr
from nanotse.models.baselines.tdse import TDSEBaseline


def test_tdse_forward_shape() -> None:
    model = TDSEBaseline()
    x = torch.randn(2, 16000)
    y = model(x)
    assert y.shape == (2, 16000), f"expected (2, 16000), got {y.shape}"


def test_tdse_overfit_8_clips_decreases_loss() -> None:
    """Overfit on 8 deterministic synthetic clips.

    The PLAN W2 bar of ``+10 dB SI-SDRi on 8-clip overfit`` is for real
    VoxCeleb2-mix speech. On synthetic gaussian mixes the model has no
    speech structure to lock onto, so we assert ``loss decreases by
    >=1 dB`` -- enough to prove the W2 audio pipe wires through correctly.
    """
    torch.manual_seed(0)
    ds = SyntheticAVMixDataset(num_clips=8, clip_seconds=1.0, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = TDSEBaseline(n_feat=64, bottleneck=32, n_blocks=3)
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
