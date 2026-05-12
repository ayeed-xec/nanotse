"""Forward + overfit checks for the audio-only NanoTSE assembly (W2.4)."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from nanotse.data import SyntheticAVMixDataset
from nanotse.losses import negative_si_snr
from nanotse.models.nanotse import NanoTSE


def test_nanotse_audio_only_forward_shape() -> None:
    m = NanoTSE(with_visual=False)
    x = torch.randn(2, 16000)
    y, asd = m(x)
    assert y.shape == (2, 16000)
    assert asd is None


def test_nanotse_audio_only_smoke_4s_shape() -> None:
    m = NanoTSE(with_visual=False)
    x = torch.randn(1, 16000 * 4)
    y, _ = m(x)
    assert y.shape == (1, 16000 * 4)


def test_nanotse_av_forward_shape() -> None:
    """Full AV path: returns tse_out + per-slot ASD logits."""
    m = NanoTSE(
        d_model=64,
        n_heads=2,
        n_layers=1,
        d_visual=64,
        n_slots=4,
        d_slot=64,
        n_slot_iters=2,
        frame_size=32,
    )
    audio = torch.randn(1, 16000)  # 1 s -> Ta=100
    video = torch.randint(0, 256, (1, 25, 32, 32, 3), dtype=torch.uint8)  # 25 frames
    tse_out, asd_logits = m(audio, video)
    assert tse_out.shape == (1, 16000)
    assert asd_logits is not None
    assert asd_logits.shape == (1, 100, 4)


def test_nanotse_audio_only_overfit_4_clips_decreases_loss() -> None:
    """Plumbing bar: NanoTSE (audio-only) trains on 4 clips and loss drops >= 1 dB.

    The PLAN W2 +10 dB target applies to real speech, not synthetic gaussian.
    """
    torch.manual_seed(0)
    ds = SyntheticAVMixDataset(num_clips=4, clip_seconds=1.0, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = NanoTSE(d_model=64, n_heads=2, n_layers=1, cache_len=50, with_visual=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for _ in range(60):
        for batch in loader:
            est, _ = model(batch["mix"])
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
