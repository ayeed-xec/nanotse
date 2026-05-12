"""W2.4 acceptance gate: NanoTSE overfits 8 real-speech clips to >= +10 dB SI-SDRi.

This is the first paper-grade measurement in the project. Synthetic plumbing
tests verify the architecture wires up correctly; this test verifies the
model actually *learns from speech* on real audio.

Marked ``slow`` (~30-60 s on M3 MPS). Skips cleanly if real data hasn't
been fetched yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from nanotse.data import VoxCeleb2MixDataset
from nanotse.losses import negative_si_snr, si_snr
from nanotse.models.nanotse import NanoTSE

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO_ROOT / "data" / "smoke"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (SMOKE_ROOT / "manifest.json").exists(),
        reason="data/smoke/manifest.json missing; run scripts/data_prep/fetch_voxceleb2_mix_smoke.py",
    ),
]


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mean_si_snr(model: NanoTSE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            mix = batch["mix"].to(device)
            tgt = batch["target"].to(device)
            out = model(mix)
            est = out[0] if isinstance(out, tuple) else out
            total += si_snr(est, tgt).mean().item()
            n += 1
    return total / n


def _mean_baseline_si_snr(loader: DataLoader, device: torch.device) -> float:
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            mix = batch["mix"].to(device)
            tgt = batch["target"].to(device)
            total += si_snr(mix, tgt).mean().item()
            n += 1
    return total / n


def test_w24_real_speech_8_clip_overfit() -> None:
    torch.manual_seed(0)
    device = _pick_device()

    ds = VoxCeleb2MixDataset(SMOKE_ROOT, split="train", clip_seconds=2.0, num_items=8, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = NanoTSE(with_visual=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    baseline = _mean_baseline_si_snr(loader, device)

    model.train()
    for _ in range(250):
        for batch in loader:
            mix = batch["mix"].to(device)
            tgt = batch["target"].to(device)
            out = model(mix)
            est = out[0] if isinstance(out, tuple) else out
            loss = negative_si_snr(est, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()

    final = _mean_si_snr(model, loader, device)
    sdri = final - baseline
    print(
        f"\nW2.4 real-speech gate (device={device}): "
        f"baseline={baseline:+.2f} dB  final={final:+.2f} dB  "
        f"SI-SDRi={sdri:+.2f} dB  [target >= +10 dB]"
    )
    assert sdri >= 10.0, (
        f"W2.4 gate not met: SI-SDRi = {sdri:.2f} dB "
        f"(baseline {baseline:+.2f} -> final {final:+.2f})"
    )
