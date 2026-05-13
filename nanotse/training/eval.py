"""Validation pass: mean SI-SNR / SI-SDRi on the val split.

Used by ``scripts/train.py`` every ``cfg.train.val_every`` steps so the
loop can (a) detect overfitting, (b) save ``best.pt`` keyed on val SDRi,
and (c) log val metrics into the same ``metrics.jsonl`` as train rows.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from nanotse.losses import si_snr
from nanotse.models.nanotse import NanoTSE


@torch.no_grad()
def run_val_pass(
    model: nn.Module,
    val_loader: DataLoader[Any],
    device: torch.device,
    max_clips: int | None = None,
) -> dict[str, float]:
    """Forward the model over (up to ``max_clips``) val items, return mean stats.

    Returns ``{val_baseline_si_snr_db, val_estimate_si_snr_db, val_sdri_db, val_n}``.
    """
    was_training = model.training
    model.eval()

    base_sum = 0.0
    est_sum = 0.0
    n = 0
    for batch in val_loader:
        mix = batch["mix"].to(device)
        tgt = batch["target"].to(device)
        if isinstance(model, NanoTSE):
            video = batch["face"].to(device) if model.with_visual and "face" in batch else None
            enroll = (
                batch["enrollment"].to(device)
                if model.with_enrollment and "enrollment" in batch
                else None
            )
            est, _, _ = model(mix, video, enroll)
        else:
            out = model(mix)
            est = out[0] if isinstance(out, tuple) else out

        base_sum += float(si_snr(mix, tgt).mean().item()) * mix.shape[0]
        est_sum += float(si_snr(est, tgt).mean().item()) * mix.shape[0]
        n += mix.shape[0]
        if max_clips is not None and n >= max_clips:
            break

    if was_training:
        model.train()

    if n == 0:
        return {
            "val_baseline_si_snr_db": 0.0,
            "val_estimate_si_snr_db": 0.0,
            "val_sdri_db": 0.0,
            "val_n": 0,
        }

    base = base_sum / n
    est = est_sum / n
    return {
        "val_baseline_si_snr_db": base,
        "val_estimate_si_snr_db": est,
        "val_sdri_db": est - base,
        "val_n": n,
    }
