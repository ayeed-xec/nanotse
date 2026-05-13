"""Tests for Tier-2 training improvements: LR schedule + EMA + energy-VAD ASD targets."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from nanotse.training import EMA, warmup_cosine_lr_multiplier
from nanotse.training.losses import target_active_mask


def test_lr_schedule_warmup_ramps_linearly() -> None:
    """During warmup, multiplier should rise linearly from 0 to 1."""
    assert warmup_cosine_lr_multiplier(0, warmup_steps=100, total_steps=1000) == 0.0
    assert math.isclose(
        warmup_cosine_lr_multiplier(50, warmup_steps=100, total_steps=1000),
        0.5,
        abs_tol=1e-6,
    )
    assert math.isclose(
        warmup_cosine_lr_multiplier(100, warmup_steps=100, total_steps=1000),
        1.0,
        abs_tol=1e-6,
    )


def test_lr_schedule_cosine_decays_to_min_ratio() -> None:
    """At total_steps the multiplier should equal min_lr_ratio."""
    mult = warmup_cosine_lr_multiplier(1000, warmup_steps=100, total_steps=1000, min_lr_ratio=0.05)
    assert math.isclose(mult, 0.05, abs_tol=1e-6)
    # Halfway through decay: mult should be ~ min + (1-min) * 0.5
    mid = warmup_cosine_lr_multiplier(550, warmup_steps=100, total_steps=1000, min_lr_ratio=0.05)
    assert 0.45 < mid < 0.6, f"halfway multiplier looks off: {mid}"


def test_lr_schedule_clamps_negative_and_overrun() -> None:
    assert warmup_cosine_lr_multiplier(-5, warmup_steps=10, total_steps=100) == 0.0
    assert math.isclose(
        warmup_cosine_lr_multiplier(99999, warmup_steps=10, total_steps=100, min_lr_ratio=0.1),
        0.1,
        abs_tol=1e-6,
    )


def test_ema_rejects_bad_decay() -> None:
    m = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="decay"):
        EMA(m, decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        EMA(m, decay=-0.1)


def test_ema_update_moves_shadow_toward_live() -> None:
    """After update, shadow should be a convex combo of old shadow + live params."""
    torch.manual_seed(0)
    m = nn.Linear(4, 4)
    ema = EMA(m, decay=0.5)
    orig_shadow = ema.shadow["weight"].clone()

    # Set live weights to known values, update EMA.
    with torch.no_grad():
        m.weight.fill_(10.0)
    ema.update(m)
    expected = 0.5 * orig_shadow + 0.5 * torch.full_like(orig_shadow, 10.0)
    assert torch.allclose(ema.shadow["weight"], expected, atol=1e-6)


def test_ema_apply_and_restore_round_trip() -> None:
    """apply_to swaps EMA into model; restore puts the originals back."""
    torch.manual_seed(0)
    m = nn.Linear(4, 4)
    ema = EMA(m, decay=0.99)
    # Move EMA shadow away from live so the swap is detectable.
    with torch.no_grad():
        for k in ema.shadow:
            ema.shadow[k].fill_(7.0)
    live_before = m.weight.detach().clone()
    backup = ema.apply_to(m)
    assert torch.allclose(m.weight, torch.full_like(m.weight, 7.0))
    ema.restore(m, backup)
    assert torch.allclose(m.weight, live_before)


def test_target_active_mask_shape_and_dtype() -> None:
    target = torch.randn(2, 16000)
    mask = target_active_mask(target, n_frames=100)
    assert mask.shape == (2, 100)
    assert mask.dtype == torch.bool


def test_target_active_mask_all_false_when_silent() -> None:
    """All-zero target -> RMS = 0 everywhere, no frame counts as active."""
    target = torch.zeros(1, 16000)
    mask = target_active_mask(target, n_frames=100)
    assert not mask.any()


def test_target_active_mask_all_true_when_uniformly_loud() -> None:
    """Uniformly-loud target -> all frames active (every frame's RMS == median)."""
    target = torch.ones(1, 16000)
    mask = target_active_mask(target, n_frames=100, rms_threshold_ratio=0.5)
    assert mask.all()
