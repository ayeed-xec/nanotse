"""Learning-rate schedule: linear warmup + cosine decay.

A small standalone function so ``scripts/train.py`` can call it once per
step without pulling in a heavier scheduler dependency. Returns the
LR-multiplier to apply to ``cfg.train.lr``.

Curve:
    0..warmup_steps:           lr * step/warmup_steps  (linear ramp)
    warmup..total_steps:       lr * 0.5*(1 + cos(pi * progress))  (cosine decay)
    after total_steps:         lr * min_lr_ratio

The integration into the optimizer is a one-line ``for g in opt.param_groups: g["lr"] = base * mult``
in the training loop.
"""

from __future__ import annotations

import math


def warmup_cosine_lr_multiplier(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
) -> float:
    """Returns the multiplier in [min_lr_ratio, 1.0] for the current step."""
    if step < 0:
        return 0.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cos_term)
