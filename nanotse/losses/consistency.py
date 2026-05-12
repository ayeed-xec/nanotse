"""Slot embedding consistency: pull slot bank at time t toward time t+gap.

Used when the same speaker appears in both windows -- the slot bound to
them should stay stable in embedding space, which is the underlying
mechanism for "Alice leaves and returns" re-binding (IBA contribution).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def slot_consistency(slots_before: torch.Tensor, slots_after: torch.Tensor) -> torch.Tensor:
    """MSE between slot banks ``(B, N, S)`` at two timepoints.

    Caller's responsibility to ensure both banks describe the same set of
    speakers (otherwise this loss penalizes legitimate identity changes).
    """
    if slots_before.shape != slots_after.shape:
        raise ValueError(
            f"slot bank shape mismatch: {tuple(slots_before.shape)} vs {tuple(slots_after.shape)}"
        )
    out: torch.Tensor = F.mse_loss(slots_before, slots_after)
    return out
