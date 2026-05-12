"""Binary cross-entropy on per-slot active-speaker logits.

Pairs with :class:`nanotse.models.heads.asd.ASDHead`. Logits shape ``(B, T, N)``;
target is a per-frame slot index ``(B, T)`` indicating which slot is the
currently active speaker.

Library function -- wired into training once real speaker ground-truth labels
are available (W2.5+).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def asd_bce(asd_logits: torch.Tensor, target_slot: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits over per-slot active-speaker predictions.

    Args:
        asd_logits: ``(B, T, N)`` raw logits.
        target_slot: ``(B, T)`` integer slot index of the active speaker per frame.

    Returns:
        Scalar mean loss.
    """
    if asd_logits.dim() != 3:
        raise ValueError(f"asd_logits must be 3D, got shape {tuple(asd_logits.shape)}")
    n = asd_logits.shape[-1]
    target = F.one_hot(target_slot, num_classes=n).float()
    out: torch.Tensor = F.binary_cross_entropy_with_logits(asd_logits, target)
    return out
