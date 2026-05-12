"""Active-Speaker Detection head: per-slot logits over time.

W3.4 of the sprint plan. For each (audio frame, slot) pair, produces a
logit; higher logit = "this slot's speaker is the active talker right now."
The head is trained with BCE against a one-hot ground-truth target slot.

Shape: ``features (B, T, D)``, ``slots (B, N, S)`` -> ``(B, T, N)``.
"""

from __future__ import annotations

import torch
from torch import nn


class ASDHead(nn.Module):
    def __init__(self, d_model: int = 256, d_slot: int = 256, d_hidden: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model + d_slot, d_hidden)
        self.head = nn.Linear(d_hidden, 1)

    def forward(self, features: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """``features (B, T, D)`` + ``slots (B, N, S)`` -> logits ``(B, T, N)``."""
        b, t, d = features.shape
        n = slots.shape[1]
        s = slots.shape[2]
        feat_exp = features.unsqueeze(2).expand(b, t, n, d)
        slot_exp = slots.unsqueeze(1).expand(b, t, n, s)
        combined = torch.cat([feat_exp, slot_exp], dim=-1)  # (B, T, N, D+S)
        x = torch.relu(self.proj(combined))
        out: torch.Tensor = self.head(x).squeeze(-1)  # (B, T, N)
        return out
