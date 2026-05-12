"""Named-slot identity memory: Locatello slot competition + GRU update.

W3.3 of the sprint plan. **Paper contribution 1.** Each slot holds a joint
``(face, voice)`` embedding (here a single ``d_slot``-dim vector that mixes
both modalities through training). Slots compete for input via softmax-over-
slots attention; each slot updates via a ``GRUCell`` on its assigned input.

The slot bank persists across streaming chunks -- caller passes ``state``
in/out explicitly. LRU eviction is deferred to the integration layer
(W3.5+); for now slots are a fixed bank that just keeps refining.

State schema (per chunk call):
    {"slots":  (B, N, d_slot),
     "step":   int}
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn


class SlotState(TypedDict):
    slots: torch.Tensor  # (B, N, d_slot)
    step: int


class NamedSlotMemory(nn.Module):
    """Locatello-style slot attention with persistent slot state across chunks.

    Returns features augmented with the soft per-token slot mixture (so each
    input token carries identity context downstream) AND the slot bank itself
    (so the ASD head can score active-speaker per slot).
    """

    def __init__(
        self,
        n_slots: int = 16,
        d_input: int = 256,
        d_slot: int = 256,
        n_iters: int = 3,
        mlp_mult: int = 2,
    ) -> None:
        super().__init__()
        if n_slots <= 0:
            raise ValueError(f"n_slots must be > 0, got {n_slots}")
        if n_iters <= 0:
            raise ValueError(f"n_iters must be > 0, got {n_iters}")
        self.n_slots = n_slots
        self.d_input = d_input
        self.d_slot = d_slot
        self.n_iters = n_iters

        self.norm_inputs = nn.LayerNorm(d_input)
        self.norm_slots = nn.LayerNorm(d_slot)
        self.norm_mlp = nn.LayerNorm(d_slot)

        self.q_proj = nn.Linear(d_slot, d_slot)
        self.k_proj = nn.Linear(d_input, d_slot)
        self.v_proj = nn.Linear(d_input, d_slot)

        self.gru = nn.GRUCell(d_slot, d_slot)
        self.mlp = nn.Sequential(
            nn.Linear(d_slot, mlp_mult * d_slot),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_mult * d_slot, d_slot),
        )

        # Learnable initial slot bank, shared across batch at init_state time.
        self.slot_init = nn.Parameter(torch.randn(1, n_slots, d_slot) * 0.02)

    def init_state(self, batch_size: int, device: torch.device) -> SlotState:
        return {
            "slots": self.slot_init.expand(batch_size, -1, -1).contiguous().to(device),
            "step": 0,
        }

    def _iterate(self, slots: torch.Tensor, x_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One slot-competition iteration. Returns (new_slots, attn)."""
        b, _, _ = slots.shape
        q = self.q_proj(self.norm_slots(slots))  # (B, N, S)
        k = self.k_proj(x_n)  # (B, T, S)
        v = self.v_proj(x_n)  # (B, T, S)

        scale = 1.0 / (self.d_slot**0.5)
        # softmax over SLOTS -- Locatello's twist (each input token's attention sums to 1 over slots).
        attn_logits = (k @ q.transpose(-2, -1)) * scale  # (B, T, N)
        attn = attn_logits.softmax(dim=-1)

        # Normalize per-slot weights over the input axis before aggregation.
        weights = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)  # (B, T, N)
        updates = weights.transpose(1, 2) @ v  # (B, N, S)

        # GRU update
        flat_updates = updates.reshape(-1, self.d_slot)
        flat_slots = slots.reshape(-1, self.d_slot)
        new_slots = self.gru(flat_updates, flat_slots).reshape(b, self.n_slots, self.d_slot)

        # MLP residual block
        new_slots = new_slots + self.mlp(self.norm_mlp(new_slots))
        return new_slots, attn

    def forward_chunk(
        self, x: torch.Tensor, state: SlotState
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], SlotState]:
        """Returns ``((augmented_features (B,T,D+S), slots (B,N,S)), new_state)``."""
        x_n = self.norm_inputs(x)
        slots = state["slots"]
        attn = torch.zeros(x.shape[0], x.shape[1], self.n_slots, device=x.device)
        for _ in range(self.n_iters):
            slots, attn = self._iterate(slots, x_n)

        # Soft per-token slot mixture (differentiable).
        soft_slot = attn @ slots  # (B, T, S)
        augmented = torch.cat([x, soft_slot], dim=-1)  # (B, T, D + S)

        new_state: SlotState = {"slots": slots, "step": state["step"] + 1}
        return (augmented, slots), new_state

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Offline: init fresh slots, run one chunk, return augmented + slots."""
        state = self.init_state(x.shape[0], x.device)
        (augmented, slots), _ = self.forward_chunk(x, state)
        return augmented, slots
