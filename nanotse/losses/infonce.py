"""InfoNCE on slot embeddings: pull same-speaker close, push different-speaker apart.

Library function -- not yet wired into training. We'll add it to the loss
schedule in scripts/train.py once we have real speaker labels (W2.5 + real
VoxCeleb2-mix loader). Synthetic data has speaker_ids but they don't
correspond to learnable identities, so wiring it now would be cargo-cult.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def slot_infonce(
    embeddings: torch.Tensor,
    speaker_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Contrastive loss over per-sample slot embeddings.

    Args:
        embeddings: ``(B, D)`` one slot embedding per sample (e.g. dominant slot
            or mean-pooled across N).
        speaker_ids: ``(B,)`` integer speaker labels.
        temperature: softmax temperature.

    Returns:
        Scalar loss. If no positive pairs exist in the batch (all unique
        speakers) returns 0 -- caller's responsibility to ensure batches
        contain repeats.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"embeddings must be 2D, got shape {tuple(embeddings.shape)}")
    if speaker_ids.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"batch mismatch: embeddings {embeddings.shape[0]} vs speaker_ids "
            f"{speaker_ids.shape[0]}"
        )

    z = F.normalize(embeddings, dim=-1)
    sim = z @ z.t() / temperature  # (B, B)

    pos_mask = (speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0.0)

    if pos_mask.sum() == 0:
        return torch.zeros((), device=embeddings.device)

    # Exclude self from the partition function.
    sim.fill_diagonal_(float("-inf"))
    log_prob = F.log_softmax(sim, dim=-1)  # (B, B), -inf on the diagonal

    # Sum log_prob only at positive positions. Use torch.where to avoid
    # -inf * 0 -> NaN at the masked-out diagonal entries.
    pos_log_prob = torch.where(pos_mask > 0, log_prob, torch.zeros_like(log_prob)).sum(dim=-1)
    n_pos = pos_mask.sum(dim=-1)  # (B,)
    valid = n_pos > 0
    loss_per_sample = -pos_log_prob[valid] / n_pos[valid]
    out: torch.Tensor = loss_per_sample.mean()
    return out
