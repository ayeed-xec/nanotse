"""Identity-Binding Accuracy (IBA): Hungarian-matched cross-session identity tracking.

**Paper contribution 3.** Measures whether the same speaker is assigned to the
same slot across discontinuous appearances ("Alice leaves and returns 30 s
later"). Given predicted slot indices and ground-truth speaker labels across
one or more sessions, find the optimal slot->speaker matching via Hungarian
and report the fraction of frames at the optimal matching.

Multi-session evaluation: concatenate sessions' frame sequences and call
:func:`iba_multi_session`. The metric finds a *single global* slot->speaker
mapping across all sessions, so a model that re-uses the same slot for the
same speaker across sessions scores higher than one that picks a fresh slot
each session.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ArrayLike = torch.Tensor | np.ndarray


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def iba_score(slot_assignments: ArrayLike, speaker_labels: ArrayLike) -> float:
    """Hungarian-matched identity-binding accuracy.

    Args:
        slot_assignments: ``(T,)`` integer slot indices, one per frame.
        speaker_labels: ``(T,)`` integer speaker IDs, one per frame.
            For multi-session eval, concatenate all sessions' frames.

    Returns:
        Float in ``[0, 1]``. 1.0 = every frame's slot matches the canonical
        slot for its speaker.
    """
    sa = _to_numpy(slot_assignments)
    sl = _to_numpy(speaker_labels)
    if sa.shape != sl.shape:
        raise ValueError(f"shape mismatch: {sa.shape} vs {sl.shape}")
    if sa.ndim != 1:
        raise ValueError(f"expected 1D inputs, got {sa.shape}")
    if sa.size == 0:
        raise ValueError("empty input")

    unique_slots = sorted({int(s) for s in sa.tolist()})
    unique_speakers = sorted({int(s) for s in sl.tolist()})
    slot_idx = {s: i for i, s in enumerate(unique_slots)}
    spk_idx = {s: i for i, s in enumerate(unique_speakers)}

    confusion = np.zeros((len(unique_slots), len(unique_speakers)), dtype=np.int64)
    for s, k in zip(sa.tolist(), sl.tolist(), strict=True):
        confusion[slot_idx[int(s)], spk_idx[int(k)]] += 1

    # linear_sum_assignment minimizes; negate to maximize total matched count.
    row_ind, col_ind = linear_sum_assignment(-confusion)
    matched = int(confusion[row_ind, col_ind].sum())
    return matched / int(sa.size)


def iba_multi_session(sessions: list[tuple[ArrayLike, ArrayLike]]) -> float:
    """IBA across multiple sessions; concatenates all frames before scoring.

    Each session is a ``(slot_assignments, speaker_labels)`` tuple. The score
    rewards models that re-use the same slot for the same speaker across
    sessions.
    """
    if not sessions:
        raise ValueError("sessions list is empty")
    all_slots = np.concatenate([_to_numpy(s[0]) for s in sessions])
    all_speakers = np.concatenate([_to_numpy(s[1]) for s in sessions])
    return iba_score(all_slots, all_speakers)
