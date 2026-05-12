"""IBA metric behaviour: perfect alignment, Hungarian permutation, cross-session."""

from __future__ import annotations

import pytest
import torch

from nanotse.eval.iba import iba_multi_session, iba_score


def test_iba_perfect_alignment_is_one() -> None:
    slots = torch.tensor([0, 0, 1, 1, 2, 2])
    speakers = torch.tensor([0, 0, 1, 1, 2, 2])
    assert iba_score(slots, speakers) == 1.0


def test_iba_hungarian_resolves_arbitrary_permutation() -> None:
    """Slot indices != speaker indices but the mapping is consistent -> still 1.0."""
    slots = torch.tensor([3, 3, 7, 7, 1, 1])
    speakers = torch.tensor([0, 0, 1, 1, 2, 2])
    assert iba_score(slots, speakers) == 1.0


def test_iba_alice_leaves_returns_same_slot_is_one() -> None:
    """Alice in slot 3 in both sessions -> IBA = 1.0."""
    session1 = (torch.tensor([3, 3, 3, 3]), torch.tensor([0, 0, 0, 0]))
    session2 = (torch.tensor([3, 3, 3, 3]), torch.tensor([0, 0, 0, 0]))
    assert iba_multi_session([session1, session2]) == 1.0


def test_iba_alice_changes_slot_across_sessions_drops_score() -> None:
    """Alice in slot 3 then slot 7: Hungarian picks one canonical slot -> 0.5."""
    session1 = (torch.tensor([3, 3, 3, 3]), torch.tensor([0, 0, 0, 0]))
    session2 = (torch.tensor([7, 7, 7, 7]), torch.tensor([0, 0, 0, 0]))
    assert iba_multi_session([session1, session2]) == 0.5


def test_iba_random_assignment_is_low() -> None:
    torch.manual_seed(0)
    slots = torch.randint(0, 16, (1000,))
    speakers = torch.randint(0, 8, (1000,))
    score = iba_score(slots, speakers)
    assert score < 0.5  # well below random-perfect alignment


def test_iba_accepts_numpy_arrays() -> None:
    import numpy as np

    slots = np.array([0, 0, 1, 1])
    speakers = np.array([0, 0, 1, 1])
    assert iba_score(slots, speakers) == 1.0


def test_iba_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        iba_score(torch.zeros(4, dtype=torch.long), torch.zeros(5, dtype=torch.long))


def test_iba_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1D"):
        iba_score(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long))


def test_iba_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        iba_score(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))


def test_iba_multi_session_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        iba_multi_session([])
