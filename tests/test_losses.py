"""Behaviour checks for the multi-task losses (InfoNCE, ASD BCE, consistency)."""

from __future__ import annotations

import pytest
import torch

from nanotse.losses import asd_bce, slot_consistency, slot_infonce

# -------------------------- InfoNCE ----------------------------------------


def test_infonce_finite_for_identical_same_speaker() -> None:
    """All embeddings identical + all same speaker -> finite, no NaN."""
    emb = torch.ones(4, 8)
    spk = torch.zeros(4, dtype=torch.long)
    loss = slot_infonce(emb, spk)
    assert torch.isfinite(loss)
    # Floor of log(B-1) for uniform softmax among B-1 positives:
    # B=4 -> log(3) ~= 1.0986. Allow a touch of slack.
    assert loss.item() <= 1.2


def test_infonce_no_positive_pairs_returns_zero() -> None:
    """All unique speakers in batch -> no positive pairs -> loss is 0."""
    torch.manual_seed(0)
    emb = torch.randn(4, 8)
    spk = torch.arange(4, dtype=torch.long)
    loss = slot_infonce(emb, spk).item()
    assert loss == 0.0


def test_infonce_partial_positives_is_positive() -> None:
    """Some same-speaker pairs in batch with random embeddings -> loss > 0."""
    torch.manual_seed(0)
    emb = torch.randn(8, 16)
    spk = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    loss = slot_infonce(emb, spk).item()
    assert loss > 0.0


def test_infonce_differentiable() -> None:
    torch.manual_seed(0)
    emb = torch.randn(6, 8, requires_grad=True)
    spk = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    loss = slot_infonce(emb, spk)
    loss.backward()
    assert emb.grad is not None
    assert torch.isfinite(emb.grad).all()


def test_infonce_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="2D"):
        slot_infonce(torch.zeros(2, 3, 4), torch.zeros(2, dtype=torch.long))


def test_infonce_rejects_batch_mismatch() -> None:
    with pytest.raises(ValueError, match="batch mismatch"):
        slot_infonce(torch.zeros(4, 8), torch.zeros(3, dtype=torch.long))


# -------------------------- ASD BCE ----------------------------------------


def test_asd_bce_shape_and_finite() -> None:
    logits = torch.randn(2, 5, 4)
    target = torch.zeros(2, 5, dtype=torch.long)
    loss = asd_bce(logits, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_asd_bce_perfect_prediction_is_low() -> None:
    """Logits strongly favoring the correct slot -> low loss."""
    target = torch.zeros(1, 3, dtype=torch.long)
    logits = torch.zeros(1, 3, 4)
    # Put very-high logit on correct slot (index 0), very-low on others.
    logits[0, :, 0] = 10.0
    logits[0, :, 1:] = -10.0
    loss = asd_bce(logits, target).item()
    assert loss < 0.01


def test_asd_bce_wrong_prediction_is_high() -> None:
    target = torch.zeros(1, 3, dtype=torch.long)
    logits = torch.zeros(1, 3, 4)
    logits[0, :, 0] = -10.0  # wrong: low on correct slot
    logits[0, :, 1:] = 10.0
    loss = asd_bce(logits, target).item()
    assert loss > 5.0


def test_asd_bce_differentiable() -> None:
    logits = torch.randn(1, 4, 3, requires_grad=True)
    target = torch.zeros(1, 4, dtype=torch.long)
    asd_bce(logits, target).backward()
    assert logits.grad is not None


def test_asd_bce_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="3D"):
        asd_bce(torch.zeros(2, 5), torch.zeros(2, dtype=torch.long))


# -------------------------- slot consistency ------------------------------


def test_consistency_identical_slots_is_zero() -> None:
    s = torch.randn(2, 4, 16)
    assert slot_consistency(s, s).item() == 0.0


def test_consistency_different_slots_is_positive() -> None:
    a = torch.randn(2, 4, 16)
    b = torch.randn(2, 4, 16)
    assert slot_consistency(a, b).item() > 0.0


def test_consistency_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        slot_consistency(torch.zeros(2, 4, 16), torch.zeros(2, 4, 8))
