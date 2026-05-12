"""Shape, persistence, and differentiability checks for NamedSlotMemory."""

from __future__ import annotations

import pytest
import torch

from nanotse.models.memory.slot_attention import NamedSlotMemory


def test_slot_memory_shape() -> None:
    m = NamedSlotMemory(n_slots=16, d_input=64, d_slot=64)
    x = torch.randn(2, 20, 64)
    augmented, slots = m(x)
    assert augmented.shape == (2, 20, 128)  # (B, T, D + S)
    assert slots.shape == (2, 16, 64)


def test_slot_memory_rejects_invalid_n_slots() -> None:
    with pytest.raises(ValueError, match="n_slots must be > 0"):
        NamedSlotMemory(n_slots=0)


def test_slot_memory_rejects_invalid_n_iters() -> None:
    with pytest.raises(ValueError, match="n_iters must be > 0"):
        NamedSlotMemory(n_iters=0)


def test_slot_memory_slots_persist_in_state() -> None:
    """forward_chunk must return updated slots, distinct from slot_init."""
    torch.manual_seed(0)
    m = NamedSlotMemory(n_slots=4, d_input=32, d_slot=32, n_iters=3)
    state = m.init_state(1, torch.device("cpu"))
    x = torch.randn(1, 10, 32)
    (_, slots_after), new_state = m.forward_chunk(x, state)
    assert not torch.equal(slots_after, state["slots"])
    assert torch.equal(slots_after, new_state["slots"])
    assert new_state["step"] == 1


def test_slot_memory_streaming_chains_state() -> None:
    """Two consecutive forward_chunks: the second sees the slot bank from the first."""
    torch.manual_seed(0)
    m = NamedSlotMemory(n_slots=4, d_input=32, d_slot=32, n_iters=2)
    state = m.init_state(1, torch.device("cpu"))
    x1 = torch.randn(1, 5, 32)
    x2 = torch.randn(1, 5, 32)
    (_, slots1), state = m.forward_chunk(x1, state)
    (_, slots2), state = m.forward_chunk(x2, state)
    assert not torch.equal(slots1, slots2)
    assert state["step"] == 2


def test_slot_memory_differentiable() -> None:
    m = NamedSlotMemory(n_slots=8, d_input=32, d_slot=32)
    x = torch.randn(1, 10, 32, requires_grad=True)
    augmented, slots = m(x)
    (augmented.sum() + slots.sum()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for p in m.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
