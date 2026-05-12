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


def test_slot_memory_lru_updates_on_winner() -> None:
    """After forward_chunk, the dominant slot's LRU timestamp should equal new step."""
    torch.manual_seed(0)
    m = NamedSlotMemory(n_slots=4, d_input=16, d_slot=16, n_iters=2)
    state = m.init_state(1, torch.device("cpu"))
    x = torch.randn(1, 10, 16)
    (_, _), new_state = m.forward_chunk(x, state)
    # One slot must have LRU == new_state["step"]; the rest stay at 0.
    lru = new_state["lru"][0]
    assert (lru == new_state["step"]).sum().item() == 1
    assert (lru == 0).sum().item() == m.n_slots - 1


def test_slot_memory_evict_lru_resets_oldest() -> None:
    """evict_lru replaces the slot with the smallest LRU stamp; new slot equals slot_init."""
    torch.manual_seed(0)
    m = NamedSlotMemory(n_slots=4, d_input=16, d_slot=16, n_iters=1)
    # Build a state with hand-crafted LRU so eviction target is deterministic.
    state = m.init_state(1, torch.device("cpu"))
    # Hand-set: slot 0 has LRU=5, slot 1 has LRU=3, slot 2 has LRU=10, slot 3 has LRU=1 (oldest).
    state["lru"] = torch.tensor([[5.0, 3.0, 10.0, 1.0]])
    # Mutate slots so the difference between current state and slot_init is visible.
    state["slots"] = state["slots"] + 1.0  # all slots become slot_init + 1
    state["step"] = 11

    new_state = m.evict_lru(state)
    # Slot 3 should now equal slot_init[0, 3]; others unchanged.
    assert torch.allclose(new_state["slots"][0, 3], m.slot_init[0, 3])
    for i in (0, 1, 2):
        assert torch.allclose(new_state["slots"][0, i], state["slots"][0, i])
    # Newly-bound slot's LRU is bumped to the current step.
    assert new_state["lru"][0, 3].item() == 11.0


def test_slot_memory_differentiable() -> None:
    m = NamedSlotMemory(n_slots=8, d_input=32, d_slot=32)
    x = torch.randn(1, 10, 32, requires_grad=True)
    augmented, slots = m(x)
    (augmented.sum() + slots.sum()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for p in m.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
