"""MeMo baseline checks: shape contracts + bank state + differentiability."""

from __future__ import annotations

import torch

from nanotse.models.baselines.memo import ContextBank, MeMoBaseline, SpeakerBank


def test_memo_audio_only_forward_shape() -> None:
    m = MeMoBaseline(with_visual=False)
    audio = torch.randn(2, 16000)
    out = m(audio)
    assert out.shape == (2, 16000)


def test_memo_av_forward_shape() -> None:
    m = MeMoBaseline(
        d_model=64,
        n_heads=2,
        n_layers=1,
        d_visual=64,
        d_spk=48,
        d_ctx=64,
        frame_size=32,
        with_visual=True,
    )
    audio = torch.randn(1, 16000)
    video = torch.randint(0, 256, (1, 25, 32, 32, 3), dtype=torch.uint8)
    out = m(audio, video)
    assert out.shape == (1, 16000)


def test_speaker_bank_push_retrieve_shape() -> None:
    sb = SpeakerBank(d_input=64, d_spk=32)
    feat = torch.randn(2, 10, 64)
    slot = sb.push(feat)
    assert slot.shape == (2, 32)
    out = sb.retrieve(slot, 2, 10, 64, torch.device("cpu"))
    assert out.shape == (2, 10, 64)


def test_speaker_bank_cold_start_returns_zeros() -> None:
    sb = SpeakerBank(d_input=64, d_spk=32)
    out = sb.retrieve(None, 2, 5, 64, torch.device("cpu"))
    assert out.shape == (2, 5, 64)
    assert torch.equal(out, torch.zeros_like(out))


def test_context_bank_push_retrieve_shape() -> None:
    cb = ContextBank(d_input=64, d_ctx=32)
    feat = torch.randn(2, 10, 64)
    slot = cb.push(feat)
    assert slot.shape == (2, 10, 32)
    out = cb.retrieve(slot, 2, 10, 64, torch.device("cpu"))
    assert out.shape == (2, 10, 64)


def test_context_bank_cold_start_returns_zeros() -> None:
    cb = ContextBank(d_input=64, d_ctx=32)
    out = cb.retrieve(None, 1, 7, 64, torch.device("cpu"))
    assert torch.equal(out, torch.zeros_like(out))


def test_memo_state_fills_after_forward_chunk() -> None:
    m = MeMoBaseline(d_model=64, n_heads=2, n_layers=1, d_spk=32, d_ctx=32, with_visual=False)
    audio = torch.randn(1, 1600)
    state = m.init_state(1, torch.device("cpu"))
    assert state.speaker_slot is None
    assert state.context_slot is None
    _, new_state = m.forward_chunk(audio, state)
    assert new_state.speaker_slot is not None
    assert new_state.context_slot is not None
    assert new_state.step == 1


def test_memo_chained_forward_chunk_uses_prior_banks() -> None:
    """Second chunk should use the bank state set up by the first chunk."""
    torch.manual_seed(0)
    m = MeMoBaseline(d_model=32, n_heads=2, n_layers=1, d_spk=16, d_ctx=16, with_visual=False)
    m.eval()
    x1 = torch.randn(1, 1600)
    x2 = torch.randn(1, 1600)
    with torch.no_grad():
        state = m.init_state(1, x1.device)
        _, state1 = m.forward_chunk(x1, state)
        _, state2 = m.forward_chunk(x2, state1)
    assert state2.step == 2
    # State 1's speaker_slot is non-zero; state 2's should differ (new chunk's features).
    assert state1.speaker_slot is not None
    assert state2.speaker_slot is not None
    assert not torch.equal(state1.speaker_slot, state2.speaker_slot)


def test_memo_differentiable() -> None:
    m = MeMoBaseline(d_model=32, n_heads=2, n_layers=1, d_spk=16, d_ctx=16, with_visual=False)
    audio = torch.randn(1, 1600, requires_grad=True)
    out = m(audio)
    out.sum().backward()
    assert audio.grad is not None
    assert torch.isfinite(audio.grad).all()
