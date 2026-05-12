"""Shape + streaming-equivalence checks for DualCacheFusion."""

from __future__ import annotations

import pytest
import torch

from nanotse.models.fusion.dual_cache import DualCacheFusion


def test_dual_cache_shape() -> None:
    f = DualCacheFusion(d_model=256, d_visual=512)
    audio = torch.randn(2, 100, 256)
    visual = torch.randn(2, 25, 512)
    out = f(audio, visual)
    assert out.shape == audio.shape


def test_dual_cache_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        DualCacheFusion(d_model=129, n_heads=4)


def test_dual_cache_oneshot_equivalence_to_offline() -> None:
    """forward_chunk on full (audio, visual) with empty cache == forward."""
    torch.manual_seed(0)
    f = DualCacheFusion(d_model=64, d_visual=128, n_heads=4, cache_len=200)
    f.eval()
    audio = torch.randn(1, 40, 64)
    visual = torch.randn(1, 10, 128)
    with torch.no_grad():
        y_offline = f(audio, visual)
        state = f.init_state(1, audio.device)
        y_stream, _ = f.forward_chunk(audio, visual, state)
    assert torch.allclose(y_offline, y_stream, atol=1e-5)


def test_dual_cache_chunked_audio_with_full_visual_upfront() -> None:
    """Audio split into chunks; all visual delivered in chunk 1.

    Equivalent to one-shot because every audio query attends to the same
    visual KV regardless of which chunk it lives in.
    """
    torch.manual_seed(0)
    f = DualCacheFusion(d_model=64, d_visual=128, n_heads=4, cache_len=200)
    f.eval()
    audio = torch.randn(1, 12, 64)
    visual = torch.randn(1, 4, 128)
    empty_visual = torch.zeros(1, 0, 128)

    with torch.no_grad():
        state1 = f.init_state(1, audio.device)
        y_oneshot, _ = f.forward_chunk(audio, visual, state1)

        state2 = f.init_state(1, audio.device)
        chunks = list(audio.split(3, dim=1))
        ys: list[torch.Tensor] = []
        # First call: all visual goes into the cache.
        y_first, state2 = f.forward_chunk(chunks[0], visual, state2)
        ys.append(y_first)
        # Subsequent calls: cache already holds the visual.
        for c in chunks[1:]:
            y_c, state2 = f.forward_chunk(c, empty_visual, state2)
            ys.append(y_c)
        y_chunked = torch.cat(ys, dim=1)
    assert torch.allclose(y_oneshot, y_chunked, atol=1e-5)


def test_dual_cache_progressive_visual_is_more_causal_than_oneshot() -> None:
    """Streaming with progressive visual = each audio chunk only sees visual delivered so far.

    This MUST differ from one-shot (which sees all visual up front). The test
    documents that contract so we don't accidentally regress to non-causal.
    """
    torch.manual_seed(0)
    f = DualCacheFusion(d_model=64, d_visual=128, n_heads=4, cache_len=200)
    f.eval()
    audio = torch.randn(1, 12, 64)
    visual = torch.randn(1, 4, 128)
    with torch.no_grad():
        state = f.init_state(1, audio.device)
        y_oneshot, _ = f.forward_chunk(audio, visual, state)

        state2 = f.init_state(1, audio.device)
        chunks_a = list(audio.split(3, dim=1))
        chunks_v = list(visual.split(1, dim=1))
        ys: list[torch.Tensor] = []
        for ca, cv in zip(chunks_a, chunks_v, strict=True):
            y_c, state2 = f.forward_chunk(ca, cv, state2)
            ys.append(y_c)
        y_progressive = torch.cat(ys, dim=1)
    # First 3 audio frames in progressive saw only 1 visual frame; one-shot saw 4.
    assert not torch.allclose(y_oneshot[:, :3], y_progressive[:, :3], atol=1e-4)


def test_dual_cache_empty_visual_passes_through() -> None:
    """No new visual + empty cache should still produce finite audio output."""
    f = DualCacheFusion(d_model=64, d_visual=128, n_heads=4)
    f.eval()
    audio = torch.randn(1, 5, 64)
    visual = torch.zeros(1, 0, 128)  # empty along T
    state = f.init_state(1, audio.device)
    with torch.no_grad():
        out, _ = f.forward_chunk(audio, visual, state)
    assert out.shape == audio.shape
    assert torch.isfinite(out).all()


def test_dual_cache_truncates_long_cache() -> None:
    f = DualCacheFusion(d_model=32, d_visual=64, n_heads=4, cache_len=5)
    state = f.init_state(1, torch.device("cpu"))
    audio = torch.randn(1, 4, 32)
    visual = torch.randn(1, 10, 64)  # 10 > cache_len=5
    _, new_state = f.forward_chunk(audio, visual, state)
    k, _ = new_state
    assert k.shape[2] == 5
