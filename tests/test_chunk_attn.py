"""Shape, causal-mask, and streaming-equivalence checks for ChunkAttnBackbone."""

from __future__ import annotations

import pytest
import torch

from nanotse.models.backbones.chunk_attn import ChunkAttnBackbone


def test_chunk_attn_shape() -> None:
    m = ChunkAttnBackbone(d_model=128, n_heads=4, n_layers=2, cache_len=50)
    x = torch.randn(2, 20, 128)
    y = m(x)
    assert y.shape == x.shape


def test_chunk_attn_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        ChunkAttnBackbone(d_model=129, n_heads=4)


def test_chunk_attn_init_state_empty_cache() -> None:
    m = ChunkAttnBackbone(d_model=64, n_heads=4, n_layers=3, cache_len=10)
    state = m.init_state(batch_size=2, device=torch.device("cpu"))
    assert len(state) == 3
    for k, v in state:
        assert k.shape == (2, 4, 0, 16)
        assert v.shape == (2, 4, 0, 16)


def test_chunk_attn_oneshot_equivalence_to_offline() -> None:
    """forward_chunk on the full sequence (empty cache) == forward(full sequence)."""
    torch.manual_seed(0)
    m = ChunkAttnBackbone(d_model=64, n_heads=4, n_layers=2, cache_len=100)
    m.eval()
    x = torch.randn(1, 50, 64)
    with torch.no_grad():
        y_offline = m(x)
        state = m.init_state(1, x.device)
        y_stream, _ = m.forward_chunk(x, state)
    assert torch.allclose(y_offline, y_stream, atol=1e-5)


def test_chunk_attn_chunked_equivalence_to_oneshot() -> None:
    """Splitting T into chunks and folding state == one-shot forward_chunk."""
    torch.manual_seed(0)
    m = ChunkAttnBackbone(d_model=64, n_heads=4, n_layers=2, cache_len=100)
    m.eval()
    x = torch.randn(1, 12, 64)
    with torch.no_grad():
        state1 = m.init_state(1, x.device)
        y_oneshot, _ = m.forward_chunk(x, state1)

        state2 = m.init_state(1, x.device)
        ys: list[torch.Tensor] = []
        for c in x.split(4, dim=1):
            y_c, state2 = m.forward_chunk(c, state2)
            ys.append(y_c)
        y_chunked = torch.cat(ys, dim=1)
    assert torch.allclose(y_oneshot, y_chunked, atol=1e-5)


def test_chunk_attn_is_causal() -> None:
    """Changing a future token must not change past outputs."""
    torch.manual_seed(0)
    m = ChunkAttnBackbone(d_model=64, n_heads=4, n_layers=2, cache_len=100)
    m.eval()
    x = torch.randn(1, 20, 64)
    x2 = x.clone()
    x2[0, 10:] = torch.randn(10, 64)  # perturb the future
    with torch.no_grad():
        y1 = m(x)
        y2 = m(x2)
    assert torch.allclose(y1[:, :10], y2[:, :10], atol=1e-5)


def test_chunk_attn_cache_truncation() -> None:
    """When cache + chunk exceeds cache_len, cache is truncated to cache_len."""
    m = ChunkAttnBackbone(d_model=32, n_heads=2, n_layers=1, cache_len=5)
    state = m.init_state(1, torch.device("cpu"))
    x = torch.randn(1, 10, 32)
    _, new_state = m.forward_chunk(x, state)
    k_cache, _ = new_state[0]
    assert k_cache.shape[2] == 5  # truncated
