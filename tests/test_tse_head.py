"""Shape + round-trip checks for the TSE head against AudioFrontend."""

from __future__ import annotations

import pytest
import torch

from nanotse.models.frontends.audio_stft import AudioFrontend
from nanotse.models.heads.tse import TSEHead


def test_tse_head_round_trips_with_audio_frontend() -> None:
    """T -> Ta -> T must give the same length when kernel/stride/padding match."""
    af = AudioFrontend()
    head = TSEHead()
    x = torch.randn(2, 16000)
    enc = af(x)
    # use enc as both features and encoder_out for the shape contract
    y = head(enc, enc)
    assert y.shape == (2, 16000)


def test_tse_head_output_is_finite() -> None:
    """Output must be finite for any reasonable input."""
    head = TSEHead(d_model=32)
    features = torch.randn(1, 10, 32)
    encoder_out = torch.randn(1, 10, 32)
    head.eval()
    with torch.no_grad():
        y = head(features, encoder_out)
    assert y.shape[0] == 1
    assert torch.isfinite(y).all()


def test_tse_head_rejects_bad_kernel_stride() -> None:
    with pytest.raises(ValueError, match="kernel - stride"):
        TSEHead(kernel=10, stride=3)


def test_tse_head_differentiable() -> None:
    head = TSEHead(d_model=64)
    features = torch.randn(1, 50, 64, requires_grad=True)
    encoder_out = torch.randn(1, 50, 64)
    head(features, encoder_out).sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
