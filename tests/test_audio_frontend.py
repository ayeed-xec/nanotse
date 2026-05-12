"""Shape + gradient checks for AudioFrontend."""

from __future__ import annotations

import pytest
import torch

from nanotse.models.frontends.audio_stft import AudioFrontend


def test_audio_frontend_default_shape() -> None:
    af = AudioFrontend()
    x = torch.randn(2, 16000)  # 1 s mono at 16 kHz
    y = af(x)
    assert y.shape == (2, 100, 256)


def test_audio_frontend_smoke_config_4s() -> None:
    """4 s at 16 kHz -> 400 frames at 100 Hz."""
    af = AudioFrontend()
    x = torch.randn(1, 16000 * 4)
    y = af(x)
    assert y.shape == (1, 400, 256)


def test_audio_frontend_custom_d_model() -> None:
    af = AudioFrontend(d_model=128)
    x = torch.randn(3, 16000)
    y = af(x)
    assert y.shape == (3, 100, 128)


def test_audio_frontend_rejects_bad_kernel_stride() -> None:
    with pytest.raises(ValueError, match="kernel - stride"):
        AudioFrontend(kernel=10, stride=3)


def test_audio_frontend_differentiable() -> None:
    af = AudioFrontend()
    x = torch.randn(1, 1600, requires_grad=True)
    af(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
