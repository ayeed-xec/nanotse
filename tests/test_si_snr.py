"""Behavioural checks for the SI-SNR loss."""

from __future__ import annotations

import torch

from nanotse.losses import negative_si_snr, si_snr


def test_si_snr_perfect_match_is_high() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 16000)
    snr_db = si_snr(x, x).mean().item()
    assert snr_db > 60.0, f"expected high SI-SNR for identical signals, got {snr_db}"


def test_si_snr_is_scale_invariant() -> None:
    """At moderate SDRs, SI-SNR ignores the scale of the estimate."""
    torch.manual_seed(0)
    x = torch.randn(4, 16000)
    n = 0.1 * torch.randn(4, 16000)
    snr_a = si_snr(x + n, x).mean().item()
    snr_b = si_snr(2.5 * (x + n), x).mean().item()
    assert abs(snr_a - snr_b) < 0.5, f"scale-variance: {snr_a} vs {snr_b}"


def test_si_snr_is_sign_invariant() -> None:
    """SI-SDR's scale invariance includes negative scaling — sign drops out."""
    torch.manual_seed(0)
    x = torch.randn(4, 16000)
    n = 0.1 * torch.randn(4, 16000)
    snr_pos = si_snr(x + n, x).mean().item()
    snr_neg = si_snr(-(x + n), x).mean().item()
    assert abs(snr_pos - snr_neg) < 0.5, f"sign-variance: {snr_pos} vs {snr_neg}"


def test_si_snr_partial_noise_is_moderate() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 16000)
    n = torch.randn(4, 16000) * 0.1
    snr_db = si_snr(x + n, x).mean().item()
    # ~20 dB SNR mix should produce ~20 dB SI-SNR within a few dB.
    assert 10.0 < snr_db < 40.0, f"expected ~20 dB, got {snr_db}"


def test_negative_si_snr_is_scalar_and_differentiable() -> None:
    torch.manual_seed(0)
    est = torch.randn(2, 1000, requires_grad=True)
    ref = torch.randn(2, 1000)
    loss = negative_si_snr(est, ref)
    assert loss.ndim == 0
    loss.backward()
    assert est.grad is not None
    assert torch.isfinite(est.grad).all()


def test_si_snr_shape_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="shape mismatch"):
        si_snr(torch.zeros(4, 1000), torch.zeros(4, 999))
