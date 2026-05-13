"""Multi-resolution magnitude-STFT loss.

The standard "free +0.5-1.0 dB" auxiliary for time-domain separation: pair the
SI-SNR loss with a spectral-magnitude criterion so the model gets gradient
signal on both the waveform and its frequency content.

Two components, summed at each STFT resolution:

* **Spectral convergence**: ``||(|S_est| - |S_tgt|)||_F / ||S_tgt||_F``.
  Normalised Frobenius error -- emphasises absolute spectral shape.
* **Log-magnitude L1**:      ``||log(|S_est| + eps) - log(|S_tgt| + eps)||_1``.
  Equal weight across frequencies/magnitudes regardless of energy --
  emphasises perceptual fidelity (especially low-energy speech tails).

Multi-resolution: we run the pair at several ``(n_fft, hop, win)`` settings
because no single STFT trades time-vs-frequency optimally. Common ESPnet /
Parallel WaveGAN choice: ``[(512, 50, 240), (1024, 120, 600), (2048, 240, 1200)]``
on 16 kHz -- short windows catch transients, long windows catch pitch detail.

Returns a single scalar averaged over (B, resolutions). No learned parameters.
"""

from __future__ import annotations

import torch

# Resolutions tuned for 16 kHz speech (the project's only supported sample rate).
DEFAULT_RESOLUTIONS: tuple[tuple[int, int, int], ...] = (
    (512, 50, 240),
    (1024, 120, 600),
    (2048, 240, 1200),
)

_LOG_EPS = 1e-7


def _stft_magnitude(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    """``(B, T) -> (B, F, T')`` magnitude spectrogram. Hann window, center=True."""
    window = torch.hann_window(win, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=True,
        return_complex=True,
        normalized=False,
    )
    return spec.abs()


def _single_resolution_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    n_fft: int,
    hop: int,
    win: int,
) -> torch.Tensor:
    """Spectral-convergence + log-magnitude L1 at one resolution."""
    mag_est = _stft_magnitude(estimate, n_fft, hop, win)
    mag_tgt = _stft_magnitude(target, n_fft, hop, win)
    # Spectral convergence -- normalised Frobenius
    sc = torch.linalg.norm(mag_est - mag_tgt, ord="fro", dim=(-2, -1)) / (
        torch.linalg.norm(mag_tgt, ord="fro", dim=(-2, -1)) + _LOG_EPS
    )
    # Log-magnitude L1
    log_mag_l1 = (
        torch.log(mag_est.clamp_min(_LOG_EPS)) - torch.log(mag_tgt.clamp_min(_LOG_EPS))
    ).abs().mean(dim=(-2, -1))
    return (sc + log_mag_l1).mean()


def multi_res_mag_stft(
    estimate: torch.Tensor,
    target: torch.Tensor,
    resolutions: tuple[tuple[int, int, int], ...] = DEFAULT_RESOLUTIONS,
) -> torch.Tensor:
    """Mean magnitude-STFT loss across ``resolutions``. Lower is better.

    Args:
        estimate:    ``(B, T)`` predicted waveform.
        target:      ``(B, T)`` ground-truth waveform.
        resolutions: list of ``(n_fft, hop_length, win_length)`` triples.

    Returns:
        Scalar tensor on the estimate's device.
    """
    losses = [
        _single_resolution_loss(estimate, target, n_fft, hop, win)
        for (n_fft, hop, win) in resolutions
    ]
    return torch.stack(losses).mean()
