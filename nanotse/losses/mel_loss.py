"""Multi-resolution mel-spectrogram perceptual loss.

Companion to ``multi_res_mag_stft`` (linear-frequency magnitude). Mel scale is
perceptually weighted -- it emphasises lower frequencies (where speech formants
live) and de-emphasises high frequencies, matching human auditory sensitivity.

Standard recipe across HiFi-GAN, BigVGAN, NaturalSpeech, and most modern speech
synthesis/separation pipelines: log-mel L1 at 2-3 resolutions, summed and
averaged. Typically adds **+0.5 to +1.0 dB SDRi** on top of pure time-domain
+ linear-STFT losses.

Differences from ``multi_res_mag_stft``:

* Mel filterbank applied between |STFT| and log -- compresses 513 linear bins
  into 80 perceptual bins, so high-frequency noise contributes less.
* Mel filterbank precomputed once per device + dtype and cached on the module
  via ``register_buffer`` semantics (we replicate the buffer behaviour by hand
  since this is a pure-function module without parameters).

Returns a single scalar averaged over (B, resolutions). No learned weights.
"""

from __future__ import annotations

import math

import torch

# 16 kHz speech: short window catches transients (consonants), long window
# resolves pitch detail. Mel n_bins=80 is the de-facto standard.
DEFAULT_MEL_RESOLUTIONS: tuple[tuple[int, int, int, int], ...] = (
    # (n_fft, hop, win, n_mels)
    (1024, 120, 600, 80),
    (2048, 240, 1200, 80),
    (4096, 480, 2400, 80),
)

_LOG_EPS = 1e-7
_SAMPLE_RATE = 16000  # hard-coded; matches the rest of the project's contract


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an ``(n_mels, n_fft // 2 + 1)`` triangular mel filterbank.

    Mirrors librosa's default behaviour: equal area in mel space, peaks at the
    mel-band centres, zeros at the adjacent band edges.
    """
    if f_max is None:
        f_max = sr / 2
    n_bins = n_fft // 2 + 1
    # Mel-spaced bin centres including boundaries
    mel_points = torch.linspace(
        _hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2, device=device, dtype=dtype
    )
    hz_points = torch.tensor(
        [_mel_to_hz(m.item()) for m in mel_points], device=device, dtype=dtype
    )
    bin_freqs = torch.linspace(0, sr / 2, n_bins, device=device, dtype=dtype)

    fb = torch.zeros(n_mels, n_bins, device=device, dtype=dtype)
    for m in range(n_mels):
        left, center, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        up = (bin_freqs - left) / (center - left + _LOG_EPS)
        dn = (right - bin_freqs) / (right - center + _LOG_EPS)
        fb[m] = torch.clamp(torch.minimum(up, dn), min=0.0)
    return fb


# Cache mel filterbanks by (sr, n_fft, n_mels, device, dtype) to avoid rebuilding
# every call. Mel filterbanks are deterministic functions of these keys.
_FB_CACHE: dict[tuple, torch.Tensor] = {}


def _cached_mel_filterbank(
    sr: int, n_fft: int, n_mels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key = (sr, n_fft, n_mels, device.type, str(device), dtype)
    fb = _FB_CACHE.get(key)
    if fb is None:
        fb = _mel_filterbank(sr, n_fft, n_mels, device=device, dtype=dtype)
        _FB_CACHE[key] = fb
    return fb


def _log_mel(x: torch.Tensor, n_fft: int, hop: int, win: int, n_mels: int) -> torch.Tensor:
    """``(B, T) -> (B, n_mels, T')`` log-mel spectrogram on the same device."""
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
    mag = spec.abs()
    fb = _cached_mel_filterbank(_SAMPLE_RATE, n_fft, n_mels, x.device, x.dtype)
    mel = torch.matmul(fb, mag)  # (n_mels, F) @ (B, F, T') -> (B, n_mels, T')
    return torch.log(mel.clamp_min(_LOG_EPS))


def _single_resolution_mel_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    n_fft: int,
    hop: int,
    win: int,
    n_mels: int,
) -> torch.Tensor:
    """L1 distance between log-mel spectrograms at one resolution."""
    log_mel_est = _log_mel(estimate, n_fft, hop, win, n_mels)
    log_mel_tgt = _log_mel(target, n_fft, hop, win, n_mels)
    return (log_mel_est - log_mel_tgt).abs().mean()


def multi_res_mel_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    resolutions: tuple[tuple[int, int, int, int], ...] = DEFAULT_MEL_RESOLUTIONS,
) -> torch.Tensor:
    """Mean log-mel L1 loss across ``resolutions``. Lower is better.

    Args:
        estimate:    ``(B, T)`` predicted waveform (16 kHz).
        target:      ``(B, T)`` ground-truth waveform (16 kHz).
        resolutions: list of ``(n_fft, hop_length, win_length, n_mels)`` tuples.

    Returns:
        Scalar tensor on the estimate's device.
    """
    losses = [
        _single_resolution_mel_loss(estimate, target, n_fft, hop, win, n_mels)
        for (n_fft, hop, win, n_mels) in resolutions
    ]
    return torch.stack(losses).mean()
