"""Loss functions: SI-SNR, InfoNCE, ASD BCE, slot consistency, mag-STFT, mel."""

from nanotse.losses.asd_bce import asd_bce
from nanotse.losses.consistency import slot_consistency
from nanotse.losses.infonce import slot_infonce
from nanotse.losses.mag_stft import DEFAULT_RESOLUTIONS, multi_res_mag_stft
from nanotse.losses.mel_loss import DEFAULT_MEL_RESOLUTIONS, multi_res_mel_loss
from nanotse.losses.si_snr import EPS, negative_si_snr, si_snr

__all__ = [
    "DEFAULT_MEL_RESOLUTIONS",
    "DEFAULT_RESOLUTIONS",
    "EPS",
    "asd_bce",
    "multi_res_mag_stft",
    "multi_res_mel_loss",
    "negative_si_snr",
    "si_snr",
    "slot_consistency",
    "slot_infonce",
]
