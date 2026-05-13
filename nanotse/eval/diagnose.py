"""Diagnose a NanoTSE checkpoint: save mix / target / estimate wavs + SI-SDRi.

Run after a training session to *listen* to what the model produces. Output
directory ends up looking like:

    runs/<ts>/diagnose/
        00_mix.wav        # what the mic picked up (two voices + scaling)
        00_target.wav     # the speaker we want
        00_estimate.wav   # what the model recovered
        00_interferer.wav # the speaker we want to suppress
        01_mix.wav
        01_target.wav
        ...
        diagnose.json     # per-clip + average SI-SDR baseline / final / SDRi

Pairs with ``scripts/diagnose.py`` (CLI). Uses the val split if a real
``data/smoke/manifest.json`` exists, else the synthetic dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from torch.utils.data import Dataset

from nanotse.data import AVMixSample, SyntheticAVMixDataset, VoxCeleb2MixDataset
from nanotse.losses import si_snr
from nanotse.models.nanotse import NanoTSE


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _build_dataset(
    root: Path,
    clip_seconds: float,
    num_items: int,
    sample_rate: int = 16000,
) -> Dataset[AVMixSample]:
    """Real loader if `manifest.json` is present; else synthetic."""
    manifest = root / "manifest.json"
    if manifest.exists():
        ds: Dataset[AVMixSample] = VoxCeleb2MixDataset(
            root,
            split="val",
            clip_seconds=clip_seconds,
            sample_rate=sample_rate,
            num_items=num_items,
            seed=42,
        )
        return ds
    return SyntheticAVMixDataset(
        num_clips=num_items,
        clip_seconds=clip_seconds,
        sample_rate=sample_rate,
        seed=42,
    )


def _infer_with_visual(state_dict: dict[str, Any]) -> bool:
    """Detect whether the saved checkpoint was AV or audio-only."""
    return any(k.startswith("visual_frontend.") for k in state_dict)


def _infer_with_enrollment(state_dict: dict[str, Any]) -> bool:
    """Detect whether the saved checkpoint had the enrollment encoder."""
    return any(k.startswith("enrollment_encoder.") for k in state_dict)


def diagnose(
    ckpt_path: Path,
    out_dir: Path,
    num_clips: int = 6,
    clip_seconds: float = 4.0,
    sample_rate: int = 16000,
    device: str | torch.device = "auto",
    data_root: Path | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load ``ckpt_path``, run on ``num_clips`` items, save wavs + diagnose.json.

    Returns the summary dict (same content as ``diagnose.json``).

    Args:
        model_kwargs: explicit NanoTSE constructor kwargs. Resolution order:
            (1) this arg, (2) ``ckpt["model_kwargs"]`` if present,
            (3) inferred defaults (``with_visual`` auto-detected from
            state-dict keys; everything else uses NanoTSE defaults).
    """
    dev = _resolve_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    if model_kwargs is None:
        if isinstance(ckpt, dict) and "model_kwargs" in ckpt:
            model_kwargs = dict(ckpt["model_kwargs"])
        else:
            model_kwargs = {
                "with_visual": _infer_with_visual(state_dict),
                "with_enrollment": _infer_with_enrollment(state_dict),
            }

    has_visual = bool(model_kwargs.get("with_visual", False))
    has_enrollment = bool(model_kwargs.get("with_enrollment", False))
    model = NanoTSE(**model_kwargs)
    model.load_state_dict(state_dict)
    model = model.to(dev).eval()

    data_root = data_root or Path("data/smoke")
    ds = _build_dataset(
        data_root, clip_seconds=clip_seconds, num_items=num_clips, sample_rate=sample_rate
    )

    per_clip: list[dict[str, float | int]] = []
    with torch.no_grad():
        for i in range(num_clips):
            s: AVMixSample = ds[i]
            mix = s["mix"].unsqueeze(0).to(dev)
            target = s["target"].unsqueeze(0).to(dev)
            interferer = s["interferer"].unsqueeze(0).to(dev)

            video = s["face"].unsqueeze(0).to(dev) if has_visual else None
            enroll = s["enrollment"].unsqueeze(0).to(dev) if has_enrollment else None
            out = model(mix, video, enroll)
            est = out[0] if isinstance(out, tuple) else out

            baseline_db = float(si_snr(mix, target).mean().item())
            est_db = float(si_snr(est, target).mean().item())

            mix_np = mix.squeeze(0).cpu().numpy()
            tgt_np = target.squeeze(0).cpu().numpy()
            int_np = interferer.squeeze(0).cpu().numpy()
            est_np = est.squeeze(0).cpu().numpy()

            sf.write(out_dir / f"{i:02d}_mix.wav", mix_np, sample_rate)
            sf.write(out_dir / f"{i:02d}_target.wav", tgt_np, sample_rate)
            sf.write(out_dir / f"{i:02d}_interferer.wav", int_np, sample_rate)
            sf.write(out_dir / f"{i:02d}_estimate.wav", est_np, sample_rate)

            per_clip.append(
                {
                    "clip": i,
                    "baseline_si_snr_db": round(baseline_db, 3),
                    "estimate_si_snr_db": round(est_db, 3),
                    "si_sdri_db": round(est_db - baseline_db, 3),
                }
            )

    avg_sdri = sum(float(m["si_sdri_db"]) for m in per_clip) / len(per_clip)
    avg_baseline = sum(float(m["baseline_si_snr_db"]) for m in per_clip) / len(per_clip)
    avg_estimate = sum(float(m["estimate_si_snr_db"]) for m in per_clip) / len(per_clip)
    summary: dict[str, Any] = {
        "ckpt": str(ckpt_path),
        "device": str(dev),
        "num_clips": num_clips,
        "clip_seconds": clip_seconds,
        "sample_rate": sample_rate,
        "has_visual": has_visual,
        "average_baseline_si_snr_db": round(avg_baseline, 3),
        "average_estimate_si_snr_db": round(avg_estimate, 3),
        "average_si_sdri_db": round(avg_sdri, 3),
        "per_clip": per_clip,
    }
    (out_dir / "diagnose.json").write_text(json.dumps(summary, indent=2))
    return summary
