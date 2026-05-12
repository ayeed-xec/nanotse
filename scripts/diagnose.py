#!/usr/bin/env python3
"""Diagnose a NanoTSE checkpoint: save mix / target / estimate wavs + SI-SDRi.

Listen to the wavs side-by-side to *hear* whether the model is actually
cleaning up the mixture. The JSON summary captures per-clip baseline,
estimate, and improvement (SI-SDRi).

Usage:
    # Default: writes wavs into runs/<ts>/diagnose/ next to the ckpt.
    python scripts/diagnose.py --ckpt runs/<ts>/model.pt

    # Explicit output dir + more clips.
    python scripts/diagnose.py --ckpt runs/<ts>/model.pt --out /tmp/listen --num-clips 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nanotse.eval.diagnose import diagnose


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path, help="path to model.pt")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output dir (default: <ckpt-dir>/diagnose/)",
    )
    p.add_argument("--num-clips", type=int, default=6)
    p.add_argument("--clip-seconds", type=float, default=4.0)
    p.add_argument("--device", default="auto", help="auto / cpu / mps / cuda")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/smoke"),
        help="root containing manifest.json + audio/ (default: data/smoke/)",
    )
    args = p.parse_args(argv)

    out_dir = args.out or (args.ckpt.parent / "diagnose")
    summary = diagnose(
        ckpt_path=args.ckpt,
        out_dir=out_dir,
        num_clips=args.num_clips,
        clip_seconds=args.clip_seconds,
        device=args.device,
        data_root=args.data_root,
    )

    print(f"\nsaved {summary['num_clips'] * 4} wavs + diagnose.json -> {out_dir}")
    print(f"device={summary['device']}  has_visual={summary['has_visual']}")
    print("\nper-clip:")
    for m in summary["per_clip"]:
        print(
            f"  clip {m['clip']:2d}:  baseline {m['baseline_si_snr_db']:+6.2f} dB"
            f"  ->  estimate {m['estimate_si_snr_db']:+6.2f} dB"
            f"  =  SDRi {m['si_sdri_db']:+6.2f} dB"
        )
    print(
        f"\nAVERAGE:  baseline {summary['average_baseline_si_snr_db']:+.2f} dB"
        f"  ->  estimate {summary['average_estimate_si_snr_db']:+.2f} dB"
        f"  =  SDRi {summary['average_si_sdri_db']:+.2f} dB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
