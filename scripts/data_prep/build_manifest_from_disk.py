#!/usr/bin/env python3
"""Build manifest.json from an already-extracted ``<root>/audio/<spk>/<clip>.wav`` tree.

Useful after a streaming fetch interrupts mid-tar: salvages the partial
extraction without re-downloading. Splits speakers deterministically
into 80/20 train/val (disjoint, same convention as the streaming script).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="dataset root (must contain audio/)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2)
    args = p.parse_args(argv)

    audio_root = args.root / "audio"
    if not audio_root.exists():
        print(f"ERROR: {audio_root} not found", file=p.exit_on_error and __import__("sys").stderr)
        return 1

    speakers = sorted([d.name for d in audio_root.iterdir() if d.is_dir()])
    if not speakers:
        print(f"ERROR: no speaker directories under {audio_root}")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    n_val = max(1, int(len(speakers) * args.val_frac))
    n_train = len(speakers) - n_val
    train_spk = speakers[:n_train]
    val_spk = speakers[n_train:]
    assert not (set(train_spk) & set(val_spk))

    def rows(speakers_list: list[str]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for spk in speakers_list:
            for wav in sorted((audio_root / spk).glob("*.wav")):
                out.append({"speaker_id": spk, "wav": f"audio/{spk}/{wav.name}"})
        return out

    manifest = {"train": rows(train_spk), "val": rows(val_spk)}
    out_path = args.root / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"wrote {out_path}: "
        f"{len(manifest['train'])} train clips ({len(train_spk)} speakers), "
        f"{len(manifest['val'])} val clips ({len(val_spk)} speakers)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
