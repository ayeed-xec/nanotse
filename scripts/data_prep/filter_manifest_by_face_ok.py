#!/usr/bin/env python3
"""Drop speakers from manifest.json whose face cache is too sparse.

For each unique speaker_id in train+val, walks the speaker's .npz files
under ``<root>/faces/<spk>/`` and computes the *frame-count-weighted* mean
``face_ok`` ratio. Speakers below ``--min-face-ok`` are removed entirely
from both splits; speakers without any face cache are also dropped (the
training-time fallback to zero frames is the exact noise source we are
trying to eliminate).

Writes a backup at ``manifest.json.pre_face_filter`` then rewrites
``manifest.json`` in place. Prints a before/after summary.

Why: 27% of v2 clips have <50% face_ok, which translates to a fraction of
training batches being effectively audio-only. Removing those clips
tightens the per-batch gradient signal -- the high-variance source we
isolated at the +0.18 dB val plateau in 3060_v2_fresh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _speaker_face_quality(faces_root: Path, spk: str) -> tuple[float, int]:
    """Returns ``(mean_face_ok, total_frames)`` for one speaker, ``(0.0, 0)`` if missing."""
    spk_dir = faces_root / spk
    if not spk_dir.exists():
        return 0.0, 0
    weighted_ok = 0.0
    total_frames = 0
    for npz_path in spk_dir.glob("*.npz"):
        try:
            data = np.load(npz_path)
            face_ok = data["face_ok"]
        except (KeyError, OSError):
            continue
        if face_ok.size == 0:
            continue
        weighted_ok += float(face_ok.sum())
        total_frames += int(face_ok.size)
    if total_frames == 0:
        return 0.0, 0
    return weighted_ok / total_frames, total_frames


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/v2"))
    p.add_argument("--min-face-ok", type=float, default=0.5)
    args = p.parse_args(argv)

    manifest_path = args.root / "manifest.json"
    if not manifest_path.exists():
        print(f"FAIL: {manifest_path} missing")
        return 1
    manifest = json.loads(manifest_path.read_text())
    faces_root = args.root / "faces"

    speakers = sorted({r["speaker_id"] for split in ("train", "val") for r in manifest.get(split, [])})
    print(f"manifest speakers: {len(speakers)}  threshold={args.min_face_ok}")

    keep: set[str] = set()
    drop_no_cache = 0
    drop_below_threshold = 0
    quality_keep: list[tuple[str, float, int]] = []
    quality_drop: list[tuple[str, float, int]] = []
    for spk in speakers:
        q, n = _speaker_face_quality(faces_root, spk)
        if n == 0:
            drop_no_cache += 1
            quality_drop.append((spk, q, n))
            continue
        if q >= args.min_face_ok:
            keep.add(spk)
            quality_keep.append((spk, q, n))
        else:
            drop_below_threshold += 1
            quality_drop.append((spk, q, n))

    print(f"  keep: {len(keep)}  drop(no cache): {drop_no_cache}  drop(below threshold): {drop_below_threshold}")
    if quality_keep:
        qs = sorted(q for _, q, _ in quality_keep)
        print(
            f"  kept face_ok: min={qs[0]:.2f} p25={qs[len(qs) // 4]:.2f} median={qs[len(qs) // 2]:.2f} max={qs[-1]:.2f}"
        )
    if quality_drop:
        qs = sorted(q for _, q, _ in quality_drop)
        print(
            f"  dropped face_ok: min={qs[0]:.2f} median={qs[len(qs) // 2]:.2f} max={qs[-1]:.2f}"
        )

    new_manifest = {
        split: [r for r in manifest.get(split, []) if r["speaker_id"] in keep]
        for split in ("train", "val")
    }
    for split in ("train", "val"):
        before = len(manifest.get(split, []))
        after = len(new_manifest[split])
        spk_before = len({r["speaker_id"] for r in manifest.get(split, [])})
        spk_after = len({r["speaker_id"] for r in new_manifest[split]})
        print(f"  {split}: clips {before} -> {after}   speakers {spk_before} -> {spk_after}")

    overlap = {r["speaker_id"] for r in new_manifest["train"]} & {
        r["speaker_id"] for r in new_manifest["val"]
    }
    if overlap:
        print(f"FAIL: post-filter speaker overlap: {len(overlap)}")
        return 1

    backup = manifest_path.with_suffix(".json.pre_face_filter")
    if not backup.exists():
        backup.write_text(manifest_path.read_text())
        print(f"  backup -> {backup.name}")
    manifest_path.write_text(json.dumps(new_manifest, indent=2))
    print(f"  wrote -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
