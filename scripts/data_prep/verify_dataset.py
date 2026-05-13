#!/usr/bin/env python3
"""Quick-quality verification on the extracted v2 dataset.

Checks (random sample, fast):
  - manifest.json: no train/val speaker overlap, all wav paths exist
  - audio: 100 random wavs read back successfully at 16 kHz, length > 0
  - faces: 100 random .npz files load with expected shape + face_ok ratio
  - mouth-ROI quality: distribution of face_ok ratios across the sample
  - face cache coverage: what fraction of train speakers have ≥1 face .npz
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/v2"))
    p.add_argument("--sample-n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--strict-fail-threshold",
        type=float,
        default=0.05,
        help="fail if >5% of sampled files broken",
    )
    args = p.parse_args(argv)
    rng = random.Random(args.seed)

    print(f"=== Verifying {args.root}")
    manifest_path = args.root / "manifest.json"
    if not manifest_path.exists():
        print(f"FAIL: {manifest_path} missing")
        return 1

    manifest = json.loads(manifest_path.read_text())
    train_spk = {r["speaker_id"] for r in manifest.get("train", [])}
    val_spk = {r["speaker_id"] for r in manifest.get("val", [])}
    overlap = train_spk & val_spk
    print(f"\n[manifest]")
    print(f"  train clips: {len(manifest.get('train', []))} ({len(train_spk)} spk)")
    print(f"  val   clips: {len(manifest.get('val', []))} ({len(val_spk)} spk)")
    print(f"  train/val speaker overlap: {len(overlap)}  {'FAIL' if overlap else 'OK'}")

    print(f"\n[audio sample of {args.sample_n}]")
    all_rows = manifest.get("train", []) + manifest.get("val", [])
    audio_sample = rng.sample(all_rows, min(args.sample_n, len(all_rows)))
    audio_ok = audio_fail = 0
    audio_durs: list[float] = []
    audio_srs: set[int] = set()
    for row in audio_sample:
        wav = args.root / row["wav"]
        if not wav.exists():
            audio_fail += 1
            continue
        try:
            arr, sr = sf.read(wav)
            if arr.shape[0] == 0:
                audio_fail += 1
                continue
            audio_ok += 1
            audio_durs.append(arr.shape[0] / sr)
            audio_srs.add(sr)
        except Exception as e:
            print(f"    bad wav {wav}: {e}")
            audio_fail += 1
    print(f"  ok: {audio_ok} / {audio_ok + audio_fail}")
    print(f"  sample rates seen: {sorted(audio_srs)}  {'OK' if audio_srs == {16000} else 'NOTE'}")
    if audio_durs:
        a = sorted(audio_durs)
        print(f"  duration sec: min={a[0]:.2f} median={a[len(a) // 2]:.2f} max={a[-1]:.2f}")

    print(f"\n[face cache]")
    faces_root = args.root / "faces"
    face_npz = list(faces_root.glob("*/*.npz")) if faces_root.exists() else []
    face_speakers = {p.parent.name for p in face_npz}
    cache_cov = len(face_speakers & train_spk) / max(1, len(train_spk))
    print(f"  total .npz: {len(face_npz)}")
    print(f"  speakers with face: {len(face_speakers)}")
    print(f"  train-speaker face coverage: {cache_cov * 100:.1f}%")

    print(f"\n[face sample of {min(args.sample_n, len(face_npz))}]")
    face_sample = rng.sample(face_npz, min(args.sample_n, len(face_npz))) if face_npz else []
    face_ok = face_fail = 0
    face_ok_ratios: list[float] = []
    face_n_frames: list[int] = []
    face_zeroish: list[bool] = []
    for npz_path in face_sample:
        try:
            data = np.load(npz_path)
            frames = data["frames"]
            face_ok_arr = data["face_ok"]
            if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
                face_fail += 1
                continue
            face_ok += 1
            face_n_frames.append(frames.shape[0])
            face_ok_ratios.append(float(face_ok_arr.mean()) if face_ok_arr.size else 0.0)
            face_zeroish.append(bool(frames.mean() < 5.0))
        except Exception as e:
            print(f"    bad npz {npz_path}: {e}")
            face_fail += 1
    print(f"  load ok: {face_ok} / {face_ok + face_fail}")
    if face_ok_ratios:
        r = sorted(face_ok_ratios)
        print(f"  face_ok ratio (frames with detected face):")
        print(
            f"    min={r[0]:.2f}  p25={r[len(r) // 4]:.2f}  median={r[len(r) // 2]:.2f}  p75={r[3 * len(r) // 4]:.2f}  max={r[-1]:.2f}"
        )
        useful = sum(1 for x in r if x > 0.5) / len(r)
        print(f"    fraction with >50% face_ok: {useful * 100:.1f}%")
        n = sorted(face_n_frames)
        print(f"  frames per clip: min={n[0]} median={n[len(n) // 2]} max={n[-1]}")
        zero_frac = sum(face_zeroish) / len(face_zeroish)
        print(f"  near-zero (mostly-black) clips: {zero_frac * 100:.1f}%")

    print("\n[verdict]")
    failures = []
    if overlap:
        failures.append(f"speaker-overlap: {len(overlap)}")
    if audio_fail / max(1, audio_ok + audio_fail) > args.strict_fail_threshold:
        failures.append(f"audio-fail-rate: {audio_fail / (audio_ok + audio_fail) * 100:.1f}%")
    if face_npz and face_fail / max(1, face_ok + face_fail) > args.strict_fail_threshold:
        failures.append(f"face-fail-rate: {face_fail / (face_ok + face_fail) * 100:.1f}%")
    if face_ok_ratios and (sum(1 for x in face_ok_ratios if x > 0.5) / len(face_ok_ratios)) < 0.5:
        failures.append(
            f"low-face-detection-rate: {sum(1 for x in face_ok_ratios if x > 0.5) / len(face_ok_ratios) * 100:.1f}% above 50% face_ok"
        )

    if failures:
        print(f"  FAIL: {failures}")
        return 1
    print("  PASS: dataset looks usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
