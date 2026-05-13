#!/usr/bin/env python3
"""Extract mouth-ROI face frames from VoxCeleb2 mp4s using OpenCV Haar cascade.

Previous version used MediaPipe FaceMesh; that submodule was removed in
mediapipe 0.10.35. OpenCV's Haar cascade frontal-face detector ships with
``opencv-python`` (no model download), is fast on CPU, and is robust enough
on VoxCeleb-style mostly-frontal videos.

Per-clip pipeline:
  1. Decode every frame with cv2.VideoCapture.
  2. Detect the largest frontal face per frame; on miss, persist the last
     successful crop (face_ok flag tracks per-frame success).
  3. Take the bottom 50% of the detected face bbox (mouth region) with a
     small upward extension; resize to ``face_size x face_size`` uint8.
  4. Subsample frames to ``target_fps`` by sampling closest source indices.
  5. Save ``<root>/faces/<spk>/<vid>__<clip>.npz`` with ``frames`` (F,H,W,3)
     and ``face_ok`` (F,) bool.

Usage:
    python scripts/data_prep/extract_mouth_roi.py --root data/v2 --workers 4
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]


def _make_detector() -> cv2.CascadeClassifier:
    det = cv2.CascadeClassifier(_CASCADE_PATH)
    if det.empty():
        raise RuntimeError(f"failed to load haarcascade at {_CASCADE_PATH}")
    return det


def _lip_box_from_face(
    x: int, y: int, w: int, h: int, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Bottom 50 % of the face bbox, padded slightly and clipped to image bounds.

    Returns ``(x0, y0, x1, y1)`` integer corners.
    """
    mouth_h = int(h * 0.50)
    mouth_y0 = y + h - mouth_h
    # Extend slightly upward to include the upper lip region.
    mouth_y0 = max(0, mouth_y0 - int(h * 0.05))
    mouth_x0 = max(0, x - int(w * 0.05))
    mouth_x1 = min(img_w, x + w + int(w * 0.05))
    mouth_y1 = min(img_h, y + h + int(h * 0.05))
    return mouth_x0, mouth_y0, mouth_x1, mouth_y1


def _extract_lip_frames(mp4_path: Path, face_size: int, target_fps: int) -> dict[str, np.ndarray]:
    """Decode mp4, run haarcascade per sampled frame, return frames + face_ok."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(target_fps)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n_src == 0:
        cap.release()
        return {
            "frames": np.zeros((0, face_size, face_size, 3), dtype=np.uint8),
            "face_ok": np.zeros(0, dtype=bool),
        }

    duration = n_src / src_fps if src_fps > 0 else 0.0
    n_out = max(1, round(duration * target_fps))
    target_set = {min(n_src - 1, round(i * (n_src / n_out))) for i in range(n_out)}

    det = _make_detector()

    frames_out: list[np.ndarray] = []
    face_ok_out: list[bool] = []
    last_good = np.zeros((face_size, face_size, 3), dtype=np.uint8)

    src_index = 0
    while cap.isOpened():
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if src_index not in target_set:
            src_index += 1
            continue
        src_index += 1

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
        if len(faces) > 0:
            # Take the largest face (likely the speaker).
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            ih, iw = frame_bgr.shape[:2]
            x0, y0, x1, y1 = _lip_box_from_face(int(x), int(y), int(w), int(h), iw, ih)
            if x1 > x0 and y1 > y0:
                crop_bgr = frame_bgr[y0:y1, x0:x1]
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(crop_rgb, (face_size, face_size), interpolation=cv2.INTER_AREA)
                last_good = resized
                frames_out.append(resized)
                face_ok_out.append(True)
                continue
        # No face this frame -> persist last good (zeros at cold start).
        frames_out.append(last_good)
        face_ok_out.append(False)

    cap.release()
    arr = (
        np.stack(frames_out, axis=0)
        if frames_out
        else np.zeros((0, face_size, face_size, 3), dtype=np.uint8)
    )
    return {"frames": arr, "face_ok": np.asarray(face_ok_out, dtype=bool)}


def _process_one(args: tuple[Path, Path, int, int]) -> tuple[Path, int, int]:
    """Worker: ``(mp4, out_npz, face_size, fps) -> (mp4, n_frames, n_face_ok)``.

    Returns ``(_, -1, -1)`` if the output already exists (skipped).
    """
    mp4, out_npz, face_size, target_fps = args
    if out_npz.exists():
        return mp4, -1, -1
    try:
        data = _extract_lip_frames(mp4, face_size, target_fps)
    except Exception as e:
        print(f"  ERROR {mp4}: {e}", file=sys.stderr)
        return mp4, 0, 0
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, frames=data["frames"], face_ok=data["face_ok"])
    return mp4, int(data["frames"].shape[0]), int(data["face_ok"].sum())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", type=Path, default=Path("data/v2"), help="dataset root (must contain video/)"
    )
    p.add_argument("--face-size", type=int, default=112)
    p.add_argument("--fps", type=int, default=25, help="target fps after subsampling")
    p.add_argument("--workers", type=int, default=1, help="parallel processes (CPU only)")
    p.add_argument("--max-clips", type=int, default=0, help="cap total clips (0=all)")
    args = p.parse_args(argv)

    video_dir = args.root / "video"
    faces_dir = args.root / "faces"
    if not video_dir.exists():
        print(f"ERROR: {video_dir} not found", file=sys.stderr)
        return 1
    faces_dir.mkdir(exist_ok=True)

    jobs: list[tuple[Path, Path, int, int]] = []
    for mp4 in sorted(video_dir.glob("*/*.mp4")):
        spk = mp4.parent.name
        out_npz = faces_dir / spk / f"{mp4.stem}.npz"
        jobs.append((mp4, out_npz, args.face_size, args.fps))
        if args.max_clips and len(jobs) >= args.max_clips:
            break
    if not jobs:
        print(f"no mp4 files found under {video_dir}", file=sys.stderr)
        return 2

    print(
        f"extracting mouth-ROI for {len(jobs)} clip(s) @ {args.fps} fps, "
        f"{args.face_size}x{args.face_size}, workers={args.workers}"
    )

    done = skipped = total_face_ok = total_frames = 0
    if args.workers <= 1:
        for job in jobs:
            mp4, n, ok = _process_one(job)
            if n < 0:
                skipped += 1
                continue
            done += 1
            total_face_ok += ok
            total_frames += n
            if done % 50 == 0:
                print(
                    f"  [{done + skipped}/{len(jobs)}] last={mp4.name}: {n} frames, {ok} with face",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_process_one, j) for j in jobs]
            for fut in as_completed(futures):
                mp4, n, ok = fut.result()
                if n < 0:
                    skipped += 1
                    continue
                done += 1
                total_face_ok += ok
                total_frames += n
                if done % 50 == 0:
                    print(
                        f"  [{done + skipped}/{len(jobs)}] last={mp4.name}: {n} frames, {ok} with face",
                        flush=True,
                    )

    print(
        f"\ndone: {done} processed (+{skipped} skipped), "
        f"{total_frames:,} frames written, {total_face_ok:,} with face detected "
        f"({100.0 * total_face_ok / max(1, total_frames):.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
