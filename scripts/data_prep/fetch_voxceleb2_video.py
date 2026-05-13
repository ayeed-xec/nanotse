#!/usr/bin/env python3
"""Stream-download VoxCeleb2-mix ``orig_part_*`` videos with inline mouth-ROI extraction.

Disk-budget strategy (because the full video corpus is ~280 GB and we have
a 400 GB cap for the whole pipeline including audio):

1. Stream the tar via ``_ChainedHTTPStream`` (resume-capable).
2. For each mp4 whose speaker is in our audio manifest AND under the
   per-speaker cap, extract the file to a TEMP buffer (in-memory).
3. Run MediaPipe FaceMesh on the temp buffer; write the resulting
   ``<spk>/<vid>__<clip>.npz`` (lip-cropped 112x112 uint8 frames).
4. Discard the mp4 buffer immediately. Raw video is never persisted.

Net disk footprint: only the .npz face cache (~2.6 MB/clip compressed),
not the raw video (which would be ~3-5 MB/clip on disk + tar overhead).

Usage:
    python scripts/data_prep/fetch_voxceleb2_video.py \\
        --out data/v2 --all-parts --max-clips-per-speaker 5
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

# Mouth-ROI extraction.
from extract_mouth_roi import _extract_lip_frames

# Reuse the audio fetcher's chained-stream + speaker helpers + token handling.
from fetch_voxceleb2_mix_smoke import (
    HF_BASE,
    _ChainedHTTPStream,
    _hf_headers,
    _speaker_of,
)

ALL_VIDEO_PARTS = [f"orig_part_a{c}" for c in "abcdefg"]


def _video_id_of(path: str) -> str | None:
    """Extract ``<vid>`` from ``orig/<split>/<spk>/<vid>/<clip>.mp4``."""
    parts = [p for p in path.split("/") if p]
    if len(parts) < 3 or not path.lower().endswith(".mp4"):
        return None
    for i, p in enumerate(parts):
        if p.startswith("id") and len(p) > 2 and p[2:].isdigit() and i + 1 < len(parts):
            return parts[i + 1]
    return None


def cmd_fetch_inline(
    urls: list[str],
    out: Path,
    known_speakers: set[str],
    max_per_speaker: int,
    face_size: int,
    target_fps: int,
) -> int:
    """Stream tar, extract mp4 per matching speaker, run ROI inline, save .npz only."""
    faces_dir = out / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)

    per_speaker: dict[str, int] = {}
    bytes_streamed = 0
    last_report = 0
    extracted = 0
    failed = 0

    print(
        f"streaming {len(urls)} video part(s); "
        f"cap = {max_per_speaker if max_per_speaker else 'unlimited'} clips/speaker; "
        f"known speakers = {len(known_speakers)}",
        flush=True,
    )

    stream = _ChainedHTTPStream(urls, _hf_headers())
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            for member in tf:
                if not member.name.lower().endswith(".mp4") or not member.isfile():
                    continue
                spk = _speaker_of(member.name)
                if spk is None or spk not in known_speakers:
                    continue
                if max_per_speaker and per_speaker.get(spk, 0) >= max_per_speaker:
                    continue

                vid = _video_id_of(member.name) or "novid"
                clip = Path(member.name).name
                target_stem = f"{vid}__{Path(clip).stem}"
                spk_dir = faces_dir / spk
                spk_dir.mkdir(exist_ok=True)
                out_npz = spk_dir / f"{target_stem}.npz"
                if out_npz.exists():
                    per_speaker[spk] = per_speaker.get(spk, 0) + 1
                    continue

                buf = tf.extractfile(member)
                if buf is None:
                    continue
                data = buf.read()
                bytes_streamed += len(data)

                # Inline extraction: write mp4 to a temp file (OpenCV needs a path),
                # run FaceMesh, delete the temp file.
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = Path(tmp.name)
                try:
                    frames = _extract_lip_frames(tmp_path, face_size, target_fps)
                    np.savez_compressed(out_npz, frames=frames["frames"], face_ok=frames["face_ok"])
                    extracted += 1
                    per_speaker[spk] = per_speaker.get(spk, 0) + 1
                except Exception as e:
                    failed += 1
                    print(f"  ERROR {spk}/{target_stem}: {e}", file=sys.stderr, flush=True)
                finally:
                    tmp_path.unlink(missing_ok=True)

                if bytes_streamed - last_report > 1_000_000_000:
                    last_report = bytes_streamed
                    print(
                        f"  ~{bytes_streamed / 1e9:.1f} GB streamed, "
                        f"{extracted} faces extracted ({failed} failed), "
                        f"{len(per_speaker)} speakers covered",
                        flush=True,
                    )
    finally:
        stream.close()

    print(
        f"\ndone: {extracted} face .npz, {failed} failed, "
        f"~{bytes_streamed / 1e9:.2f} GB streamed, under {faces_dir}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/v2"),
        help="dataset root (must contain manifest.json from the audio fetch)",
    )
    p.add_argument(
        "--part",
        default="orig_part_aa",
        help="single orig part (default orig_part_aa). Ignored if --all-parts.",
    )
    p.add_argument(
        "--all-parts",
        action="store_true",
        help="stream all 7 orig parts (aa..ag) as one concatenated tar",
    )
    p.add_argument(
        "--max-clips-per-speaker",
        type=int,
        default=5,
        help="cap face extraction per speaker (default 5; 0 = unlimited)",
    )
    p.add_argument("--face-size", type=int, default=112)
    p.add_argument("--fps", type=int, default=25, help="target fps after subsampling")
    args = p.parse_args(argv)

    manifest_path = args.out / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found; run the audio fetch first", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())
    known: set[str] = set()
    for split in ("train", "val"):
        for row in manifest.get(split, []):
            known.add(row["speaker_id"])
    print(f"manifest: {len(known)} unique speaker ids across train + val")

    urls = (
        [f"{HF_BASE}/{p}" for p in ALL_VIDEO_PARTS]
        if args.all_parts
        else [f"{HF_BASE}/{args.part}"]
    )
    return cmd_fetch_inline(
        urls,
        args.out,
        known,
        args.max_clips_per_speaker,
        args.face_size,
        args.fps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
