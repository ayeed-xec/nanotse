#!/usr/bin/env python3
"""Run inline mouth-ROI extraction on locally-downloaded video tars.

Reads ``<root>/raw_video/orig_part_aa..ag`` as a concatenated stream
(matching the streaming-fetch behaviour) and produces face .npz files
under ``<root>/faces/<spk>/<vid>__<clip>.npz``. Already-extracted clips
are skipped via ``if out_npz.exists(): continue``.

After this succeeds, you can ``rm -rf <root>/raw_video`` to recover disk.

Usage:
    python scripts/data_prep/extract_from_local_tars.py --root data/v2 \\
        --max-clips-per-speaker 15
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
from extract_mouth_roi import _extract_lip_frames
from fetch_voxceleb2_mix_smoke import _speaker_of
from fetch_voxceleb2_video import _video_id_of

_PARTS_ORDER = [f"orig_part_a{c}" for c in "abcdefg"]


class _ChainedFileStream(io.RawIOBase):
    """File-like that concatenates a list of local files into one byte stream."""

    def __init__(self, paths: list[Path]) -> None:
        self._paths = list(paths)
        self._idx = 0
        self._fh: io.BufferedReader | None = None
        self._open_next()

    def _open_next(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        while self._idx < len(self._paths):
            try:
                self._fh = self._paths[self._idx].open("rb")
                print(f"  reading: {self._paths[self._idx].name}", flush=True)
                self._idx += 1
                return
            except OSError as e:
                print(f"  skip {self._paths[self._idx]}: {e}", file=sys.stderr)
                self._idx += 1

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        if self._fh is None:
            return b""
        out = bytearray()
        remaining = n if n != -1 else 1 << 30
        while remaining > 0 and self._fh is not None:
            chunk = self._fh.read(min(remaining, 1 << 20))
            if chunk:
                out.extend(chunk)
                remaining -= len(chunk)
                continue
            self._open_next()
        return bytes(out)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/v2"))
    p.add_argument("--max-clips-per-speaker", type=int, default=15)
    p.add_argument("--face-size", type=int, default=112)
    p.add_argument("--fps", type=int, default=25)
    args = p.parse_args(argv)

    manifest_path = args.root / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())
    known: set[str] = set()
    for split in ("train", "val"):
        for row in manifest.get(split, []):
            known.add(row["speaker_id"])
    print(f"manifest: {len(known)} unique speaker ids")

    raw_dir = args.root / "raw_video"
    paths = [raw_dir / name for name in _PARTS_ORDER if (raw_dir / name).exists()]
    if not paths:
        print(f"ERROR: no orig_part_* under {raw_dir}", file=sys.stderr)
        return 1
    total = sum(p.stat().st_size for p in paths)
    print(f"reading {len(paths)} tar part(s), total {total / 1e9:.2f} GB")

    faces_dir = args.root / "faces"
    faces_dir.mkdir(exist_ok=True)
    per_speaker: dict[str, int] = {}
    extracted = failed = skipped = 0
    bytes_read = 0
    last_report = 0

    stream = _ChainedFileStream(paths)
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            for member in tf:
                if not member.name.lower().endswith(".mp4") or not member.isfile():
                    continue
                spk = _speaker_of(member.name)
                if spk is None or spk not in known:
                    continue
                if (
                    args.max_clips_per_speaker
                    and per_speaker.get(spk, 0) >= args.max_clips_per_speaker
                ):
                    continue

                vid = _video_id_of(member.name) or "novid"
                clip = Path(member.name).name
                stem = f"{vid}__{Path(clip).stem}"
                out_npz = faces_dir / spk / f"{stem}.npz"
                if out_npz.exists():
                    per_speaker[spk] = per_speaker.get(spk, 0) + 1
                    skipped += 1
                    continue

                buf = tf.extractfile(member)
                if buf is None:
                    continue
                data = buf.read()
                bytes_read += len(data)

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = Path(tmp.name)
                try:
                    frames = _extract_lip_frames(tmp_path, args.face_size, args.fps)
                    out_npz.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out_npz, frames=frames["frames"], face_ok=frames["face_ok"])
                    extracted += 1
                    per_speaker[spk] = per_speaker.get(spk, 0) + 1
                except Exception as e:
                    failed += 1
                    print(f"  ERROR {spk}/{stem}: {e}", file=sys.stderr, flush=True)
                finally:
                    tmp_path.unlink(missing_ok=True)

                if bytes_read - last_report > 1_000_000_000:
                    last_report = bytes_read
                    print(
                        f"  ~{bytes_read / 1e9:.1f} GB mp4 bytes processed, "
                        f"{extracted} new faces, {skipped} already-present skipped, "
                        f"{len(per_speaker)} speakers seen",
                        flush=True,
                    )
    finally:
        stream.close()

    print(
        f"\ndone: {extracted} new face .npz, {skipped} already present, {failed} failed, "
        f"{len(per_speaker)} unique speakers, under {faces_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
