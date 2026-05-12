#!/usr/bin/env python3
"""Stream-download the VoxCeleb2-mix smoke subset from HuggingFace.

The dataset is ~563 GB total across 14 multi-part tar files; we stream only
the start of `audio_clean_part_aa` (the first 40 GB tar segment) and stop
as soon as we have enough disjoint speakers and clips per speaker. Typical
smoke run reads on the order of 30-80 MB before bailing out.

Train/val speakers are disjoint by construction -- we partition speakers
in deterministic alternating order, controlled by `--seed`. The W1 plan
requires this so single-speaker val splits can't silently hide passthrough
bugs.

Usage:
    # Inspect tar structure first (no extraction, ~1 MB read)
    python scripts/data_prep/fetch_voxceleb2_mix_smoke.py --list

    # Real fetch with the W1 defaults (30 train + 10 val speakers x 5 clips)
    python scripts/data_prep/fetch_voxceleb2_mix_smoke.py

Outputs:
    data/smoke/audio/<speaker_id>/<clip>.wav
    data/smoke/manifest.json   (with "train" and "val" arrays of {speaker_id, wav})
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

HF_URL = (
    "https://huggingface.co/datasets/alibabasglab/VoxCeleb2-mix/resolve/main/audio_clean_part_aa"
)
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "smoke"
USER_AGENT = "nanotse-fetch/0.0.1"


def _speaker_of(path: str) -> str | None:
    """Best-effort: pull the speaker id out of a VoxCeleb-style path.

    VoxCeleb2 uses ``idXXXXX/<video>/<clip>.wav``. The tar may have a
    leading directory (``audio_clean/idXXXXX/...``) -- we tolerate either.
    """
    parts = [p for p in path.split("/") if p]
    if not parts or not path.lower().endswith(".wav"):
        return None
    for p in parts:
        if p.startswith("id") and p[2:].isdigit():
            return p
    return None


def _stream_tar(url: str) -> Iterable[tarfile.TarInfo]:
    """Yield TarInfo members from a streaming HTTP response."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with (
        urllib.request.urlopen(req, timeout=600) as resp,
        tarfile.open(fileobj=resp, mode="r|") as tf,
    ):
        yield from tf


def cmd_list(url: str, n: int) -> int:
    print(f"streaming {url}", file=sys.stderr)
    print(f"first {n} tar members:", file=sys.stderr)
    for i, m in enumerate(_stream_tar(url)):
        if i >= n:
            break
        print(f"  {m.name}  ({m.size} bytes)")
    return 0


def cmd_fetch(
    url: str,
    out: Path,
    num_train_spk: int,
    num_val_spk: int,
    clips_per_spk: int,
    seed: int,
) -> int:
    out.mkdir(parents=True, exist_ok=True)
    audio_dir = out / "audio"
    audio_dir.mkdir(exist_ok=True)

    needed_spk = num_train_spk + num_val_spk
    spk_clips: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
    finished_spk: list[str] = []

    print(f"streaming {url} -- will stop at {needed_spk} speakers x {clips_per_spk} clips")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    bytes_read = 0
    with (
        urllib.request.urlopen(req, timeout=600) as resp,
        tarfile.open(fileobj=resp, mode="r|") as tf,
    ):
        for member in tf:
            spk = _speaker_of(member.name)
            if spk is None or not member.isfile():
                continue
            if spk in finished_spk:
                continue
            buf = tf.extractfile(member)
            if buf is None:
                continue
            data = buf.read()
            bytes_read += len(data)

            spk_dir = audio_dir / spk
            spk_dir.mkdir(exist_ok=True)
            clip_name = Path(member.name).name
            (spk_dir / clip_name).write_bytes(data)
            spk_clips[spk].append(member)

            if len(spk_clips[spk]) >= clips_per_spk:
                finished_spk.append(spk)
                print(
                    f"  speaker {spk} done "
                    f"({len(finished_spk)}/{needed_spk}, ~{bytes_read / 1e6:.1f} MB read)"
                )
                if len(finished_spk) >= needed_spk:
                    break

    if len(finished_spk) < needed_spk:
        print(
            f"WARNING: only collected {len(finished_spk)} speakers (asked for {needed_spk})",
            file=sys.stderr,
        )

    # Deterministic disjoint train/val split.
    import random

    rng = random.Random(seed)
    rng.shuffle(finished_spk)
    train_spk = finished_spk[:num_train_spk]
    val_spk = finished_spk[num_train_spk : num_train_spk + num_val_spk]

    def manifest_for(speakers: list[str]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for spk in speakers:
            for m in spk_clips[spk][:clips_per_spk]:
                rows.append({"speaker_id": spk, "wav": f"audio/{spk}/{Path(m.name).name}"})
        return rows

    manifest = {"train": manifest_for(train_spk), "val": manifest_for(val_spk)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"\nwrote {out / 'manifest.json'}: "
        f"{len(manifest['train'])} train clips ({len(train_spk)} speakers), "
        f"{len(manifest['val'])} val clips ({len(val_spk)} speakers)"
    )
    # Hard guard against accidentally non-disjoint splits.
    assert not (set(train_spk) & set(val_spk)), "train and val speakers overlap"
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=HF_URL, help="HF URL of the tar to stream")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output dir (default data/smoke/)")
    p.add_argument(
        "--list",
        action="store_true",
        help="just print the first N tar member names and exit (no extraction)",
    )
    p.add_argument("--list-n", type=int, default=30, help="number of members to print with --list")
    p.add_argument("--num-train-speakers", type=int, default=30)
    p.add_argument("--num-val-speakers", type=int, default=10)
    p.add_argument("--clips-per-speaker", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.list:
        return cmd_list(args.url, args.list_n)
    return cmd_fetch(
        args.url,
        args.out,
        args.num_train_speakers,
        args.num_val_speakers,
        args.clips_per_speaker,
        args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
