#!/usr/bin/env python3
"""Stream-download the VoxCeleb2-mix audio_clean subset from HuggingFace.

The dataset is ~563 GB across 14 multi-part tar files (7 audio + 7 video).
This script handles the audio side. Two modes:

* **Smoke / subset** (default): stream `audio_clean_part_aa` only and stop as
  soon as we have enough disjoint speakers and clips per speaker. Reads on the
  order of 30-80 MB before bailing out. Used for `make smoke` and CI.
* **Full** (``--all-parts --unlimited``): stream every audio_clean part
  (aa..ag) and extract every speaker / clip. Output ~150-200 GB depending on
  source clip lengths; the streamed tars are never written to disk. Used for
  3060 full-training runs.

Train/val speakers are disjoint by construction -- we partition speakers in
deterministic shuffled order, controlled by `--seed`.

Usage:
    # Inspect tar structure (no extraction, ~1 MB read)
    python scripts/data_prep/fetch_voxceleb2_mix_smoke.py --list

    # Subset fetch (W1 defaults: 30 train + 10 val speakers x 5 clips)
    python scripts/data_prep/fetch_voxceleb2_mix_smoke.py

    # Full dataset (every speaker, every clip, all 7 audio parts)
    python scripts/data_prep/fetch_voxceleb2_mix_smoke.py \
        --all-parts --unlimited --out data/full

Outputs:
    <out>/audio/<speaker_id>/<clip>.wav
    <out>/manifest.json   (with "train" and "val" arrays of {speaker_id, wav})
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import sys
import tarfile
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/alibabasglab/VoxCeleb2-mix/resolve/main"
DEFAULT_PART = "audio_clean_part_aa"
ALL_AUDIO_PARTS = [f"audio_clean_part_a{c}" for c in "abcdefg"]
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "smoke"
USER_AGENT = "nanotse-fetch/0.0.3"
UNLIMITED_SENTINEL = 10**9
_HF_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"


def _hf_headers() -> dict[str, str]:
    """Build HTTP headers, attaching the HF token when one is configured.

    Token resolution order (matches huggingface_hub conventions):
      1. ``HF_TOKEN`` env var
      2. ``HUGGING_FACE_HUB_TOKEN`` env var
      3. ``~/.cache/huggingface/token`` file
    Auth boosts download rate limits and is required for gated datasets;
    public files still work without it.
    """
    import os

    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and _HF_TOKEN_PATH.exists():
        with contextlib.suppress(OSError):
            token = _HF_TOKEN_PATH.read_text().strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _speaker_of(path: str) -> str | None:
    """Best-effort: pull the speaker id out of a VoxCeleb-style path.

    VoxCeleb2 uses ``idXXXXX/<video>/<clip>.<ext>``. The tar may have a
    leading directory (``audio_clean/idXXXXX/...`` or ``orig/<split>/idXXXXX/...``).
    Format-agnostic: works for .wav and .mp4. The caller filters by extension
    if it cares.
    """
    parts = [p for p in path.split("/") if p]
    if not parts:
        return None
    for p in parts:
        if p.startswith("id") and len(p) > 2 and p[2:].isdigit():
            return p
    return None


class _ChainedHTTPStream:
    """File-like that concatenates the bodies of multiple HTTP GETs, with
    transparent Range-resume on mid-stream socket drops.

    VoxCeleb2-mix's tars are **split archives** -- only the concatenation
    of ``audio_clean_part_aa..ag`` is a valid tar; each individual part
    will fail to open. ``tarfile.open(mode="r|")`` needs ``.read(n)`` and
    nothing else.

    HuggingFace's CDN has a habit of closing the connection mid-stream
    on large files (~5-40 GB in). When that happens ``urllib.read()``
    returns empty bytes before ``Content-Length`` is exhausted. We detect
    that here and reopen the SAME url with ``Range: bytes=pos-`` so
    tarfile sees a seamless stream.
    """

    _MAX_RETRIES_PER_URL = 8

    def __init__(self, urls: list[str], headers: dict[str, str]) -> None:
        self._urls = list(urls)
        self._url_idx = 0
        self._headers = dict(headers)
        self._current: object | None = None
        self._url_pos = 0
        self._url_total = 0  # 0 means "unknown" (skip resume logic)
        self._retries_on_current = 0
        self._open_at(0)

    # Per-read timeout (seconds). HF's CDN occasionally leaves a TCP socket
    # ESTABLISHED with no data flowing; without a read deadline urllib blocks
    # forever. 60 s is generous for one chunk fetch while still allowing the
    # resume path to fire on a stuck connection.
    _READ_TIMEOUT = 60.0

    def _open_at(self, byte_offset: int) -> None:
        """Open self._urls[self._url_idx] starting at byte_offset (Range request)."""
        if self._current is not None:
            with contextlib.suppress(Exception):
                self._current.close()  # type: ignore[attr-defined]
            self._current = None
        if self._url_idx >= len(self._urls):
            return
        url = self._urls[self._url_idx]
        headers = dict(self._headers)
        verb = "fetching"
        if byte_offset > 0:
            headers["Range"] = f"bytes={byte_offset}-"
            verb = f"resuming @ {byte_offset / 1e9:.2f} GB"
        print(f"  {verb}: {url}", file=sys.stderr, flush=True)
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=self._READ_TIMEOUT)
        # Set socket-level read deadline so individual chunk reads can't hang
        # forever. `fp` is the underlying socket reader.
        with contextlib.suppress(AttributeError, OSError):
            resp.fp.raw._sock.settimeout(self._READ_TIMEOUT)  # type: ignore[attr-defined,union-attr]
        cl = resp.headers.get("Content-Length")
        try:
            cl_int = int(cl) if cl else 0
        except ValueError:
            cl_int = 0
        # When using Range, Content-Length reports remaining bytes (cl_int);
        # absolute total is byte_offset + cl_int.
        self._url_total = (byte_offset + cl_int) if cl_int else 0
        self._url_pos = byte_offset
        self._current = resp

    def _advance(self) -> None:
        self._url_idx += 1
        self._retries_on_current = 0
        self._url_pos = 0
        self._url_total = 0
        if self._url_idx < len(self._urls):
            self._open_at(0)
        else:
            if self._current is not None:
                with contextlib.suppress(Exception):
                    self._current.close()  # type: ignore[attr-defined]
            self._current = None

    def _handle_short_read(self) -> bool:
        """urllib returned empty before url_total reached. Returns True if resumed."""
        if self._url_total > 0 and self._url_pos < self._url_total:
            if self._retries_on_current >= self._MAX_RETRIES_PER_URL:
                print(
                    f"  GIVING UP on {self._urls[self._url_idx]} after "
                    f"{self._retries_on_current} retries at byte {self._url_pos}",
                    file=sys.stderr,
                    flush=True,
                )
                return False
            self._retries_on_current += 1
            self._open_at(self._url_pos)
            return True
        return False

    def _safe_read(self, n: int) -> bytes:
        """Wrap urllib.read() so socket drops, IncompleteReads, AND read timeouts
        all trigger our resume path. ``socket.timeout`` (TimeoutError on 3.10+)
        is the one that fires when HF leaves a TCP socket ESTABLISHED with no
        data flowing -- the failure mode that hung the pipeline for 1h40m
        before the timeout was added."""
        if self._current is None:
            return b""
        try:
            return self._current.read(n)  # type: ignore[attr-defined,no-any-return]
        except (
            TimeoutError,
            OSError,
            urllib.error.URLError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
        ) as e:
            print(
                f"  read error at byte {self._url_pos / 1e9:.2f} GB: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            return b""

    def read(self, n: int = -1) -> bytes:
        if self._current is None:
            return b""
        out = bytearray()
        # The n=-1 path is only used by tarfile in non-streaming mode; we
        # don't actually use it but keep it correct.
        if n == -1 or n is None:
            while self._current is not None:
                chunk = self._safe_read(8192)
                if chunk:
                    out.extend(chunk)
                    self._url_pos += len(chunk)
                    continue
                if self._handle_short_read():
                    continue
                self._advance()
            return bytes(out)
        remaining = n
        while remaining > 0 and self._current is not None:
            chunk = self._safe_read(remaining)
            if chunk:
                out.extend(chunk)
                self._url_pos += len(chunk)
                remaining -= len(chunk)
                self._retries_on_current = 0
                continue
            if self._handle_short_read():
                continue
            self._advance()
        return bytes(out)

    def close(self) -> None:
        if self._current is not None:
            with contextlib.suppress(Exception):
                self._current.close()  # type: ignore[attr-defined]
        self._current = None
        self._url_idx = len(self._urls)


def cmd_list(url: str, n: int) -> int:
    """Print the first ``n`` tar members of a single tar URL (no chained-stream)."""
    print(f"streaming {url}", file=sys.stderr)
    print(f"first {n} tar members:", file=sys.stderr)
    req = urllib.request.Request(url, headers=_hf_headers())
    with (
        urllib.request.urlopen(req, timeout=600) as resp,
        tarfile.open(fileobj=resp, mode="r|") as tf,
    ):
        for i, m in enumerate(tf):
            if i >= n:
                break
            print(f"  {m.name}  ({m.size} bytes)")
    return 0


def cmd_fetch(
    urls: list[str],
    out: Path,
    num_train_spk: int,
    num_val_spk: int,
    clips_per_spk: int,
    seed: int,
) -> int:
    out.mkdir(parents=True, exist_ok=True)
    audio_dir = out / "audio"
    audio_dir.mkdir(exist_ok=True)

    target_spk = num_train_spk + num_val_spk
    unlimited = target_spk >= UNLIMITED_SENTINEL or clips_per_spk == 0
    if unlimited:
        target_spk = 0  # sentinel meaning "no cap"

    spk_clips: dict[str, list[str]] = defaultdict(list)
    finished_spk: set[str] = set()
    bytes_read = 0
    last_report = 0

    print(
        f"streaming {len(urls)} tar part(s) as one concatenated tar; "
        f"target = {target_spk if target_spk else 'all'} speakers, "
        f"clips_per_speaker = {clips_per_spk if clips_per_spk else 'all'}",
        flush=True,
    )

    stream = _ChainedHTTPStream(urls, _hf_headers())
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            for member in tf:
                if not member.name.lower().endswith(".wav"):
                    continue
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
                spk_clips[spk].append(clip_name)

                # Throttled progress: print roughly every GB or every speaker completion.
                if bytes_read - last_report > 1_000_000_000:
                    last_report = bytes_read
                    print(
                        f"  ~{bytes_read / 1e9:.1f} GB read, "
                        f"{len(spk_clips)} speakers seen, "
                        f"{len(finished_spk)} finished",
                        flush=True,
                    )

                if clips_per_spk > 0 and len(spk_clips[spk]) >= clips_per_spk:
                    finished_spk.add(spk)
                    print(
                        f"  speaker {spk} done "
                        f"({len(finished_spk)}/{target_spk or 'all'}, "
                        f"~{bytes_read / 1e6:.1f} MB read)",
                        flush=True,
                    )
                    if target_spk and len(finished_spk) >= target_spk:
                        break
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"  HTTP error during stream: {e}", file=sys.stderr)
    finally:
        stream.close()

    if clips_per_spk == 0:
        # When clips_per_spk is unlimited, no speaker was marked "finished"
        # mid-stream; treat every speaker we saw as a candidate.
        finished_spk = set(spk_clips.keys())

    if target_spk and len(finished_spk) < target_spk:
        print(
            f"WARNING: only collected {len(finished_spk)} speakers (asked for {target_spk})",
            file=sys.stderr,
        )

    # Deterministic disjoint train/val split.
    import random

    rng = random.Random(seed)
    speakers = sorted(finished_spk)
    rng.shuffle(speakers)

    if unlimited:
        # Use a 80/20 train/val split when no explicit counts given.
        n_val = max(1, len(speakers) // 5)
        n_train = len(speakers) - n_val
    else:
        n_train = min(num_train_spk, len(speakers))
        n_val = min(num_val_spk, len(speakers) - n_train)

    train_spk = speakers[:n_train]
    val_spk = speakers[n_train : n_train + n_val]

    def manifest_for(speakers_list: list[str]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for spk in speakers_list:
            clips = spk_clips[spk] if clips_per_spk == 0 else spk_clips[spk][:clips_per_spk]
            for clip_name in clips:
                rows.append({"speaker_id": spk, "wav": f"audio/{spk}/{clip_name}"})
        return rows

    manifest = {"train": manifest_for(train_spk), "val": manifest_for(val_spk)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"\nwrote {out / 'manifest.json'}: "
        f"{len(manifest['train'])} train clips ({len(train_spk)} speakers), "
        f"{len(manifest['val'])} val clips ({len(val_spk)} speakers); "
        f"~{bytes_read / 1e9:.2f} GB read"
    )
    assert not (set(train_spk) & set(val_spk)), "train and val speakers overlap"
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url", default=None, help="HF URL of a single tar to stream (overrides --part)"
    )
    p.add_argument(
        "--part",
        default=DEFAULT_PART,
        help="audio_clean part name (default audio_clean_part_aa). Ignored if --url or --all-parts.",
    )
    p.add_argument(
        "--all-parts",
        action="store_true",
        help="stream all 7 audio_clean parts (aa..ag) in order",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output dir (default data/smoke/)")
    p.add_argument(
        "--list",
        action="store_true",
        help="just print the first N tar member names and exit (no extraction)",
    )
    p.add_argument("--list-n", type=int, default=30, help="number of members to print with --list")
    p.add_argument(
        "--num-train-speakers", type=int, default=30, help="target train speakers (0 = unlimited)"
    )
    p.add_argument(
        "--num-val-speakers", type=int, default=10, help="target val speakers (0 = unlimited)"
    )
    p.add_argument(
        "--clips-per-speaker", type=int, default=5, help="max clips per speaker (0 = all)"
    )
    p.add_argument(
        "--unlimited",
        action="store_true",
        help="extract every speaker / every clip (sets train+val and clips-per-speaker to 0)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.unlimited:
        args.num_train_speakers = UNLIMITED_SENTINEL
        args.num_val_speakers = 0
        args.clips_per_speaker = 0

    if args.list:
        url = args.url or f"{HF_BASE}/{args.part}"
        return cmd_list(url, args.list_n)

    if args.url:
        urls = [args.url]
    elif args.all_parts:
        urls = [f"{HF_BASE}/{p}" for p in ALL_AUDIO_PARTS]
    else:
        urls = [f"{HF_BASE}/{args.part}"]

    return cmd_fetch(
        urls,
        args.out,
        args.num_train_speakers,
        args.num_val_speakers,
        args.clips_per_speaker,
        args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
