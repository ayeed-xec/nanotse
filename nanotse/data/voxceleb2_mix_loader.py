"""VoxCeleb2-mix real loader: reads ``data/smoke/manifest.json`` + wavs.

W2.5 of the sprint plan. Pairs target + interferer clips from disjoint
speakers, mixes at random SNR in ``snr_db_range``. Returns the same
``AVMixSample`` contract as the synthetic dataset, so model + train code
swap in transparently.

**Face frames are still synthetic placeholders.** Only `audio_clean_part_aa`
has been fetched (audio only). Real face frames arrive when we fetch
``orig_part_aa`` or run mouth-ROI extraction; until then the visual stack
processes deterministic random uint8 frames.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from nanotse.data.voxceleb2_mix import AVMixSample


class VoxCeleb2MixDataset(Dataset[AVMixSample]):
    """Real-audio 2-speaker mix loader for ``data/smoke/`` produced by the fetch script.

    Mixes are generated on the fly: each ``__getitem__`` picks a target speaker
    and a *different* interferer speaker (deterministic via ``seed + idx``)
    and scales the interferer to a random SNR in ``snr_db_range`` dB.
    """

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "val"] = "train",
        clip_seconds: float = 4.0,
        sample_rate: int = 16000,
        fps: int = 25,
        face_size: int = 112,
        snr_db_range: tuple[float, float] = (0.0, 5.0),
        num_items: int | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.clip_samples = int(clip_seconds * sample_rate)
        self.clip_frames = max(1, int(clip_seconds * fps))
        self.sample_rate = sample_rate
        self.fps = fps
        self.face_size = face_size
        self.snr_db_range = snr_db_range
        self.seed = seed

        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"missing {manifest_path}; run scripts/data_prep/fetch_voxceleb2_mix_smoke.py"
            )
        manifest = json.loads(manifest_path.read_text())
        if split not in manifest:
            raise KeyError(f"split '{split}' not in manifest (keys: {list(manifest)})")
        clips = manifest[split]

        self.by_speaker: dict[str, list[str]] = defaultdict(list)
        for c in clips:
            self.by_speaker[c["speaker_id"]].append(c["wav"])
        self.speakers = sorted(self.by_speaker.keys())
        if len(self.speakers) < 2:
            raise ValueError(
                f"split '{split}' has only {len(self.speakers)} speaker(s); need >= 2 for mixing"
            )

        self.num_items = num_items if num_items is not None else len(clips)

    def __len__(self) -> int:
        return self.num_items

    def _load_wav(self, rel_path: str) -> torch.Tensor:
        path = self.root / rel_path
        arr, sr = sf.read(path)
        if sr != self.sample_rate:
            raise ValueError(f"sample_rate mismatch at {path}: {sr} != {self.sample_rate}")
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))

    def _crop_or_pad(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] >= self.clip_samples:
            start = (x.shape[0] - self.clip_samples) // 2  # center crop
            return x[start : start + self.clip_samples]
        return torch.cat([x, torch.zeros(self.clip_samples - x.shape[0])])

    def __getitem__(self, idx: int) -> AVMixSample:
        if idx < 0 or idx >= self.num_items:
            raise IndexError(idx)
        gen = torch.Generator().manual_seed(self.seed + idx)

        tgt_spk_i = int(torch.randint(0, len(self.speakers), (1,), generator=gen).item())
        tgt_spk = self.speakers[tgt_spk_i]
        tgt_clip = self.by_speaker[tgt_spk][
            int(torch.randint(0, len(self.by_speaker[tgt_spk]), (1,), generator=gen).item())
        ]

        # interferer speaker must differ from target
        others = [s for s in self.speakers if s != tgt_spk]
        intf_spk = others[int(torch.randint(0, len(others), (1,), generator=gen).item())]
        intf_clip = self.by_speaker[intf_spk][
            int(torch.randint(0, len(self.by_speaker[intf_spk]), (1,), generator=gen).item())
        ]

        target = self._crop_or_pad(self._load_wav(tgt_clip))
        interferer = self._crop_or_pad(self._load_wav(intf_clip))

        lo, hi = self.snr_db_range
        snr_db = float((torch.rand((), generator=gen) * (hi - lo) + lo).item())
        tgt_power = (target**2).mean()
        intf_power = (interferer**2).mean()
        scale = torch.sqrt(tgt_power / (intf_power * (10 ** (snr_db / 10.0)) + 1e-10))
        interferer = interferer * scale

        mix = target + interferer

        face = torch.randint(
            0,
            256,
            (self.clip_frames, self.face_size, self.face_size, 3),
            generator=gen,
            dtype=torch.uint8,
        )

        return AVMixSample(
            mix=mix,
            target=target,
            interferer=interferer,
            face=face,
            speaker_id=tgt_spk_i,
            mix_id=idx,
        )
