"""VoxCeleb2-mix real loader: reads ``<root>/manifest.json`` + wavs.

W2.5 of the sprint plan. Pairs target + interferer clips from disjoint
speakers, mixes at random SNR in ``snr_db_range``. Returns the same
``AVMixSample`` contract as the synthetic dataset, so model + train code
swap in transparently.

**Face frames:** auto-uses real mouth-ROI cache at
``<root>/faces/<spk>/*.npz`` (produced by
``scripts/data_prep/extract_mouth_roi.py``). The per-sample face is
deterministically selected from the target speaker's available .npz
files based on ``idx`` so the same item always yields the same face --
important for reproducibility. If the cache is absent or the picked .npz
holds zero frames, we silently fall back to deterministic random uint8
placeholders (the pre-W3.x behaviour, kept so audio-only smoke tests
still pass without the visual data).
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

    def target_speaker_of(self, idx: int) -> int:
        """Deterministic target-speaker index for a given dataset position.

        Mirrors the per-item ``manual_seed(self.seed + idx)`` used inside
        ``__getitem__`` so a sampler can group same-speaker indices without
        materialising the whole dataset first.
        """
        if idx < 0 or idx >= self.num_items:
            raise IndexError(idx)
        gen = torch.Generator().manual_seed(self.seed + idx)
        return int(torch.randint(0, len(self.speakers), (1,), generator=gen).item())

    def speakers_with_face_cache(self) -> set[int]:
        """Subset of self.speakers (by index) that have at least one .npz under faces/.

        Useful when building a sampler that prefers AV-able speakers so the
        visual stream sees real lip motion most batches instead of zero-padded
        fallbacks.
        """
        faces_root = self.root / "faces"
        if not faces_root.exists():
            return set()
        out: set[int] = set()
        for i, spk in enumerate(self.speakers):
            if any((faces_root / spk).glob("*.npz")):
                out.add(i)
        return out

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
        tgt_clips_all = self.by_speaker[tgt_spk]
        tgt_clip_idx = int(torch.randint(0, len(tgt_clips_all), (1,), generator=gen).item())
        tgt_clip = tgt_clips_all[tgt_clip_idx]

        # Enrollment: a *different* clip from the same speaker. Falls back to
        # the same clip with a shifted crop if the speaker has only one clip.
        if len(tgt_clips_all) > 1:
            other_indices = [i for i in range(len(tgt_clips_all)) if i != tgt_clip_idx]
            enroll_pos = int(torch.randint(0, len(other_indices), (1,), generator=gen).item())
            enroll_clip = tgt_clips_all[other_indices[enroll_pos]]
        else:
            enroll_clip = tgt_clip

        # interferer speaker must differ from target
        others = [s for s in self.speakers if s != tgt_spk]
        intf_spk = others[int(torch.randint(0, len(others), (1,), generator=gen).item())]
        intf_clip = self.by_speaker[intf_spk][
            int(torch.randint(0, len(self.by_speaker[intf_spk]), (1,), generator=gen).item())
        ]

        target = self._crop_or_pad(self._load_wav(tgt_clip))
        interferer = self._crop_or_pad(self._load_wav(intf_clip))
        enrollment = self._crop_or_pad(self._load_wav(enroll_clip))

        lo, hi = self.snr_db_range
        snr_db = float((torch.rand((), generator=gen) * (hi - lo) + lo).item())
        tgt_power = (target**2).mean()
        intf_power = (interferer**2).mean()
        scale = torch.sqrt(tgt_power / (intf_power * (10 ** (snr_db / 10.0)) + 1e-10))
        interferer = interferer * scale

        mix = target + interferer

        face = self._load_face(tgt_spk, idx, gen)

        return AVMixSample(
            mix=mix,
            target=target,
            interferer=interferer,
            face=face,
            enrollment=enrollment,
            speaker_id=tgt_spk_i,
            mix_id=idx,
        )

    def _load_face(self, spk: str, idx: int, gen: torch.Generator) -> torch.Tensor:
        """Real mouth-ROI cache when present; zero fallback (NOT random) when missing.

        Zero frames are neutral: they propagate cleanly through the visual encoder
        without injecting per-clip random bias into the training signal. The
        previous random-uint8 fallback effectively trained the model to ignore
        the visual stream on the (currently many) clips without face cache.
        """
        spk_faces = self.root / "faces" / spk
        if spk_faces.exists():
            npzs = sorted(spk_faces.glob("*.npz"))
            if npzs:
                chosen = npzs[idx % len(npzs)]
                frames = np.load(chosen)["frames"]  # (F, H, W, 3) uint8
                if frames.shape[0] > 0:
                    frames = self._align_frames(frames)
                    return torch.from_numpy(frames)
        return torch.zeros(
            (self.clip_frames, self.face_size, self.face_size, 3),
            dtype=torch.uint8,
        )

    def _align_frames(self, frames: np.ndarray) -> np.ndarray:
        """Center-crop or zero-pad to ``self.clip_frames`` time steps."""
        f, h, w, c = frames.shape
        if f == self.clip_frames:
            return frames
        if f > self.clip_frames:
            start = (f - self.clip_frames) // 2
            return frames[start : start + self.clip_frames]
        pad = np.zeros((self.clip_frames - f, h, w, c), dtype=frames.dtype)
        return np.concatenate([frames, pad], axis=0)
