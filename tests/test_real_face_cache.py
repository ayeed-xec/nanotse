"""Loader correctly switches between real face cache and synthetic fallback."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from nanotse.data import VoxCeleb2MixDataset


def _make_tiny_manifest(tmp_path: Path) -> Path:
    """Build a 2-speaker, 2-clips-each manifest with 1-sec @16k wavs."""
    audio = tmp_path / "audio"
    for spk in ("id00001", "id00002"):
        (audio / spk).mkdir(parents=True, exist_ok=True)
        for clip in ("00001.wav", "00002.wav"):
            arr = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
            sf.write(audio / spk / clip, arr, 16000)
    manifest = {
        "train": [
            {"speaker_id": "id00001", "wav": "audio/id00001/00001.wav"},
            {"speaker_id": "id00001", "wav": "audio/id00001/00002.wav"},
            {"speaker_id": "id00002", "wav": "audio/id00002/00001.wav"},
            {"speaker_id": "id00002", "wav": "audio/id00002/00002.wav"},
        ],
        "val": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_loader_falls_back_to_synthetic_when_no_face_cache(tmp_path: Path) -> None:
    root = _make_tiny_manifest(tmp_path)
    ds = VoxCeleb2MixDataset(root, split="train", clip_seconds=1.0, num_items=4, seed=0)
    sample = ds[0]
    assert sample["face"].shape == (25, 112, 112, 3)
    assert sample["face"].dtype.is_floating_point is False  # uint8


def test_loader_uses_real_face_cache_when_present(tmp_path: Path) -> None:
    root = _make_tiny_manifest(tmp_path)
    faces_dir = root / "faces" / "id00001"
    faces_dir.mkdir(parents=True)
    real_frames = (np.arange(25 * 112 * 112 * 3) % 256).astype(np.uint8).reshape(25, 112, 112, 3)
    np.savez_compressed(
        faces_dir / "v1__00001.npz", frames=real_frames, face_ok=np.ones(25, dtype=bool)
    )

    ds = VoxCeleb2MixDataset(root, split="train", clip_seconds=1.0, num_items=4, seed=0)
    # Repeatedly fetch idx until we hit a sample whose target speaker is id00001.
    found = False
    for idx in range(len(ds)):
        sample = ds[idx]
        if sample["speaker_id"] == 0:  # id00001 is sorted-first => speaker_id 0
            assert sample["face"].shape == (25, 112, 112, 3)
            assert int(sample["face"].sum().item()) == int(real_frames.sum())
            found = True
            break
    assert found, "no sample hit id00001 in 4 tries"


def test_loader_aligns_frames_to_clip_length(tmp_path: Path) -> None:
    root = _make_tiny_manifest(tmp_path)
    faces_dir = root / "faces" / "id00001"
    faces_dir.mkdir(parents=True)
    long_frames = np.zeros((40, 112, 112, 3), dtype=np.uint8)  # 40 > target 25
    np.savez_compressed(
        faces_dir / "v1__00001.npz", frames=long_frames, face_ok=np.ones(40, dtype=bool)
    )
    short_frames = np.full((10, 112, 112, 3), 200, dtype=np.uint8)
    np.savez_compressed(
        faces_dir / "v2__00002.npz", frames=short_frames, face_ok=np.ones(10, dtype=bool)
    )

    ds = VoxCeleb2MixDataset(root, split="train", clip_seconds=1.0, num_items=8, seed=0)
    for idx in range(len(ds)):
        sample = ds[idx]
        assert sample["face"].shape == (25, 112, 112, 3), f"idx {idx}: {sample['face'].shape}"
