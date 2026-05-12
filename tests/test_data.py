"""Shape, determinism, and contract checks for the synthetic AV-mix dataset."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from nanotse.data import SyntheticAVMixDataset


def test_dataset_length_matches_request() -> None:
    ds = SyntheticAVMixDataset(num_clips=200, clip_seconds=4.0)
    assert len(ds) == 200


def test_item_shapes_match_config() -> None:
    sr, fps, face = 16000, 25, 112
    ds = SyntheticAVMixDataset(
        num_clips=2,
        clip_seconds=2.0,
        sample_rate=sr,
        fps=fps,
        face_size=face,
    )
    s = ds[0]
    assert s["mix"].shape == (sr * 2,)
    assert s["target"].shape == (sr * 2,)
    assert s["interferer"].shape == (sr * 2,)
    assert s["face"].shape == (fps * 2, face, face, 3)
    assert s["face"].dtype == torch.uint8
    assert isinstance(s["speaker_id"], int)
    assert s["mix_id"] == 0


def test_mix_equals_target_plus_interferer() -> None:
    ds = SyntheticAVMixDataset(num_clips=4, clip_seconds=1.0)
    s = ds[2]
    assert torch.allclose(s["mix"], s["target"] + s["interferer"], atol=1e-6)


def test_same_seed_same_item() -> None:
    a = SyntheticAVMixDataset(num_clips=4, clip_seconds=1.0, seed=42)
    b = SyntheticAVMixDataset(num_clips=4, clip_seconds=1.0, seed=42)
    assert torch.equal(a[3]["target"], b[3]["target"])
    assert torch.equal(a[3]["mix"], b[3]["mix"])


def test_different_seed_different_item() -> None:
    a = SyntheticAVMixDataset(num_clips=4, clip_seconds=1.0, seed=0)
    b = SyntheticAVMixDataset(num_clips=4, clip_seconds=1.0, seed=1)
    assert not torch.equal(a[0]["target"], b[0]["target"])


def test_dataloader_batches_cleanly() -> None:
    ds = SyntheticAVMixDataset(num_clips=8, clip_seconds=0.5)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["mix"].shape == (4, 8000)
    assert batch["target"].shape == (4, 8000)
    assert batch["face"].shape == (4, 12, 112, 112, 3)
    assert batch["speaker_id"].shape == (4,)
