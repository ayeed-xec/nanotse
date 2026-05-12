"""Real-audio VoxCeleb2MixDataset checks against the fetched smoke sample.

These tests use the real ``data/smoke/`` produced by the HF fetch script.
If the manifest is missing they skip cleanly (so CI on a fresh checkout
without the data still passes).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from nanotse.data import VoxCeleb2MixDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO_ROOT / "data" / "smoke"
HAS_REAL_DATA = (SMOKE_ROOT / "manifest.json").exists()

pytestmark = pytest.mark.skipif(
    not HAS_REAL_DATA,
    reason="data/smoke/manifest.json missing; run scripts/data_prep/fetch_voxceleb2_mix_smoke.py",
)


def test_train_val_speakers_are_disjoint() -> None:
    """Stratified-split regression: manifest train and val speakers must NOT overlap."""
    manifest = json.loads((SMOKE_ROOT / "manifest.json").read_text())
    train_spk = {c["speaker_id"] for c in manifest["train"]}
    val_spk = {c["speaker_id"] for c in manifest["val"]}
    assert not (train_spk & val_spk), (
        f"speaker overlap between train and val: {train_spk & val_spk}"
    )


def test_loader_returns_correct_shapes() -> None:
    ds = VoxCeleb2MixDataset(SMOKE_ROOT, split="train", clip_seconds=2.0, num_items=4)
    s = ds[0]
    assert s["mix"].shape == (16000 * 2,)
    assert s["target"].shape == (16000 * 2,)
    assert s["interferer"].shape == (16000 * 2,)
    assert s["face"].dtype == torch.uint8
    assert isinstance(s["speaker_id"], int)


def test_mix_equals_target_plus_interferer() -> None:
    ds = VoxCeleb2MixDataset(SMOKE_ROOT, split="train", clip_seconds=1.0, num_items=2)
    s = ds[1]
    assert torch.allclose(s["mix"], s["target"] + s["interferer"], atol=1e-5)


def test_target_and_interferer_are_from_different_speakers() -> None:
    """Every sample should mix two distinct speakers; check via amplitude RMS sanity.

    We can't directly assert speaker disjointness from tensors -- but at least
    confirm interferer is non-zero (i.e. a real different audio source was found
    and scaled in).
    """
    ds = VoxCeleb2MixDataset(SMOKE_ROOT, split="train", clip_seconds=1.0, num_items=4)
    for i in range(len(ds)):
        s = ds[i]
        assert s["interferer"].abs().mean() > 0, f"item {i} has zero interferer"


def test_loader_determinism() -> None:
    """Same seed + idx -> identical sample, on repeated dataset construction."""
    a = VoxCeleb2MixDataset(SMOKE_ROOT, split="train", clip_seconds=1.0, num_items=4, seed=7)
    b = VoxCeleb2MixDataset(SMOKE_ROOT, split="train", clip_seconds=1.0, num_items=4, seed=7)
    assert torch.equal(a[2]["target"], b[2]["target"])
    assert torch.equal(a[2]["mix"], b[2]["mix"])


def test_loader_different_seed_different_pairing() -> None:
    a = VoxCeleb2MixDataset(SMOKE_ROOT, split="train", clip_seconds=1.0, num_items=4, seed=0)
    b = VoxCeleb2MixDataset(SMOKE_ROOT, split="train", clip_seconds=1.0, num_items=4, seed=1)
    # With 3 train speakers and small num_items, some collisions possible -- we
    # just assert at least one item differs.
    differs = any(not torch.equal(a[i]["mix"], b[i]["mix"]) for i in range(len(a)))
    assert differs, "different seeds produced identical samples"


def test_loader_batches_through_dataloader() -> None:
    ds = VoxCeleb2MixDataset(SMOKE_ROOT, split="train", clip_seconds=1.0, num_items=4)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["mix"].shape == (2, 16000)
    assert batch["target"].shape == (2, 16000)


def test_loader_missing_manifest_raises() -> None:
    with pytest.raises(FileNotFoundError, match="manifest"):
        VoxCeleb2MixDataset(REPO_ROOT / "nowhere", split="train")


def test_loader_rejects_single_speaker_split(tmp_path: Path) -> None:
    """Need >= 2 speakers in the split for mixing."""
    fake = {"train": [{"speaker_id": "id00001", "wav": "audio/id00001/a.wav"}], "val": []}
    (tmp_path / "manifest.json").write_text(json.dumps(fake))
    with pytest.raises(ValueError, match=">= 2"):
        VoxCeleb2MixDataset(tmp_path, split="train")
