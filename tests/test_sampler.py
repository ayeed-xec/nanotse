"""Tests for StratifiedSpeakerBatchSampler.

Verifies the sampler emits batches where each batch contains the requested
number of clips per speaker (the per-batch positive-pair guarantee for
``slot_infonce``) and rejects mis-shaped requests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nanotse.data import StratifiedSpeakerBatchSampler


@dataclass
class _FakeDS:
    """Minimal stand-in: ``target_speaker_of(idx)`` + ``__len__``."""

    speaker_of: list[int]

    def __len__(self) -> int:
        return len(self.speaker_of)

    def target_speaker_of(self, idx: int) -> int:
        return self.speaker_of[idx]


def test_sampler_emits_pairs_per_batch() -> None:
    speakers = [s for s in range(10) for _ in range(4)]  # 10 speakers x 4 clips
    ds = _FakeDS(speaker_of=speakers)
    sampler = StratifiedSpeakerBatchSampler(ds, batch_size=8, items_per_speaker=2, seed=0)
    for batch in sampler:
        assert len(batch) == 8
        counts: dict[int, int] = {}
        for idx in batch:
            spk = ds.target_speaker_of(idx)
            counts[spk] = counts.get(spk, 0) + 1
        assert max(counts.values()) >= 2, f"no positive pair in batch {batch}: {counts}"


def test_sampler_rejects_bad_batch_size() -> None:
    ds = _FakeDS(speaker_of=[0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="divisible"):
        StratifiedSpeakerBatchSampler(ds, batch_size=5, items_per_speaker=2)


def test_sampler_rejects_too_few_speakers() -> None:
    """Fewer usable speakers than speakers_per_batch => raises."""
    ds = _FakeDS(speaker_of=[0, 0, 1])  # only spk 0 has >=2 clips
    with pytest.raises(ValueError, match="not enough speakers"):
        StratifiedSpeakerBatchSampler(ds, batch_size=4, items_per_speaker=2)


def test_sampler_reshuffles_per_epoch() -> None:
    speakers = [s for s in range(8) for _ in range(4)]
    ds = _FakeDS(speaker_of=speakers)
    sampler = StratifiedSpeakerBatchSampler(ds, batch_size=8, items_per_speaker=2, seed=0)

    sampler.set_epoch(0)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    # Different shuffles => at least one batch differs.
    assert epoch0 != epoch1, "set_epoch should change sample order"


def test_sampler_len_matches_emitted_count() -> None:
    speakers = [s for s in range(8) for _ in range(4)]
    ds = _FakeDS(speaker_of=speakers)
    sampler = StratifiedSpeakerBatchSampler(ds, batch_size=8, items_per_speaker=2, seed=0)
    actual = sum(1 for _ in sampler)
    assert actual == len(sampler), f"len() = {len(sampler)}, but iter yielded {actual}"
