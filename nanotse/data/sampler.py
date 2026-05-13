"""Batch samplers that shape minibatches for the loss schedule.

``StratifiedSpeakerBatchSampler`` guarantees every emitted batch contains
``items_per_speaker`` clips from each of ``batch_size // items_per_speaker``
distinct target-speakers. This is what makes ``slot_infonce`` produce a
non-zero gradient on every step instead of only when random sampling
happens to put two clips of the same speaker in the batch -- at scale
(thousands of speakers vs. batch size 8) random batching almost never
yields a positive pair.

Used with ``VoxCeleb2MixDataset.target_speaker_of(idx)`` to pre-group
indices by target speaker without materialising the dataset.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import torch
from torch.utils.data import Sampler


class _SupportsTargetSpeaker(Protocol):
    def __len__(self) -> int: ...
    def target_speaker_of(self, idx: int) -> int: ...


class StratifiedSpeakerBatchSampler(Sampler[list[int]]):
    """Batches of (k speakers x m clips), shuffled by epoch.

    Args:
        dataset: object exposing ``__len__`` and ``target_speaker_of(idx)``.
        batch_size: total items per batch. Must be divisible by
            ``items_per_speaker``.
        items_per_speaker: clips per speaker in each batch (default 2).
        seed: RNG seed; the sampler reshuffles deterministically per epoch.

    Raises:
        ValueError: if ``batch_size`` is not divisible by ``items_per_speaker``,
            or if fewer than ``items_per_speaker`` clips exist for any speaker.
    """

    def __init__(
        self,
        dataset: _SupportsTargetSpeaker,
        batch_size: int,
        items_per_speaker: int = 2,
        seed: int = 0,
        drop_last: bool = True,
        eligible_speakers: set[int] | None = None,
    ) -> None:
        """Args:
        eligible_speakers: optional whitelist of speaker indices. When set,
            only indices whose target speaker is in this set are sampled --
            useful for AV training when only some speakers have face cache.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if items_per_speaker <= 0:
            raise ValueError(f"items_per_speaker must be > 0, got {items_per_speaker}")
        if batch_size % items_per_speaker != 0:
            raise ValueError(
                f"batch_size {batch_size} must be divisible by "
                f"items_per_speaker {items_per_speaker}"
            )

        self.batch_size = batch_size
        self.items_per_speaker = items_per_speaker
        self.speakers_per_batch = batch_size // items_per_speaker
        self.seed = seed
        self.drop_last = drop_last

        # One-pass scan: group dataset indices by target speaker.
        by_speaker: dict[int, list[int]] = {}
        for i in range(len(dataset)):
            spk = dataset.target_speaker_of(i)
            if eligible_speakers is not None and spk not in eligible_speakers:
                continue
            by_speaker.setdefault(spk, []).append(i)

        usable = {s: idxs for s, idxs in by_speaker.items() if len(idxs) >= items_per_speaker}
        if len(usable) < self.speakers_per_batch:
            raise ValueError(
                f"not enough speakers with >= {items_per_speaker} clips: "
                f"have {len(usable)} usable speakers, need {self.speakers_per_batch} per batch"
                + (
                    f" (after filtering to {len(eligible_speakers)} eligible)"
                    if eligible_speakers
                    else ""
                )
            )

        self._by_speaker: dict[int, list[int]] = usable
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Optional hook for multi-epoch training to reshuffle deterministically."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        gen = torch.Generator().manual_seed(self.seed + self._epoch)

        # Shuffle each speaker's clip pool, then build a queue of available speakers.
        pools: dict[int, list[int]] = {}
        for spk, idxs in self._by_speaker.items():
            perm = torch.randperm(len(idxs), generator=gen).tolist()
            pools[spk] = [idxs[i] for i in perm]

        speakers = list(pools.keys())
        speaker_order = [speakers[i] for i in torch.randperm(len(speakers), generator=gen).tolist()]
        cursor = 0

        while True:
            batch: list[int] = []
            picked: set[int] = set()
            attempts = 0

            while len(batch) < self.batch_size and attempts < 2 * len(speaker_order):
                spk = speaker_order[cursor % len(speaker_order)]
                cursor += 1
                attempts += 1
                if spk in picked:
                    continue
                if len(pools[spk]) < self.items_per_speaker:
                    continue
                for _ in range(self.items_per_speaker):
                    batch.append(pools[spk].pop())
                picked.add(spk)

            if len(batch) < self.batch_size:
                if self.drop_last or not batch:
                    return
                yield batch
                return
            yield batch

    def __len__(self) -> int:
        total_items = sum(
            (len(v) // self.items_per_speaker) * self.items_per_speaker
            for v in self._by_speaker.values()
        )
        return total_items // self.batch_size
