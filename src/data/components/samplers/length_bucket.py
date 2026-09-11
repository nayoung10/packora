"""Distributed length-bucketed batch sampler."""

import math
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import Sampler

from src.data.components.samplers.common import (
    get_distributed_rank_info,
    load_selected_lengths,
)


class DistributedLengthBucketBatchSampler(Sampler[list[int]]):
    """Build DDP-safe batches from similarly sized structures."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        bucket_size_multiplier: int = 50,
        drop_last: bool = False,
        seed: int = 0,
        balance_across_ranks: bool = False,
        sampler: Sampler[Any] | None = None,
    ) -> None:
        """Initialize length-bucketed batching state."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if bucket_size_multiplier < 1:
            raise ValueError("bucket_size_multiplier must be >= 1.")
        if sampler is not None:
            dataset = getattr(sampler, "data_source", dataset)
        elif not hasattr(dataset, "selected_indices") and hasattr(
            dataset, "data_source"
        ):
            dataset = dataset.data_source

        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier
        self.drop_last = drop_last
        self.seed = seed
        self.balance_across_ranks = balance_across_ranks
        self.epoch = 0
        self.lengths = load_selected_lengths(dataset)
        self.rank, self.world_size = get_distributed_rank_info()

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for deterministic reshuffling."""
        self.epoch = epoch

    def _build_global_batches(self) -> list[list[int]]:
        """Build shuffled full-dataset batches before rank sharding."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()

        batches: list[list[int]] = []
        carry: list[int] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = carry + indices[start : start + self.bucket_size]
            bucket.sort(key=self.lengths.__getitem__)
            full_count = (len(bucket) // self.batch_size) * self.batch_size
            for batch_start in range(0, full_count, self.batch_size):
                batches.append(bucket[batch_start : batch_start + self.batch_size])
            carry = bucket[full_count:]

        if carry and not self.drop_last:
            batches.append(carry)

        order = torch.randperm(len(batches), generator=generator).tolist()
        batches = [batches[index] for index in order]

        remainder = len(batches) % self.world_size
        if remainder != 0:
            if self.drop_last:
                batches = batches[: len(batches) - remainder]
            else:
                batches.extend(
                    self._filler_batches(batches, self.world_size - remainder)
                )

        if self.balance_across_ranks and self.world_size > 1:
            batches.sort(key=self._batch_max_length)
            groups = [
                batches[start : start + self.world_size]
                for start in range(0, len(batches), self.world_size)
            ]
            group_order = torch.randperm(len(groups), generator=generator).tolist()
            balanced_batches: list[list[int]] = []
            for group_index in group_order:
                group = groups[group_index]
                rank_order = torch.randperm(len(group), generator=generator).tolist()
                balanced_batches.extend(group[index] for index in rank_order)
            return balanced_batches
        return batches

    def _filler_batches(
        self,
        batches: list[list[int]],
        count: int,
    ) -> list[list[int]]:
        """Return duplicated filler batches with the requested count."""
        if count <= 0 or not batches:
            return []
        # Repeat filler batches so every DDP rank gets the same number of steps
        repeat_count = math.ceil(count / len(batches))
        return (batches * repeat_count)[:count]

    def _batch_max_length(self, batch: list[int]) -> int:
        """Return the maximum atom count in a candidate batch."""
        return max(self.lengths[index] for index in batch)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield this rank's length-bucketed batches."""
        batches = self._build_global_batches()
        yield from batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        """Return the number of batches yielded on this rank."""
        total_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            total_batches += 1
        if self.drop_last:
            total_batches -= total_batches % self.world_size
        else:
            total_batches = math.ceil(total_batches / self.world_size) * self.world_size
        return total_batches // self.world_size
