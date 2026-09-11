"""Distributed fixed-shape bucketed batch sampler."""

import math
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch.utils.data import BatchSampler, Sampler, SequentialSampler

from src.data.components.samplers.common import (
    get_distributed_rank_info,
    load_selected_families,
    load_selected_lengths,
)

DEFAULT_ATOM_BUCKETS = [32, 64, 96, 128, 160, 192, 224, 256, 288, 300]


class DistributedFixedShapeBucketBatchSampler(BatchSampler):
    """Build DDP-safe batches mapped to a small set of atom-count buckets."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        atom_buckets: list[int] | None = None,
        drop_last: bool = True,
        pad_to_full_batch: bool = False,
        seed: int = 0,
        balance_across_ranks: bool = True,
        sampler: Sampler[Any] | None = None,
        batch_sizes: Mapping[int, int] | None = None,
    ) -> None:
        """Initialize fixed-shape batching state."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        base_sampler = sampler
        if sampler is not None:
            dataset = getattr(sampler, "data_source", dataset)
        elif not hasattr(dataset, "selected_indices") and hasattr(
            dataset, "data_source"
        ):
            base_sampler = dataset
            dataset = dataset.data_source
        if drop_last and pad_to_full_batch:
            raise ValueError("pad_to_full_batch requires drop_last=False.")
        if atom_buckets is None:
            atom_buckets = DEFAULT_ATOM_BUCKETS
        if not atom_buckets:
            raise ValueError("atom_buckets must not be empty.")

        if base_sampler is None:
            base_sampler = SequentialSampler(dataset)
        super().__init__(base_sampler, batch_size, drop_last)
        self.dataset = dataset
        self.atom_buckets = sorted({int(bucket) for bucket in atom_buckets})
        self.batch_sizes = self._normalize_batch_sizes(batch_sizes)
        self.pad_to_full_batch = pad_to_full_batch
        self.seed = seed
        self.balance_across_ranks = balance_across_ranks
        self.epoch = 0
        self.lengths = load_selected_lengths(dataset)
        self.rank, self.world_size = get_distributed_rank_info()
        self._indices_by_bucket = self._build_indices_by_bucket()

    def _normalize_batch_sizes(
        self,
        batch_sizes: Mapping[int, int] | None,
    ) -> dict[int, int] | None:
        """Validate and normalize optional per-bucket batch sizes."""
        if batch_sizes is None:
            return None
        normalized = {int(bucket): int(size) for bucket, size in batch_sizes.items()}
        expected = set(self.atom_buckets)
        configured = set(normalized)
        if configured != expected:
            missing = sorted(expected - configured)
            extra = sorted(configured - expected)
            raise ValueError(
                "batch_sizes keys must exactly match atom_buckets; "
                f"missing={missing}, extra={extra}."
            )
        invalid = {bucket: size for bucket, size in normalized.items() if size < 1}
        if invalid:
            raise ValueError(f"batch_sizes values must be >= 1, got {invalid}.")
        return normalized

    def batch_size_for_bucket(self, bucket: int) -> int:
        """Return the configured batch size for one atom bucket."""
        if self.batch_sizes is None:
            return self.batch_size
        return self.batch_sizes[bucket]

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for deterministic reshuffling."""
        self.epoch = epoch

    def bucket_for_length(self, length: int) -> int:
        """Return the smallest atom bucket that can contain length."""
        for bucket in self.atom_buckets:
            if length <= bucket:
                return bucket
        raise ValueError(
            f"No atom bucket can contain length {length}; "
            f"largest bucket is {self.atom_buckets[-1]}."
        )

    def dropped_sample_count(self) -> int:
        """Return the number of samples dropped by full-batch bucketing."""
        if not self.drop_last:
            return 0
        return sum(
            len(indices) % self.batch_size_for_bucket(bucket)
            for bucket, indices in self._indices_by_bucket.items()
        )

    def dropped_sample_count_by_bucket(self) -> dict[int, int]:
        """Return dropped sample counts keyed by atom bucket."""
        if not self.drop_last:
            return {bucket: 0 for bucket in self.atom_buckets}
        return {
            bucket: len(indices) % self.batch_size_for_bucket(bucket)
            for bucket, indices in self._indices_by_bucket.items()
        }

    def _build_indices_by_bucket(self) -> dict[int, list[int]]:
        """Group dataloader-local indices by target atom bucket."""
        indices_by_bucket: dict[int, list[int]] = {
            bucket: [] for bucket in self.atom_buckets
        }
        for index, length in enumerate(self.lengths):
            indices_by_bucket[self.bucket_for_length(length)].append(index)
        return indices_by_bucket

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

    def _bucket_batches(
        self,
        bucket: int,
        bucket_indices: list[int],
        generator: torch.Generator,
    ) -> list[list[int]]:
        """Build shuffled batches within one atom bucket."""
        if not bucket_indices:
            return []
        batch_size = self.batch_size_for_bucket(bucket)
        order = torch.randperm(len(bucket_indices), generator=generator).tolist()
        shuffled = [bucket_indices[index] for index in order]
        full_count = (len(shuffled) // batch_size) * batch_size
        batches = [
            shuffled[start : start + batch_size]
            for start in range(0, full_count, batch_size)
        ]
        if not self.drop_last and full_count < len(shuffled):
            tail = shuffled[full_count:]
            if self.pad_to_full_batch:
                pad_count = batch_size - len(tail)
                repeat_count = math.ceil(pad_count / len(tail))
                tail = tail + (tail * repeat_count)[:pad_count]
            batches.append(tail)
        return batches

    def _build_balanced_global_batches(
        self,
        generator: torch.Generator,
        indices_by_bucket: dict[int, list[int]] | None = None,
    ) -> list[list[int]]:
        """Build global batches whose rank groups share one atom bucket."""
        if indices_by_bucket is None:
            indices_by_bucket = self._indices_by_bucket
        groups: list[list[list[int]]] = []
        for bucket in self.atom_buckets:
            bucket_batches = self._bucket_batches(
                bucket,
                indices_by_bucket[bucket],
                generator,
            )
            remainder = len(bucket_batches) % self.world_size
            if remainder != 0:
                if self.drop_last:
                    bucket_batches = bucket_batches[: len(bucket_batches) - remainder]
                else:
                    bucket_batches.extend(
                        self._filler_batches(
                            bucket_batches,
                            self.world_size - remainder,
                        )
                    )
            for start in range(0, len(bucket_batches), self.world_size):
                group = bucket_batches[start : start + self.world_size]
                if len(group) == self.world_size:
                    groups.append(group)

        order = torch.randperm(len(groups), generator=generator).tolist()
        return [batch for group_index in order for batch in groups[group_index]]

    def _build_global_batches(self) -> list[list[int]]:
        """Build shuffled fixed-shape batches before rank sharding."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.balance_across_ranks and self.world_size > 1:
            return self._build_balanced_global_batches(generator)

        batches: list[list[int]] = []
        for bucket in self.atom_buckets:
            batches.extend(
                self._bucket_batches(
                    bucket,
                    self._indices_by_bucket[bucket],
                    generator,
                )
            )

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
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        """Yield this rank's fixed-shape batches."""
        self.rank, self.world_size = get_distributed_rank_info()
        batches = self._build_global_batches()
        yield from batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        """Return the number of batches yielded on this rank."""
        self.rank, self.world_size = get_distributed_rank_info()
        total_batches = 0
        for bucket, indices in self._indices_by_bucket.items():
            batch_size = self.batch_size_for_bucket(bucket)
            bucket_batches = len(indices) // batch_size
            if not self.drop_last and len(indices) % batch_size:
                bucket_batches += 1
            if self.balance_across_ranks and self.world_size > 1:
                remainder = bucket_batches % self.world_size
                if remainder != 0:
                    if self.drop_last:
                        bucket_batches -= remainder
                    else:
                        bucket_batches += self.world_size - remainder
            total_batches += bucket_batches

        if total_batches == 0:
            return 0
        if self.drop_last:
            total_batches -= total_batches % self.world_size
        else:
            total_batches = math.ceil(total_batches / self.world_size) * self.world_size
        return total_batches // self.world_size


class DistributedFamilyFixedShapeBucketBatchSampler(
    DistributedFixedShapeBucketBatchSampler
):
    """Build fixed-shape batches from one random entry per CSD family each epoch."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        atom_buckets: list[int] | None = None,
        drop_last: bool = True,
        pad_to_full_batch: bool = False,
        seed: int = 0,
        balance_across_ranks: bool = True,
        sampler: Sampler[Any] | None = None,
        batch_sizes: Mapping[int, int] | None = None,
    ) -> None:
        """Initialize family-grouped fixed-shape batching state."""
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            atom_buckets=atom_buckets,
            drop_last=drop_last,
            pad_to_full_batch=pad_to_full_batch,
            seed=seed,
            balance_across_ranks=balance_across_ranks,
            sampler=sampler,
            batch_sizes=batch_sizes,
        )
        families = load_selected_families(self.dataset)
        grouped: dict[str, list[int]] = {}
        for index, family in enumerate(families):
            grouped.setdefault(family, []).append(index)
        self.family_indices = [grouped[family] for family in sorted(grouped)]

    def _representatives_by_bucket(
        self,
        generator: torch.Generator,
    ) -> dict[int, list[int]]:
        """Choose one representative per family and group them by atom bucket."""
        indices_by_bucket = {bucket: [] for bucket in self.atom_buckets}
        random_values = torch.rand(
            len(self.family_indices),
            generator=generator,
        ).tolist()
        for family_indices, random_value in zip(
            self.family_indices,
            random_values,
        ):
            choice = min(
                int(random_value * len(family_indices)), len(family_indices) - 1
            )
            index = family_indices[choice]
            bucket = self.bucket_for_length(self.lengths[index])
            indices_by_bucket[bucket].append(index)
        return indices_by_bucket

    def _build_global_batches(self) -> list[list[int]]:
        """Build family-balanced fixed-shape batches before rank sharding."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices_by_bucket = self._representatives_by_bucket(generator)
        if self.balance_across_ranks and self.world_size > 1:
            return self._build_balanced_global_batches(
                generator,
                indices_by_bucket=indices_by_bucket,
            )

        batches: list[list[int]] = []
        for bucket in self.atom_buckets:
            batches.extend(
                self._bucket_batches(
                    bucket,
                    indices_by_bucket[bucket],
                    generator,
                )
            )
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
        return batches

    def __len__(self) -> int:
        """Return this rank's family-balanced batch count for the current epoch."""
        self.rank, self.world_size = get_distributed_rank_info()
        return len(self._build_global_batches()) // self.world_size
