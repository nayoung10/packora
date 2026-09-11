import numpy as np
from torch.utils.data import BatchSampler, Dataset, SequentialSampler
from unittest.mock import patch
from pathlib import Path

from src.data.components.samplers import (
    DistributedFamilyFixedShapeBucketBatchSampler,
    DistributedFixedShapeBucketBatchSampler,
    DistributedLengthBucketBatchSampler,
)


class _AtomCountDataset(Dataset):
    """Minimal dataset exposing atom-count metadata for samplers."""

    def __init__(self, lengths: list[int]) -> None:
        """Initialize the fake dataset with atom counts."""
        self.num_atoms_by_index = np.asarray(lengths, dtype=np.int64)
        self.selected_indices = list(range(len(lengths)))

    def __len__(self) -> int:
        """Return the fake dataset size."""
        return len(self.selected_indices)

    def __getitem__(self, index: int) -> int:
        """Return the index as fake sample data."""
        return index


class _FamilyDataset(_AtomCountDataset):
    """Minimal dataset exposing aligned atom-count and CSD-family caches."""

    def __init__(
        self,
        tmp_path: Path,
        lengths: list[int],
        families: list[str],
        selected_indices: list[int] | None = None,
    ) -> None:
        """Initialize family metadata and optional entry selection."""
        super().__init__(lengths)
        self.num_samples_total = len(lengths)
        self.selected_indices = (
            list(range(len(lengths))) if selected_indices is None else selected_indices
        )
        self.csd_families_cache_path = tmp_path / "families.npy"
        np.save(self.csd_families_cache_path, np.asarray(families, dtype="<U6"))


def test_fixed_shape_sampler_drops_bucket_tails() -> None:
    """Fixed-shape sampling drops incomplete batches within each bucket."""
    dataset = _AtomCountDataset([10, 20, 30, 40, 50])
    sampler = DistributedFixedShapeBucketBatchSampler(
        dataset=dataset,
        batch_size=2,
        atom_buckets=[32, 64],
        drop_last=True,
        seed=0,
    )

    batches = list(iter(sampler))

    assert len(sampler) == 2
    assert len(batches) == 2
    assert all(len(batch) == 2 for batch in batches)
    assert sampler.dropped_sample_count() == 1
    assert sampler.dropped_sample_count_by_bucket() == {32: 1, 64: 0}


def test_fixed_shape_sampler_implements_batch_sampler_interface() -> None:
    """Fixed-shape sampling implements PyTorch's batch-sampler contract."""
    dataset = _AtomCountDataset([10, 20])

    sampler = DistributedFixedShapeBucketBatchSampler(
        dataset=dataset,
        batch_size=2,
        atom_buckets=[32],
    )

    assert isinstance(sampler, BatchSampler)


def test_fixed_shape_sampler_preserves_ddp_rank_sequences() -> None:
    """Fixed-shape sampling preserves exact deterministic batches across ranks."""
    dataset = _AtomCountDataset([10] * 11 + [40] * 13 + [70] * 17 + [110] * 19)
    expected = {
        0: [[29, 32, 40, 25], [15, 22, 16, 18], [45, 54, 43, 58]],
        1: [[36, 38, 33, 26], [11, 20, 23, 19], [52, 51, 48, 59]],
        2: [[35, 30, 34, 28], [21, 12, 13, 14], [42, 49, 46, 57]],
    }

    for rank, expected_batches in expected.items():
        with patch(
            "src.data.components.samplers.fixed_shape.get_distributed_rank_info",
            return_value=(rank, 3),
        ):
            sampler = DistributedFixedShapeBucketBatchSampler(
                dataset=dataset,
                batch_size=4,
                atom_buckets=[32, 64, 96, 128],
                drop_last=True,
                seed=42,
                balance_across_ranks=True,
            )

            assert len(sampler) == 3
            assert list(sampler) == expected_batches


def test_fixed_shape_sampler_preserves_legacy_scalar_batch_sequence() -> None:
    """Scalar batch sizing preserves the pre-extension deterministic sequence."""
    dataset = _AtomCountDataset(
        [10, 11, 12, 13, 14, 33, 34, 35, 36, 37, 70, 71, 72, 73]
    )
    sampler = DistributedFixedShapeBucketBatchSampler(
        dataset,
        2,
        [32, 64, 96],
        True,
        False,
        7,
        True,
    )

    assert sampler._build_global_batches() == [
        [3, 2],
        [10, 13],
        [6, 7],
        [12, 11],
        [0, 1],
        [8, 9],
    ]
    assert len(sampler) == 6
    assert sampler.dropped_sample_count() == 2
    assert sampler.dropped_sample_count_by_bucket() == {32: 1, 64: 1, 96: 0}


def test_fixed_shape_sampler_uniform_mapping_matches_scalar_behavior() -> None:
    """Uniform per-bucket sizing exactly matches legacy scalar sizing."""
    dataset = _AtomCountDataset(
        [10, 11, 12, 13, 14, 33, 34, 35, 36, 37, 70, 71, 72, 73]
    )
    common = {
        "dataset": dataset,
        "batch_size": 2,
        "atom_buckets": [32, 64, 96],
        "drop_last": True,
        "seed": 7,
        "balance_across_ranks": True,
    }
    with patch(
        "src.data.components.samplers.fixed_shape.get_distributed_rank_info",
        return_value=(0, 2),
    ):
        scalar = DistributedFixedShapeBucketBatchSampler(**common)
        mapped = DistributedFixedShapeBucketBatchSampler(
            **common,
            batch_sizes={32: 2, 64: 2, 96: 2},
        )

        assert mapped._build_global_batches() == scalar._build_global_batches()
        assert list(mapped) == list(scalar)
        assert len(mapped) == len(scalar)
        assert mapped.dropped_sample_count() == scalar.dropped_sample_count()
        assert (
            mapped.dropped_sample_count_by_bucket()
            == scalar.dropped_sample_count_by_bucket()
        )


def test_fixed_shape_sampler_uses_per_bucket_batch_sizes() -> None:
    """Per-bucket sizing varies batch length without mixing atom buckets."""
    dataset = _AtomCountDataset([10] * 8 + [40] * 6 + [70] * 4)
    sampler = DistributedFixedShapeBucketBatchSampler(
        dataset=dataset,
        batch_size=8,
        atom_buckets=[32, 64, 96],
        batch_sizes={32: 4, 64: 3, 96: 2},
        drop_last=True,
        seed=0,
        balance_across_ranks=False,
    )

    batches = sampler._build_global_batches()

    assert sorted(len(batch) for batch in batches) == [2, 2, 3, 3, 4, 4]
    assert all(
        len(batch)
        == sampler.batch_size_for_bucket(
            sampler.bucket_for_length(max(sampler.lengths[index] for index in batch))
        )
        for batch in batches
    )


def test_fixed_shape_sampler_rejects_incomplete_batch_size_mapping() -> None:
    """Per-bucket sizing requires one valid value for every atom bucket."""
    dataset = _AtomCountDataset([10, 40])

    with np.testing.assert_raises_regex(ValueError, "exactly match"):
        DistributedFixedShapeBucketBatchSampler(
            dataset=dataset,
            batch_size=2,
            atom_buckets=[32, 64],
            batch_sizes={32: 2},
        )


def test_fixed_shape_sampler_pads_bucket_tails() -> None:
    """Pad incomplete bucket batches by duplicating samples from that bucket."""
    dataset = _AtomCountDataset([10, 11, 12, 13, 14])
    sampler = DistributedFixedShapeBucketBatchSampler(
        dataset=dataset,
        batch_size=4,
        atom_buckets=[32],
        drop_last=False,
        pad_to_full_batch=True,
        seed=0,
        balance_across_ranks=False,
    )

    batches = list(iter(sampler))
    observed = {index for batch in batches for index in batch}

    assert len(sampler) == 2
    assert len(batches) == 2
    assert all(len(batch) == 4 for batch in batches)
    assert observed == set(range(5))
    assert sampler.dropped_sample_count() == 0


def test_fixed_shape_sampler_pads_balanced_rank_groups() -> None:
    """Pad batch count to world size while keeping fixed-size batches."""
    dataset = _AtomCountDataset([10, 11, 12, 13, 14])
    sampler = DistributedFixedShapeBucketBatchSampler(
        dataset=dataset,
        batch_size=4,
        atom_buckets=[32],
        drop_last=False,
        pad_to_full_batch=True,
        seed=0,
        balance_across_ranks=True,
    )
    sampler.world_size = 3

    batches = sampler._build_global_batches()
    observed = {index for batch in batches for index in batch}

    assert len(batches) == 3
    assert len(batches) % sampler.world_size == 0
    assert all(len(batch) == 4 for batch in batches)
    assert observed == set(range(5))


def test_fixed_shape_sampler_repeats_filler_batches_for_tiny_buckets() -> None:
    """Pad tiny buckets enough to give every rank one batch."""
    dataset = _AtomCountDataset([10])
    sampler = DistributedFixedShapeBucketBatchSampler(
        dataset=dataset,
        batch_size=4,
        atom_buckets=[32],
        drop_last=False,
        pad_to_full_batch=True,
        seed=0,
        balance_across_ranks=True,
    )
    sampler.world_size = 4

    batches = sampler._build_global_batches()
    observed = {index for batch in batches for index in batch}

    assert len(batches) == 4
    assert all(len(batch) == 4 for batch in batches)
    assert observed == {0}


def test_fixed_shape_sampler_aligns_rank_bucket_groups() -> None:
    """Balanced fixed-shape sampling gives each rank group one atom bucket."""
    lengths = [16] * 8 + [48] * 8 + [80] * 8
    dataset = _AtomCountDataset(lengths)
    sampler = DistributedFixedShapeBucketBatchSampler(
        dataset=dataset,
        batch_size=2,
        atom_buckets=[32, 64, 96],
        drop_last=True,
        seed=0,
        balance_across_ranks=True,
    )
    sampler.world_size = 2
    sampler.rank = 0

    batches = sampler._build_global_batches()

    assert len(batches) % sampler.world_size == 0
    for start in range(0, len(batches), sampler.world_size):
        group = batches[start : start + sampler.world_size]
        group_buckets = {
            sampler.bucket_for_length(max(sampler.lengths[index] for index in batch))
            for batch in group
        }
        assert len(group_buckets) == 1


def test_fixed_shape_sampler_rejects_oversized_sample() -> None:
    """Fixed-shape sampling fails when a sample exceeds the largest bucket."""
    dataset = _AtomCountDataset([10, 65])

    try:
        DistributedFixedShapeBucketBatchSampler(
            dataset=dataset,
            batch_size=2,
            atom_buckets=[32, 64],
        )
    except ValueError as exc:
        assert "largest bucket" in str(exc)
    else:
        raise AssertionError("Expected oversized sample to fail.")


def test_length_bucket_sampler_repeats_filler_batches_for_tiny_dataset() -> None:
    """Pad tiny datasets enough to give every rank one batch."""
    dataset = _AtomCountDataset([10])
    sampler = DistributedLengthBucketBatchSampler(
        dataset=dataset,
        batch_size=4,
        bucket_size_multiplier=1,
        drop_last=False,
        seed=0,
    )
    sampler.world_size = 4

    batches = sampler._build_global_batches()
    observed = {index for batch in batches for index in batch}

    assert len(batches) == 4
    assert observed == {0}


def test_length_bucket_sampler_accepts_lightning_sampler_wrapper() -> None:
    """Length-bucket sampling unwraps Lightning's sampler reconstruction input."""
    dataset = _AtomCountDataset([10, 20, 30, 40])
    sampler = DistributedLengthBucketBatchSampler(
        dataset=SequentialSampler(dataset),
        batch_size=2,
        bucket_size_multiplier=1,
        drop_last=False,
        seed=0,
    )

    batches = sampler._build_global_batches()
    observed = {index for batch in batches for index in batch}

    assert observed == set(range(4))


def test_family_fixed_shape_sampler_selects_one_entry_per_family(
    tmp_path: Path,
) -> None:
    """Select exactly one random representative from every eligible family."""
    dataset = _FamilyDataset(
        tmp_path,
        lengths=[20, 22, 24, 26],
        families=["AAAAAA", "AAAAAA", "BBBBBB", "CCCCCC"],
    )
    sampler = DistributedFamilyFixedShapeBucketBatchSampler(
        dataset=dataset,
        batch_size=1,
        atom_buckets=[32],
        drop_last=False,
        seed=7,
        balance_across_ranks=False,
    )

    observed = [index for batch in sampler for index in batch]

    assert len(observed) == 3
    assert len(set(observed).intersection({0, 1})) == 1
    assert {2, 3}.issubset(observed)
