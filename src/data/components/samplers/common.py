"""Shared helpers for atom-count-aware batch samplers."""

from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset


def get_distributed_rank_info() -> tuple[int, int]:
    """Return the current distributed rank and world size."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def load_selected_lengths(dataset: Any) -> list[int]:
    """Load atom counts in dataloader-local index order."""
    if isinstance(dataset, ConcatDataset):
        lengths: list[int] = []
        for child in dataset.datasets:
            lengths.extend(load_selected_lengths(child))
        return lengths

    if getattr(dataset, "num_atoms_by_index", None) is not None:
        all_lengths = dataset.num_atoms_by_index
    else:
        all_lengths = np.load(dataset.num_atoms_cache_path, allow_pickle=False)
    selected_indices = np.asarray(dataset.selected_indices, dtype=np.int64)
    return all_lengths[selected_indices].astype(np.int64, copy=False).tolist()


def load_selected_families(dataset: Any) -> list[str]:
    """Load CSD families in dataloader-local index order."""
    cache_path = getattr(dataset, "csd_families_cache_path", None)
    if cache_path is None:
        raise ValueError("Family sampling requires a CSD-family cache path.")
    if not cache_path.is_file():
        raise FileNotFoundError(f"CSD-family cache not found: {cache_path}")
    all_families = np.load(cache_path, allow_pickle=False)
    if all_families.shape != (dataset.num_samples_total,):
        raise ValueError(
            f"CSD-family cache {cache_path} has shape {all_families.shape}; "
            f"expected {(dataset.num_samples_total,)}."
        )
    selected_indices = np.asarray(dataset.selected_indices, dtype=np.int64)
    return all_families[selected_indices].astype(str, copy=False).tolist()
