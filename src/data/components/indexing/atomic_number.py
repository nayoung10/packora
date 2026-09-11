# ruff: noqa: F722,F821
import torch

from src.utils.tensor_typing import Float, Int


def atomic_number_random(sample: dict, descending: bool = False) -> Int["n"]:
    """Assign positional indices by atomic number, with random tiebreaking."""
    a = sample["conditioning"].atomic_numbers

    # Random tiebreaker ensures different ordering each call (data augmentation)
    noise = torch.rand_like(a, dtype=torch.float)

    # Composite key: atomic number dominates, noise breaks ties
    sign = -1.0 if descending else 1.0
    sort_key = sign * a.float() + noise * 1e-6

    # Double argsort: first gets ordering, second gets rank (positional index)
    order = sort_key.argsort(dim=-1)
    indices = order.argsort(dim=-1)
    return indices


def atomic_number_xyz(sample: dict, descending: bool = False) -> Int["n"]:
    """Assign positional indices by atomic number, ties broken by (x, y, z) lexicographically."""
    a = sample["conditioning"].atomic_numbers
    x: Float["n 3"] = sample["cart_coords"]

    # Normalize coordinates to [0, 1) per structure per dimension
    x_min = x.min(dim=0, keepdim=True).values
    x_range = x.max(dim=0, keepdim=True).values - x_min
    x_scaled = (x - x_min) / (x_range + 1e-8)

    # Encode lexicographic (x, y, z) order as a single tiebreaker < 1
    tiebreaker = (
        x_scaled[..., 0] * 1e-3 + x_scaled[..., 1] * 1e-6 + x_scaled[..., 2] * 1e-9
    )

    sign = -1.0 if descending else 1.0
    sort_key = sign * a.float() + tiebreaker

    # Stable sort preserves tiebreaker ordering within equal atomic numbers
    order = sort_key.argsort(dim=-1, stable=True)
    indices = order.argsort(dim=-1)
    return indices
