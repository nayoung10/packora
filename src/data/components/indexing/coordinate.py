import torch

from src.utils.tensor_typing import Float, Int


def xyz(sample: dict, descending: bool = False) -> Int['n']:
    """Assign positional indices by lexicographic (x, y, z) coordinate sorting."""
    x: Float['n 3'] = sample["cart_coords"]

    # Normalize coordinates to [0, 1) per structure per dimension
    x_min = x.min(dim=0, keepdim=True).values
    x_range = x.max(dim=0, keepdim=True).values - x_min
    x_scaled = (x - x_min) / (x_range + 1e-8)

    # Encode lexicographic (x, y, z) order as a single composite key
    sort_key = x_scaled[..., 0] + x_scaled[..., 1] * 1e-3 + x_scaled[..., 2] * 1e-6

    if descending:
        sort_key = -sort_key

    # Double argsort: first gets ordering, second gets rank (positional index)
    order = sort_key.argsort(dim=-1, stable=True)
    indices = order.argsort(dim=-1)
    return indices
