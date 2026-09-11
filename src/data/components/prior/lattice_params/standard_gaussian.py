import torch

from src.utils.tensor_typing import Float


def standard_gaussian(batch_size: int, lattice_dim: int = 6) -> Float['b d']:
    """Sample lattice from standard Gaussian N(0, I)."""
    return torch.randn(batch_size, lattice_dim)
