# ruff: noqa: F722,F821
import torch

from src.utils.tensor_typing import Int


def sequential(sample: dict) -> Int["n"]:
    """Return sequential positional indices 0..N-1."""
    N = sample["conditioning"].atomic_numbers.shape[0]
    return torch.arange(N)
