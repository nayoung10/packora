"""Embed clean endpoint estimates independently of fixed conditions."""

# ruff: noqa: F722,F821
from typing import Any

import torch
from einops import rearrange
from torch import nn

from src.utils.tensor_typing import Float


class SelfConditioningEmbedder(nn.Module):
    """Add coordinate and lattice features only where a guess is available."""

    def __init__(self, coord_embedder: nn.Module, lattice_embedder: nn.Module) -> None:
        """Initialize independent coordinate and lattice embedders."""
        super().__init__()
        self.coord_embedder = coord_embedder
        self.lattice_embedder = lattice_embedder

    def forward(self, noisy_batch: dict[str, Any]) -> Float["b n d"]:
        """Embed model-unit predictions and gate after embedding."""
        guess = noisy_batch.get("self_conditioning")
        if guess is None:
            coords = torch.zeros_like(noisy_batch["x_t"])
            lattice = torch.zeros_like(noisy_batch["l_t"])
            available = torch.zeros_like(noisy_batch["times"], dtype=torch.bool)
        else:
            coords = guess["coords"]
            lattice = guess["lattice"]
            available = guess["available"]
        valid = rearrange(available, "b -> b 1") & noisy_batch["atom_mask"].bool()
        coords = torch.where(rearrange(valid, "b n -> b n 1"), coords, 0.0)
        lattice = torch.where(rearrange(available, "b -> b 1"), lattice, 0.0)
        update = self.coord_embedder(coords) + self.lattice_embedder(
            lattice, coords.shape[1]
        )
        return torch.where(rearrange(valid, "b n -> b n 1"), update, 0.0)
