import torch.nn as nn
from einops import repeat

from src.models.components.activations import ActivationType, build_activation
from src.utils.tensor_typing import Float


class MLPLatticeEmbedder(nn.Module):
    """Project lattice via MLP and broadcast to per-atom dimension."""

    def __init__(
        self,
        dim: int,
        input_dim: int = 6,
        activation: ActivationType = "silu",
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dim),
            build_activation(activation),
            nn.Linear(dim, dim),
        )

    def forward(self, l: Float['b d'], num_atoms_dim: int) -> Float['b n d']:
        """Embed lattice (B, D) and broadcast to (B, N, D)."""
        h = self.mlp(l)
        return repeat(h, 'b d -> b n d', n=num_atoms_dim)
