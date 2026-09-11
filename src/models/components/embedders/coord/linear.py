import torch.nn as nn

from src.utils.tensor_typing import Float


class LinearCartCoordEmbedder(nn.Module):
    """Project Cartesian coordinates to model dimension via a single linear layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3, dim)

    def forward(self, x: Float['b n 3']) -> Float['b n d']:
        """Embed (B, N, 3) coordinates to (B, N, D)."""
        return self.proj(x)

# TODO: Fourier features, MLP, etc.