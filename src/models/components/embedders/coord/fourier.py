import math

import torch
import torch.nn as nn
from einops import rearrange

from src.models.components.activations import ActivationType, build_activation
from src.utils.tensor_typing import Float


class FourierCartCoordEmbedder(nn.Module):
    """Embed Cartesian coordinates via random Fourier features and MLP projection."""

    def __init__(
        self,
        dim: int,
        num_channels: int = 256,
        bandwidth: float = 10.0,
        learnable: bool = False,
        include_linear: bool = False,
        activation: ActivationType = "silu",
    ) -> None:
        super().__init__()

        # Random Fourier feature parameters (EDM2 formula)
        freqs = 2 * math.pi * torch.randn(3, num_channels) * bandwidth
        phases = 2 * math.pi * torch.rand(3, num_channels)

        if learnable:
            self.freqs = nn.Parameter(freqs)
            self.phases = nn.Parameter(phases)
        else:
            self.register_buffer("freqs", freqs)
            self.register_buffer("phases", phases)

        # Fourier features -> model dim
        fourier_dim = 3 * num_channels
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, dim, bias=True),
            build_activation(activation),
            nn.Linear(dim, dim, bias=True),
        )

        # Optional additive linear projection
        self.linear_proj = nn.Linear(3, dim) if include_linear else None

    def _fourier_features(self, x: Float['b n 3']) -> Float['b n fourier_dim']:
        """Compute per-dimension random Fourier features."""
        x_expanded = rearrange(x, 'b n d -> b n d 1')
        args = x_expanded * self.freqs + self.phases
        features = math.sqrt(2.0) * torch.cos(args)
        features = rearrange(features, 'b n d c -> b n (d c)')
        return features

    def forward(self, x: Float['b n 3']) -> Float['b n d']:
        """Embed (B, N, 3) coordinates to (B, N, D) via Fourier features."""
        h = self.mlp(self._fourier_features(x))
        if self.linear_proj is not None:
            h = h + self.linear_proj(x)
        return h
