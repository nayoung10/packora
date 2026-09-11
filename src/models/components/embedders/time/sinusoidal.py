import math

import torch
import torch.nn as nn

from src.models.components.activations import ActivationType, build_activation
from src.utils.tensor_typing import Float


class SinusoidalEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_dim: int,
        frequency_embedding_dim: int = 256,
        time_factor: float = 1000.0,
        activation: ActivationType = "silu",
    ) -> None:
        """Initialize sinusoidal timestep embedding MLP."""
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            build_activation(activation),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim
        self.time_factor = time_factor

    @staticmethod
    def timestep_embedding(
        t: Float['b'], dim: int, max_period: int = 10000
    ) -> Float['b d']:
        """Compute sinusoidal frequency embedding for scalar timesteps."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: Float['b']) -> Float['b d']:
        """Embed timesteps to (B, D)."""
        t_freq = self.timestep_embedding(
            t * self.time_factor, self.frequency_embedding_dim
        )
        t_emb = self.mlp(t_freq)
        return t_emb
