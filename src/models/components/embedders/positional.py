import math

import torch
import torch.nn as nn

from src.models.components.activations import ActivationType, build_activation
from src.utils.tensor_typing import Float, Int


class PositionalEmbedder(nn.Module):
    """Embeds integer indices into vector representations using sine/cosine positional encoding."""

    def __init__(
        self,
        hidden_dim: int,
        frequency_embedding_dim: int = 256,
        max_len: int = 2048,
        activation: ActivationType = "silu",
    ) -> None:
        super().__init__()
        if frequency_embedding_dim % 2 != 0:
            raise ValueError(
                f"frequency_embedding_dim must be even, got {frequency_embedding_dim}"
            )
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim, bias=True),
            build_activation(activation),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_dim = frequency_embedding_dim
        self.max_len = max_len

    @staticmethod
    def _create_positional_embedding(
        indices: Int['... n'], dim: int, max_len: int = 2048
    ) -> Float['... n d']:
        """Compute sine/cosine positional encoding from integer indices."""
        K = torch.arange(dim // 2, device=indices.device)
        div_term = max_len ** (2 * K / dim)
        angles = indices[..., None] * math.pi / div_term
        pos_embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return pos_embedding

    def forward(self, indices: Int['b n']) -> Float['b n d']:
        """Embed integer indices to (B, N, D)."""
        pos_freq = self._create_positional_embedding(
            indices, self.frequency_embedding_dim, self.max_len
        )
        pos_emb = self.mlp(pos_freq)
        return pos_emb
