import torch.nn as nn

from src.models.components.activations import ActivationType
from src.models.components.embedders.time.sinusoidal import SinusoidalEmbedder
from src.utils.tensor_typing import Float


class TimestepEmbedder(nn.Module):
    """Embed timestep conditioning into a global vector."""

    def __init__(
        self,
        hidden_dim: int,
        edm_preconditioning_enabled: bool,
        frequency_embedding_dim: int = 256,
        raw_time_factor: float = 1000.0,
        edm_time_factor: float = 1.0,
        activation: ActivationType = "silu",
    ) -> None:
        """Initialize scalar timestep embedder with automatic time scaling."""
        super().__init__()
        time_factor = (
            edm_time_factor if edm_preconditioning_enabled else raw_time_factor
        )
        self.t_embedder = SinusoidalEmbedder(
            hidden_dim=hidden_dim,
            frequency_embedding_dim=frequency_embedding_dim,
            time_factor=time_factor,
            activation=activation,
        )

    def forward(
        self,
        t: Float['b'],
    ) -> Float['b d']:
        """Produce global timestep conditioning vector."""
        return self.t_embedder(t)


__all__ = ["SinusoidalEmbedder", "TimestepEmbedder"]
