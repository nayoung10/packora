import torch

from src.models.components.time_samplers import TimeSampler
from src.utils.tensor_typing import Float


class BetaTimeSampler(TimeSampler):
    """Beta(alpha, beta) distribution over [0, 1)."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        """Initialize with shape parameters."""
        self.alpha = alpha
        self.beta = beta

    def sample(self, batch_size: int, device: torch.device) -> Float['b']:
        """Sample from Beta distribution."""
        dist = torch.distributions.Beta(self.alpha, self.beta)
        return dist.sample((batch_size,)).to(device)
