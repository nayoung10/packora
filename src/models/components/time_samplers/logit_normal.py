import torch

from src.models.components.time_samplers import TimeSampler
from src.utils.tensor_typing import Float


class LogitNormalTimeSampler(TimeSampler):
    """Logit-normal distribution: sigmoid of N(mu, sigma^2)."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        """Initialize with location and scale parameters."""
        self.mu = mu
        self.sigma = sigma

    def sample(self, batch_size: int, device: torch.device) -> Float['b']:
        """Sample via logit-normal transform."""
        z = self.mu + self.sigma * torch.randn(batch_size, device=device)
        return torch.sigmoid(z)
