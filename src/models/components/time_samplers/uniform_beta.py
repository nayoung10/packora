# ruff: noqa: F821

import torch

from src.models.components.time_samplers import TimeSampler
from src.utils.tensor_typing import Float


class UniformBetaTimeSampler(TimeSampler):
    """Mixture of uniform and Beta(alpha, beta) timesteps."""

    def __init__(
        self,
        alpha: float = 1.9,
        beta: float = 1.0,
        uniform_probability: float = 0.02,
    ) -> None:
        """Initialize mixture parameters."""
        self.alpha = alpha
        self.beta = beta
        self.uniform_probability = uniform_probability

    def sample(self, batch_size: int, device: torch.device) -> Float["b"]:
        """Sample timesteps from the uniform-beta mixture."""
        dist = torch.distributions.Beta(self.alpha, self.beta)
        samples_beta = dist.sample((batch_size,)).to(device)
        samples_uniform = torch.rand(batch_size, device=device)

        # Select each sample's component independently
        use_uniform = torch.rand(batch_size, device=device) < self.uniform_probability
        return torch.where(use_uniform, samples_uniform, samples_beta)
