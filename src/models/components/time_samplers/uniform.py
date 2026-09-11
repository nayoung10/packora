import torch

from src.models.components.time_samplers import TimeSampler
from src.utils.tensor_typing import Float


class UniformTimeSampler(TimeSampler):
    """Uniform distribution over [0, 1)."""

    def sample(self, batch_size: int, device: torch.device) -> Float['b']:
        """Sample uniformly from [0, 1)."""
        return torch.rand(batch_size, device=device)
