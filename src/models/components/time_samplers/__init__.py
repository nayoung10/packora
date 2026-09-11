from abc import ABC, abstractmethod

import torch

from src.utils.tensor_typing import Float


class TimeSampler(ABC):
    """Base class for training time distribution."""

    @abstractmethod
    def sample(self, batch_size: int, device: torch.device) -> Float['b']:
        """Sample batch_size timesteps in [0, 1)."""
        ...
