import torch

from src.models.components.interpolants import Interpolant
from src.utils.tensor_typing import Float


class LinearInterpolant(Interpolant):
    """Linear interpolation: x_t = (1-t)*x0 + t*x1."""

    def a(self, t: Float['b']) -> Float['b']:
        """Prior coefficient: 1-t."""
        return 1.0 - t

    def adot(self, t: Float['b']) -> Float['b']:
        """Time derivative of prior coefficient: -1."""
        return -torch.ones_like(t)

    def b(self, t: Float['b']) -> Float['b']:
        """Data coefficient: t."""
        return t

    def bdot(self, t: Float['b']) -> Float['b']:
        """Time derivative of data coefficient: 1."""
        return torch.ones_like(t)
