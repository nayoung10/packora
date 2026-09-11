import math

import torch

from src.models.components.interpolants import Interpolant
from src.utils.tensor_typing import Float


class TrigonometricInterpolant(Interpolant):
    """Trigonometric interpolation: x_t = cos(πt/2)*x0 + sin(πt/2)*x1."""

    def a(self, t: Float['b']) -> Float['b']:
        """Prior coefficient: cos(πt/2)."""
        return torch.cos(math.pi * t / 2.0)

    def adot(self, t: Float['b']) -> Float['b']:
        """Time derivative of prior coefficient: -(π/2)sin(πt/2)."""
        return -(math.pi / 2.0) * torch.sin(math.pi * t / 2.0)

    def b(self, t: Float['b']) -> Float['b']:
        """Data coefficient: sin(πt/2)."""
        return torch.sin(math.pi * t / 2.0)

    def bdot(self, t: Float['b']) -> Float['b']:
        """Time derivative of data coefficient: (π/2)cos(πt/2)."""
        return (math.pi / 2.0) * torch.cos(math.pi * t / 2.0)
