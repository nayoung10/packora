from abc import ABC, abstractmethod

from torch import Tensor

from src.utils.tensor_typing import Float


class Interpolant(ABC):
    """Base class for flow matching interpolation schedules."""

    @abstractmethod
    def a(self, t: Float['b']) -> Float['b']:
        """Coefficient for the prior sample x0."""
        ...

    @abstractmethod
    def adot(self, t: Float['b']) -> Float['b']:
        """Time derivative of a(t)."""
        ...

    @abstractmethod
    def b(self, t: Float['b']) -> Float['b']:
        """Coefficient for the data sample x1."""
        ...

    @abstractmethod
    def bdot(self, t: Float['b']) -> Float['b']:
        """Time derivative of b(t)."""
        ...

    def _expand_t(self, t_coeff: Float['b'], target: Tensor) -> Tensor:
        """Broadcast a (B,) coefficient to match target's shape."""
        for _ in range(target.ndim - 1):
            t_coeff = t_coeff.unsqueeze(-1)
        return t_coeff

    def It(self, t: Float['b'], x0: Tensor, x1: Tensor) -> Tensor:
        """Compute interpolation: a(t)*x0 + b(t)*x1."""
        a_t = self._expand_t(self.a(t), x0)
        b_t = self._expand_t(self.b(t), x0)
        return a_t * x0 + b_t * x1

    def dtIt(self, t: Float['b'], x0: Tensor, x1: Tensor) -> Tensor:
        """Compute time derivative: adot(t)*x0 + bdot(t)*x1."""
        adot_t = self._expand_t(self.adot(t), x0)
        bdot_t = self._expand_t(self.bdot(t), x0)
        return adot_t * x0 + bdot_t * x1

    def velocity_from_x1_pred(
        self, t: Float['b'], x_t: Tensor, x1_pred: Tensor,
    ) -> Tensor:
        """Compute velocity from x1 prediction: v = adot/a * x_t + (bdot - adot*b/a) * x1."""
        a_t = self._expand_t(self.a(t), x_t)
        adot_t = self._expand_t(self.adot(t), x_t)
        b_t = self._expand_t(self.b(t), x_t)
        bdot_t = self._expand_t(self.bdot(t), x_t)
        return (adot_t / a_t) * x_t + (bdot_t - adot_t * b_t / a_t) * x1_pred

    def score_from_x1_pred(
        self, t: Float['b'], x_t: Tensor, x1_pred: Tensor,
    ) -> Tensor:
        """Compute score: nabla log p_t(x_t) = (b(t)*x1_pred - x_t) / a(t)^2."""
        a_t = self._expand_t(self.a(t), x_t)
        b_t = self._expand_t(self.b(t), x_t)
        return (b_t * x1_pred - x_t) / (a_t ** 2)

    def diffusion_coeff(self, t: Float['b']) -> Float['b']:
        """Natural diffusion coefficient: beta(t) = -2 * adot(t) * a(t)."""
        return -2.0 * self.adot(t) * self.a(t)
