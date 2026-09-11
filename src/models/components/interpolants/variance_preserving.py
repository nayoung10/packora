import math

import torch

from src.models.components.interpolants import Interpolant
from src.utils.tensor_typing import Float


class TauConstantSchedule:
    """Constant noise schedule: tau(t) = t."""

    def tau(self, t: Float['b']) -> Float['b']:
        """Evaluate tau(t) = t."""
        return t.clone()

    def tau_dot(self, t: Float['b']) -> Float['b']:
        """Derivative: d/dt tau(t) = 1."""
        return torch.ones_like(t)


class TauLinearSchedule:
    """Linear noise schedule: beta(s) = beta_min + (beta_max - beta_min) * s."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        """Store noise schedule bounds."""
        self.beta_min = beta_min
        self.beta_max = beta_max

    def tau(self, t: Float['b']) -> Float['b']:
        """Evaluate tau(t) = exp(beta_min/2 * log(t) - (beta_max-beta_min)/4 * log^2(t))."""
        log_t = torch.log(t)
        return torch.exp(
            0.5 * self.beta_min * log_t
            - 0.25 * (self.beta_max - self.beta_min) * log_t ** 2
        )

    def tau_dot(self, t: Float['b']) -> Float['b']:
        """Derivative of tau via chain rule."""
        log_t = torch.log(t)
        tau_val = torch.exp(
            0.5 * self.beta_min * log_t
            - 0.25 * (self.beta_max - self.beta_min) * log_t ** 2
        )
        return tau_val * (
            0.5 * self.beta_min / t
            - 0.5 * (self.beta_max - self.beta_min) * log_t / t
        )


class TauCosineSchedule:
    """Cosine noise schedule with offset to avoid boundary singularities."""

    def __init__(self, offset: float = 0.008) -> None:
        """Store offset and precompute constants."""
        self.offset = offset
        self._factor = math.pi / (2.0 + 2.0 * offset)
        self._csc = 1.0 / math.sin(self._factor)
        self._cutoff = 1.0 / math.e

    def tau(self, t: Float['b']) -> Float['b']:
        """Evaluate tau(t), clamped to zero for t <= 1/e where log(t) < -1."""
        raw = self._csc * torch.sin(self._factor * (1.0 + torch.log(t.clamp(min=self._cutoff))))
        return torch.where(t > self._cutoff, raw, torch.zeros_like(t))

    def tau_dot(self, t: Float['b']) -> Float['b']:
        """Derivative of tau, zero for t <= 1/e."""
        t_safe = t.clamp(min=self._cutoff)
        raw = self._csc * self._factor * torch.cos(
            self._factor * (1.0 + torch.log(t_safe))
        ) / t_safe
        return torch.where(t > self._cutoff, raw, torch.zeros_like(t))


class VariancePreservingInterpolant(Interpolant):
    """VP interpolation: x_t = sqrt(1 - tau(t)^2)*x0 + tau(t)*x1."""

    def __init__(self, tau_schedule: object) -> None:
        """Store the tau noise schedule."""
        self.tau_schedule = tau_schedule

    def a(self, t: Float['b']) -> Float['b']:
        """Prior coefficient: sqrt(1 - tau(t)^2)."""
        tau = self.tau_schedule.tau(t)
        return torch.sqrt(1.0 - tau ** 2)

    def adot(self, t: Float['b']) -> Float['b']:
        """Time derivative: -tau * tau_dot / sqrt(1 - tau^2)."""
        tau = self.tau_schedule.tau(t)
        tau_dot = self.tau_schedule.tau_dot(t)
        return -tau * tau_dot / torch.sqrt(1.0 - tau ** 2)

    def b(self, t: Float['b']) -> Float['b']:
        """Data coefficient: tau(t)."""
        return self.tau_schedule.tau(t)

    def bdot(self, t: Float['b']) -> Float['b']:
        """Time derivative: tau_dot(t)."""
        return self.tau_schedule.tau_dot(t)
