"""EDM/Karras Heun helpers for flow-compatible sampling."""

import math
from dataclasses import dataclass, fields
from typing import Any

import torch

from src.utils.tensor_typing import Float


@dataclass(frozen=True)
class EDMHeunSamplerConfig:
    """Configuration for stochastic EDM Heun sampling."""

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    s_churn: float = 60.0
    s_min: float = 0.0
    s_max: float = 999.0
    s_noise: float = 1.003

    def __post_init__(self) -> None:
        """Validate sampler hyperparameters."""
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite, got {value}.")
        if self.sigma_min <= 0.0:
            raise ValueError("sigma_min must be > 0.")
        if self.sigma_max <= self.sigma_min:
            raise ValueError("sigma_max must be > sigma_min.")
        if self.rho <= 0.0:
            raise ValueError("rho must be > 0.")
        if self.s_churn < 0.0:
            raise ValueError("s_churn must be >= 0.")
        if self.s_min < 0.0:
            raise ValueError("s_min must be >= 0.")
        if self.s_max < self.s_min:
            raise ValueError("s_max must be >= s_min.")
        if self.s_noise < 0.0:
            raise ValueError("s_noise must be >= 0.")


def edm_heun_config_from_dict(
    sampler_args: dict[str, Any] | None,
) -> EDMHeunSamplerConfig:
    """Build a sampler config from optional override arguments."""
    if sampler_args is None:
        return EDMHeunSamplerConfig()

    valid_names = {field.name for field in fields(EDMHeunSamplerConfig)}
    unknown = sorted(set(sampler_args).difference(valid_names))
    if unknown:
        raise ValueError(f"Unknown EDM Heun sampler args: {unknown}.")
    return EDMHeunSamplerConfig(**dict(sampler_args))


def karras_sigma_schedule(
    num_steps: int,
    config: EDMHeunSamplerConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Float["s"]:
    """Return the Karras sigma schedule with terminal zero."""
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0 for EDM Heun sampling.")
    if num_steps == 1:
        steps = torch.tensor([config.sigma_max], device=device, dtype=dtype)
    else:
        step_indices = torch.arange(num_steps, device=device, dtype=dtype)
        sigma_min_root = config.sigma_min ** (1.0 / config.rho)
        sigma_max_root = config.sigma_max ** (1.0 / config.rho)
        steps = (
            sigma_max_root
            + step_indices / float(num_steps - 1) * (sigma_min_root - sigma_max_root)
        ) ** config.rho
    return torch.cat([steps, torch.zeros_like(steps[:1])], dim=0)


def sigma_to_flow_time(sigma: Float["..."]) -> Float["..."]:
    """Map EDM sigma to flow-matching time for the linear interpolant."""
    return 1.0 / (1.0 + sigma)


def edm_derivative(
    y_sigma: Float["..."],
    x1_pred: Float["..."],
    sigma: Float[""],
) -> Float["..."]:
    """Return the EDM probability-flow derivative from a clean endpoint prediction."""
    return (y_sigma - x1_pred) / sigma.clamp_min(1e-12)


def churn_gamma(
    sigma: Float[""],
    num_steps: int,
    config: EDMHeunSamplerConfig,
) -> float:
    """Return the stochastic churn multiplier for one sigma step."""
    sigma_value = float(sigma.item())
    if sigma_value < config.s_min or sigma_value > config.s_max:
        return 0.0
    return min(config.s_churn / float(num_steps), math.sqrt(2.0) - 1.0)
