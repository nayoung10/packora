"""Normalization helpers for model components."""

from __future__ import annotations

from typing import Any, Literal

import torch.nn as nn
import torch.nn.functional as F

from src.utils.tensor_typing import Float

NormType = Literal["layernorm", "rmsnorm"]


def build_norm(
    dim: int,
    norm_type: NormType = "layernorm",
    elementwise_affine: bool = True,
    eps: float = 1e-5,
) -> nn.Module:
    """Build a configured normalization module."""
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, elementwise_affine=elementwise_affine, eps=eps)
    if norm_type == "rmsnorm":
        return nn.RMSNorm(dim, elementwise_affine=elementwise_affine, eps=eps)
    raise ValueError(f"Unknown norm_type: {norm_type}")


def normalize_without_affine(
    norm: nn.Module,
    x: Float["... d"],
    norm_type: NormType,
) -> Float["... d"]:
    """Apply normalization without affine parameters."""
    if norm_type == "layernorm":
        return F.layer_norm(x, norm.normalized_shape, weight=None, bias=None, eps=norm.eps)
    if norm_type == "rmsnorm":
        return F.rms_norm(x, norm.normalized_shape, weight=None, eps=norm.eps)
    raise ValueError(f"Unknown norm_type: {norm_type}")


def folded_norm_linear_params(
    norm: nn.Module,
    linear: nn.Linear,
    norm_type: NormType,
) -> tuple[Any, Any]:
    """Fold norm affine parameters into a following linear projection."""
    weight = linear.weight
    bias = linear.bias
    if norm_type == "layernorm":
        weight = linear.weight * norm.weight.unsqueeze(0)
        norm_bias = F.linear(norm.bias, linear.weight)
        bias = norm_bias if linear.bias is None else norm_bias + linear.bias
    elif norm_type == "rmsnorm":
        weight = linear.weight * norm.weight.unsqueeze(0)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
    return weight, bias
