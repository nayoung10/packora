"""Activation helpers for model components."""

from typing import Literal

import torch.nn as nn

ActivationType = Literal["silu", "gelu", "gelu_tanh", "relu"]


def build_activation(activation: ActivationType = "silu") -> nn.Module:
    """Build a configured activation module."""
    if activation == "silu":
        return nn.SiLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "gelu_tanh":
        return nn.GELU(approximate="tanh")
    if activation == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {activation}")
