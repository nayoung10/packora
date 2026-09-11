# ruff: noqa: F722,F821

import torch

from src.utils.tensor_typing import Float


def centered_gaussian(
    cart_coords: Float["b n 3"],
    atom_mask: Float["b n"],
    sigma: float = 1.0,
    center: bool = True,
) -> Float["b n 3"]:
    """Sample optionally centered Gaussian noise scaled by sigma."""
    # Sample isotropic Gaussian noise scaled by data-dependent sigma
    x0 = sigma * torch.randn_like(cart_coords)

    mask_expanded = atom_mask[:, :, None]
    if center:
        mean = torch.sum(x0 * mask_expanded, dim=1, keepdim=True) / torch.sum(
            mask_expanded, dim=1, keepdim=True
        ).clamp_min(1)
        x0 = x0 - mean

    # Zero out padded positions
    x0 = x0 * mask_expanded

    return x0
