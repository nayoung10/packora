# ruff: noqa: F722,F821

import torch
from einops import rearrange

from src.utils.tensor_typing import Float


def _as_cell_matrix(cell: Float["b ..."]) -> Float["b 3 3"]:
    """Normalize a batched flattened or matrix cell tensor to matrix form."""
    if cell.ndim == 3 and cell.shape[-2:] == (3, 3):
        return cell
    if cell.ndim == 2 and cell.shape[-1] == 9:
        return rearrange(cell, "b (i j) -> b i j", i=3, j=3)
    raise ValueError(
        f"Expected cell shape (B, 3, 3) or (B, 9), got {tuple(cell.shape)}."
    )


def cell_to_ltri_latent(cell: Float["b ..."]) -> Float["b 6"]:
    """Convert raw cell matrices to Crystalite-style log-ltri latents."""
    cell_matrix = _as_cell_matrix(cell)
    calc_dtype = torch.float64 if cell_matrix.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=cell_matrix.device.type, enabled=False):
        cell_matrix = cell_matrix.to(dtype=calc_dtype)
        gram = cell_matrix @ rearrange(cell_matrix, "b i j -> b j i")
        chol = torch.linalg.cholesky(gram)
    diag = torch.diagonal(chol, dim1=-2, dim2=-1).clamp_min(1e-12)
    return torch.stack(
        [
            torch.log(diag[:, 0]),
            chol[:, 1, 0],
            torch.log(diag[:, 1]),
            chol[:, 2, 0],
            chol[:, 2, 1],
            torch.log(diag[:, 2]),
        ],
        dim=-1,
    )


def ltri_latent_to_cell(lattice: Float["b 6"]) -> Float["b 9"]:
    """Decode Crystalite-style log-ltri latents to flattened cell matrices."""
    if lattice.ndim != 2 or lattice.shape[-1] != 6:
        raise ValueError(f"Expected lattice shape (B, 6), got {tuple(lattice.shape)}.")
    calc_dtype = torch.float64 if lattice.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=lattice.device.type, enabled=False):
        lattice = lattice.to(dtype=calc_dtype)
        cell = torch.zeros(
            lattice.shape[0], 3, 3, dtype=lattice.dtype, device=lattice.device
        )
        cell[:, 0, 0] = torch.exp(lattice[:, 0])
        cell[:, 1, 0] = lattice[:, 1]
        cell[:, 1, 1] = torch.exp(lattice[:, 2])
        cell[:, 2, 0] = lattice[:, 3]
        cell[:, 2, 1] = lattice[:, 4]
        cell[:, 2, 2] = torch.exp(lattice[:, 5])
    return rearrange(cell, "b i j -> b (i j)")


def lattice_params_to_cell(lattice: Float["b 6"]) -> Float["b 9"]:
    """Convert lengths and degree angles to flattened cell matrices."""
    if lattice.ndim != 2 or lattice.shape[-1] != 6:
        raise ValueError(f"Expected lattice shape (B, 6), got {tuple(lattice.shape)}.")
    lengths = lattice[:, :3]
    angles = torch.deg2rad(lattice[:, 3:])
    a, b, c = lengths.unbind(dim=-1)
    alpha, beta, gamma = angles.unbind(dim=-1)

    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma).clamp_min(1e-12)

    row_a = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
    row_b = torch.stack(
        [b * cos_gamma, b * sin_gamma, torch.zeros_like(b)],
        dim=-1,
    )
    c_x = c * cos_beta
    c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    c_z = (c.square() - c_x.square() - c_y.square()).clamp_min(1e-12).sqrt()
    row_c = torch.stack([c_x, c_y, c_z], dim=-1)
    cell = torch.stack([row_a, row_b, row_c], dim=1)
    return rearrange(cell, "b i j -> b (i j)")


def cell_to_y1_features(cell: Float["b ..."]) -> Float["b 6"]:
    """Convert flattened or matrix cell tensors to log-length/cos-angle features."""
    cell_matrix = _as_cell_matrix(cell)
    gram = cell_matrix @ rearrange(cell_matrix, "b i j -> b j i")
    eps = torch.tensor(1e-12, device=cell_matrix.device, dtype=cell_matrix.dtype)
    diag = torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(eps)
    lengths = diag.sqrt()
    a, b, c = lengths.unbind(dim=-1)

    cos_gamma = (gram[:, 0, 1] / (a * b).clamp_min(eps)).clamp(-1.0, 1.0)
    cos_beta = (gram[:, 0, 2] / (a * c).clamp_min(eps)).clamp(-1.0, 1.0)
    cos_alpha = (gram[:, 1, 2] / (b * c).clamp_min(eps)).clamp(-1.0, 1.0)
    return torch.stack(
        [
            torch.log(a.clamp_min(eps)),
            torch.log(b.clamp_min(eps)),
            torch.log(c.clamp_min(eps)),
            cos_alpha,
            cos_beta,
            cos_gamma,
        ],
        dim=-1,
    )
