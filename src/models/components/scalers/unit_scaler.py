# ruff: noqa: F722,F821

import math

import torch

from src.models.components.lattice_repr import cell_to_ltri_latent, ltri_latent_to_cell
from src.models.components.scalers import Scaler
from src.utils.tensor_typing import Bool, Float


class UnitScaler(Scaler):
    """Convert angstroms to nanometers for lengths, degrees to radians for angles."""

    ANG_TO_NM: float = 0.1
    NM_TO_ANG: float = 10.0
    DEG_TO_RAD: float = math.pi / 180.0
    RAD_TO_DEG: float = 180.0 / math.pi

    def __init__(self, lattice_repr: str = "params") -> None:
        """Initialize with lattice representation choice."""
        self.lattice_repr = lattice_repr

    def scale_coords(
        self,
        cart_coords: Float["b n 3"],
        atom_mask: Bool["b n"] | None = None,
    ) -> Float["b n 3"]:
        """Convert Cartesian coordinates from angstroms to nanometers."""
        scaled = cart_coords * self.ANG_TO_NM
        if atom_mask is not None:
            scaled = scaled * atom_mask[..., None].to(dtype=scaled.dtype)
        return scaled

    def scale_lattice(
        self,
        lattice: Float["b d"],
    ) -> Float["b d"]:
        """Convert lattice to nanometers (cell) or nm+radians (params)."""
        if self.lattice_repr == "cell":
            return lattice * self.ANG_TO_NM
        if self.lattice_repr == "ltri":
            return cell_to_ltri_latent(lattice * self.ANG_TO_NM)

        lengths = lattice[:, :3] * self.ANG_TO_NM
        angles = lattice[:, 3:] * self.DEG_TO_RAD
        return torch.cat([lengths, angles], dim=-1)

    def unscale_coords(
        self,
        cart_coords: Float["b n 3"],
        atom_mask: Bool["b n"] | None = None,
    ) -> Float["b n 3"]:
        """Convert Cartesian coordinates from nanometers to angstroms."""
        unscaled = cart_coords * self.NM_TO_ANG
        if atom_mask is not None:
            unscaled = unscaled * atom_mask[..., None].to(dtype=unscaled.dtype)
        return unscaled

    def unscale_lattice(
        self,
        lattice: Float["b d"],
    ) -> Float["b d"]:
        """Convert lattice to angstroms (cell) or angstroms+degrees (params)."""
        if self.lattice_repr == "cell":
            return lattice * self.NM_TO_ANG
        if self.lattice_repr == "ltri":
            return ltri_latent_to_cell(lattice) * self.NM_TO_ANG

        lengths = lattice[:, :3] * self.NM_TO_ANG
        angles = lattice[:, 3:] * self.RAD_TO_DEG
        return torch.cat([lengths, angles], dim=-1)
