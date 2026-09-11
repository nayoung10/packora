"""Utilities for converting model output tensors to ASE Atoms and CIF files."""

import logging
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write

from src.utils.tensor_typing import Bool, Float, Int

logger = logging.getLogger(__name__)


def tensors_to_atoms(
    coords: Float['n 3'],
    lattice: Float['d'],
    atom_numbers: Int['n'],
    mask: Bool['n'],
) -> Atoms:
    """Build an ASE Atoms object from model output tensors."""
    lattice_np = lattice.detach().cpu().numpy()
    # 9-element lattice → reshape to 3×3 cell matrix; 6-element → pass as-is
    cell = lattice_np.reshape(3, 3) if lattice_np.shape[0] == 9 else lattice_np.tolist()
    return Atoms(
        numbers=atom_numbers[mask].cpu().numpy().tolist(),
        positions=coords[mask].detach().cpu().numpy(),
        cell=cell,
        pbc=True,
    )


def tensors_to_atoms_traj(
    coords: Float['n 3'],
    lattice: Float['d'],
    atom_numbers: Int['n'],
    mask: Bool['n'],
) -> Atoms:
    """Build an ASE Atoms object from a trajectory frame."""
    return tensors_to_atoms(coords, lattice, atom_numbers, mask)


def save_structures_as_cif(
    output_dir: Path,
    results: dict[str, torch.Tensor],
) -> list[Path]:
    """Convert batched model outputs to individual CIF files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    coords = results["cart_coords"]
    lattices = results["lattice"]
    atom_numbers = results["atomic_numbers"]
    masks = results["atom_mask"]

    num_structures = coords.shape[0]
    paths: list[Path] = []

    for i in range(num_structures):
        atoms = tensors_to_atoms(coords[i], lattices[i], atom_numbers[i], masks[i])
        cif_path = output_dir / f"structure_{i:06d}.cif"
        write(str(cif_path), atoms)
        paths.append(cif_path)

    logger.info("Saved %d CIF files to %s", num_structures, output_dir)
    return paths
