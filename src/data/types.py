# ruff: noqa: F722,F821
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from src.utils.tensor_typing import Bool, Float, Int


FIELD_NAMES = (
    "atomic_numbers",
    "template_coords",
    "template_present",
    "formal_charges",
    "atom_chirality",
    "membership",
    "bond_adj",
    "bond_type_adj",
    "bond_stereochemistry_adj",
    "stereochemistry_present",
    "spacegroup_number",
    "spacegroup_present",
)


@dataclass(frozen=True)
class ConditioningTensors:
    """Per-sample conditioning tensors with CSD-native labels."""

    atomic_numbers: Int["n"]
    template_coords: Float["n 3"]
    template_present: Bool[""]
    formal_charges: Int["n"]
    atom_chirality: Int["n"]
    membership: Int["n"]
    bond_adj: Bool["n n"]
    bond_type_adj: Int["n n"]
    bond_stereochemistry_adj: Int["n n"]
    stereochemistry_present: Bool[""]
    spacegroup_number: Int[""]
    spacegroup_present: Bool[""]


def _dense_pair_labels(
    bond_indices: np.ndarray,
    bond_types: np.ndarray,
    bond_stereochemistry: np.ndarray,
    n_atoms: int,
) -> tuple[Bool["n n"], Int["n n"], Int["n n"]]:
    """Convert sparse directed bonds into dense adjacency tensors."""
    bond_adj = torch.zeros(n_atoms, n_atoms, dtype=torch.bool)
    bond_type_adj = torch.zeros(n_atoms, n_atoms, dtype=torch.long)
    bond_stereo_adj = torch.zeros(n_atoms, n_atoms, dtype=torch.long)
    if bond_indices.size == 0:
        return bond_adj, bond_type_adj, bond_stereo_adj

    src = torch.from_numpy(bond_indices[0].astype(np.int64))
    dst = torch.from_numpy(bond_indices[1].astype(np.int64))
    types = torch.from_numpy(bond_types.astype(np.int64))
    stereo = torch.from_numpy(bond_stereochemistry.astype(np.int64))
    bond_adj[src, dst] = True
    bond_type_adj[src, dst] = types
    bond_stereo_adj[src, dst] = stereo
    return bond_adj, bond_type_adj, bond_stereo_adj


@dataclass
class MoleculeConditioning:
    """CSD-native molecular conditioning for one crystal."""

    atomic_numbers: np.ndarray
    bond_indices: np.ndarray
    bond_types: np.ndarray
    formal_charges: np.ndarray
    template_coords: np.ndarray
    template_present: bool
    membership: np.ndarray
    spacegroup_number: int
    spacegroup_present: bool
    atom_chirality: np.ndarray
    bond_stereochemistry: np.ndarray
    stereochemistry_present: bool
    num_molecules: int

    def to_tensors(self) -> ConditioningTensors:
        """Build full ConditioningTensors with all fields populated."""
        n_atoms = self.atomic_numbers.shape[0]
        bond_adj, bond_type_adj, bond_stereo_adj = _dense_pair_labels(
            self.bond_indices,
            self.bond_types,
            self.bond_stereochemistry,
            n_atoms,
        )
        return ConditioningTensors(
            atomic_numbers=torch.from_numpy(self.atomic_numbers.astype(np.int64)),
            template_coords=torch.from_numpy(self.template_coords.astype(np.float32)),
            template_present=torch.tensor(bool(self.template_present)),
            formal_charges=torch.from_numpy(self.formal_charges.astype(np.int64)),
            atom_chirality=torch.from_numpy(self.atom_chirality.astype(np.int64)),
            membership=torch.from_numpy(self.membership.astype(np.int64)),
            bond_adj=bond_adj,
            bond_type_adj=bond_type_adj,
            bond_stereochemistry_adj=bond_stereo_adj,
            stereochemistry_present=torch.tensor(bool(self.stereochemistry_present)),
            spacegroup_number=torch.tensor(
                int(self.spacegroup_number),
                dtype=torch.long,
            ),
            spacegroup_present=torch.tensor(bool(self.spacegroup_present)),
        )


@dataclass
class Material:
    """Molecular crystal structure container."""

    cart_coords: np.ndarray
    frac_coords: np.ndarray
    lattice_parameters: np.ndarray
    cell: np.ndarray
    conditioning: MoleculeConditioning
    info: Dict[str, Any] | None = None

    def conditioning_tensors(self) -> ConditioningTensors:
        """Build per-sample ConditioningTensors."""
        return self.conditioning.to_tensors()
