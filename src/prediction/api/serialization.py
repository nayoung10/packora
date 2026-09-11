"""In-memory and CIF serialization for Packora predictions."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from ase.cell import Cell
from ase.data import atomic_masses, chemical_symbols
from einops import rearrange, reduce

from src.prediction.api.backend import PredictionArrays
from src.prediction.api.chemistry import FeaturizedInput


class SerializationError(ValueError):
    """Raised when a prediction cannot be serialized safely."""


@dataclass(frozen=True)
class SerializedPrediction:
    """Contain JSON-ready structure data, CIF text, and result summary."""

    structure: dict[str, object]
    cif: str
    summary: dict[str, object]


def safe_identifier(value: str) -> str:
    """Return a CIF- and filename-safe identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "packora_prediction"


def _atom_labels(atomic_numbers: np.ndarray) -> list[str]:
    """Return unique element-prefixed labels for one atom sequence."""
    counts: dict[str, int] = {}
    labels: list[str] = []
    for atomic_number in atomic_numbers.astype(np.int64, copy=False).tolist():
        symbol = chemical_symbols[int(atomic_number)]
        counts[symbol] = counts.get(symbol, 0) + 1
        labels.append(f"{symbol}{counts[symbol]}")
    return labels


def center_molecules_in_cell(
    cart_coords: np.ndarray,
    cell: np.ndarray,
    atomic_numbers: np.ndarray,
    membership: np.ndarray,
) -> np.ndarray:
    """Center the assembly, then wrap every molecular COM into the cell."""
    coords = np.asarray(cart_coords, dtype=np.float64)
    cell_array = np.asarray(cell, dtype=np.float64)
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    molecule_ids = np.asarray(membership, dtype=np.int64)
    if coords.shape != (numbers.shape[0], 3):
        raise SerializationError("Coordinates and atomic numbers do not align.")
    if molecule_ids.shape != numbers.shape or molecule_ids.size == 0:
        raise SerializationError("Molecule membership does not align with atoms.")
    if cell_array.shape != (3, 3) or abs(float(np.linalg.det(cell_array))) < 1e-8:
        raise SerializationError("Cannot wrap molecules into an invalid cell.")

    masses = np.asarray(atomic_masses[numbers], dtype=np.float64)
    fractional = np.matmul(coords, np.linalg.inv(cell_array))
    reimaged = np.array(fractional, copy=True)
    for molecule_id in np.unique(molecule_ids):
        mask = molecule_ids == molecule_id
        molecule_masses = masses[mask]
        molecule_com = reduce(
            reimaged[mask] * rearrange(molecule_masses, "n -> n 1"),
            "n c -> c",
            "sum",
        ) / float(reduce(molecule_masses, "n ->", "sum"))
        reimaged[mask] = reimaged[mask] - np.floor(molecule_com)

    combined_com = reduce(
        reimaged * rearrange(masses, "n -> n 1"),
        "n c -> c",
        "sum",
    ) / float(reduce(masses, "n ->", "sum"))
    centered = reimaged + (np.full(3, 0.5) - combined_com)
    wrapped = np.array(centered, copy=True)
    for molecule_id in np.unique(molecule_ids):
        mask = molecule_ids == molecule_id
        molecule_masses = masses[mask]
        molecule_com = reduce(
            wrapped[mask] * rearrange(molecule_masses, "n -> n 1"),
            "n c -> c",
            "sum",
        ) / float(reduce(molecule_masses, "n ->", "sum"))
        wrapped[mask] = wrapped[mask] - np.floor(molecule_com)
    return np.asarray(np.matmul(wrapped, cell_array), dtype=np.float64)


def _unique_bonds(featurized: FeaturizedInput) -> list[dict[str, int]]:
    """Return unique authoritative intramolecular bonds in atom order."""
    conditioning = featurized.conditioning
    bonds: list[dict[str, int]] = []
    for index in range(int(conditioning.bond_indices.shape[1])):
        left = int(conditioning.bond_indices[0, index])
        right = int(conditioning.bond_indices[1, index])
        if left >= right:
            continue
        bonds.append(
            {
                "start": left,
                "end": right,
                "type": int(conditioning.bond_types[index]),
                "stereochemistry": int(conditioning.bond_stereochemistry[index]),
            }
        )
    return bonds


def _mol2_bond_type(type_id: int) -> str:
    """Map Packora bond IDs to MOL2 bond labels."""
    return {2: "1", 3: "2", 4: "3", 5: "4", 6: "ar"}.get(type_id, "un")


def build_mol2(
    identifier: str,
    cart_coords: np.ndarray,
    atomic_numbers: np.ndarray,
    membership: np.ndarray,
    bonds: list[dict[str, int]],
) -> str:
    """Build a MOL2 model with fixed molecule membership and bond types."""
    labels = _atom_labels(atomic_numbers)
    lines = [
        "@<TRIPOS>MOLECULE",
        safe_identifier(identifier),
        f"{atomic_numbers.shape[0]} {len(bonds)} 0 0 0",
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for index, (label, atomic_number, position, molecule_id) in enumerate(
        zip(
            labels,
            atomic_numbers.tolist(),
            cart_coords,
            membership.tolist(),
            strict=True,
        ),
        start=1,
    ):
        symbol = chemical_symbols[int(atomic_number)]
        lines.append(
            f"{index} {label} {position[0]:.8f} {position[1]:.8f} "
            f"{position[2]:.8f} {symbol} {int(molecule_id) + 1} "
            f"MOL{int(molecule_id) + 1} 0.0"
        )
    lines.append("@<TRIPOS>BOND")
    for index, bond in enumerate(bonds, start=1):
        lines.append(
            f"{index} {bond['start'] + 1} {bond['end'] + 1} "
            f"{_mol2_bond_type(bond['type'])}"
        )
    return "\n".join(lines) + "\n"


def build_cif(
    identifier: str,
    cart_coords: np.ndarray,
    cell: np.ndarray,
    atomic_numbers: np.ndarray,
    bonds: list[dict[str, int]],
) -> str:
    """Build an essential P1 CIF with authoritative intramolecular bonds."""
    params = Cell(cell).cellpar()
    fractional = np.matmul(cart_coords, np.linalg.inv(cell))
    labels = _atom_labels(atomic_numbers)
    lines = [
        f"data_{safe_identifier(identifier)}",
        "_symmetry_cell_setting triclinic",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "_space_group_name_Hall 'P 1'",
        "loop_",
        "_symmetry_equiv_pos_site_id",
        "_symmetry_equiv_pos_as_xyz",
        "1 x,y,z",
        f"_cell_length_a {params[0]:.8f}",
        f"_cell_length_b {params[1]:.8f}",
        f"_cell_length_c {params[2]:.8f}",
        f"_cell_angle_alpha {params[3]:.8f}",
        f"_cell_angle_beta {params[4]:.8f}",
        f"_cell_angle_gamma {params[5]:.8f}",
        f"_cell_volume {abs(float(np.linalg.det(cell))):.8f}",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    for label, atomic_number, position in zip(
        labels,
        atomic_numbers.tolist(),
        fractional,
        strict=True,
    ):
        symbol = chemical_symbols[int(atomic_number)]
        lines.append(
            f"{label} {symbol} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f}"
        )
    if bonds:
        lines.extend(
            [
                "loop_",
                "_geom_bond_atom_site_label_1",
                "_geom_bond_atom_site_label_2",
                "_geom_bond_site_symmetry_1",
                "_geom_bond_site_symmetry_2",
            ]
        )
        for bond in bonds:
            lines.append(f"{labels[bond['start']]} {labels[bond['end']]} 1_555 1_555")
    return "\n".join(lines) + "\n"


def serialize_prediction(
    identifier: str,
    featurized: FeaturizedInput,
    prediction: PredictionArrays,
    model_id: str,
) -> SerializedPrediction:
    """Serialize one prediction for Python, CLI, and browser consumption."""
    membership = featurized.conditioning.membership.astype(np.int64, copy=False)
    display_coords = center_molecules_in_cell(
        prediction.cart_coords,
        prediction.cell,
        prediction.atomic_numbers,
        membership,
    )
    display_fractional = np.matmul(display_coords, np.linalg.inv(prediction.cell))
    bonds = _unique_bonds(featurized)
    mol2 = build_mol2(
        identifier,
        display_coords,
        prediction.atomic_numbers,
        membership,
        bonds,
    )
    cif = build_cif(
        identifier,
        display_coords,
        prediction.cell,
        prediction.atomic_numbers,
        bonds,
    )
    cell_parameters = Cell(prediction.cell).cellpar().tolist()
    components = [
        {
            "smiles": component.canonical_smiles,
            "ratio": component.ratio,
            "atom_count": component.atom_count,
        }
        for component in featurized.components
    ]
    z_prior = None
    if featurized.z_draw is not None:
        z_prior = {
            "probability": featurized.z_draw.probability,
            "unconstrained_probability": featurized.z_draw.unconstrained_probability,
            "excluded_probability_mass": featurized.z_draw.excluded_probability_mass,
            "sha256": featurized.z_draw.prior_sha256,
        }
    summary: dict[str, object] = {
        "model": model_id,
        "model_label": prediction.model_label,
        "checkpoint_id": prediction.checkpoint_id,
        "z": featurized.z_value,
        "z_source": featurized.z_source,
        "components": components,
        "formula_atom_count": featurized.formula_atom_count,
        "total_atom_count": featurized.total_atom_count,
        "num_molecules": featurized.num_molecules,
        "elapsed_seconds": prediction.elapsed_seconds,
        "cell_parameters": cell_parameters,
        "z_prior": z_prior,
    }
    structure: dict[str, object] = {
        "identifier": safe_identifier(identifier),
        "atomic_numbers": prediction.atomic_numbers.tolist(),
        "atomic_symbols": [
            chemical_symbols[int(number)]
            for number in prediction.atomic_numbers.tolist()
        ],
        "raw_cart_coords": prediction.cart_coords.tolist(),
        "cart_coords": display_coords.tolist(),
        "frac_coords": display_fractional.tolist(),
        "cell": prediction.cell.tolist(),
        "membership": membership.tolist(),
        "bonds": bonds,
        "mol2": mol2,
        "summary": summary,
    }
    return SerializedPrediction(structure=structure, cif=cif, summary=summary)
