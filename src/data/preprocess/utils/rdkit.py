"""RDKit conformer helpers for preprocessing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np


def _block_rdkit_logs() -> Any:
    """Suppress RDKit warning output during expected parse/UFF failures."""
    from rdkit import rdBase

    return rdBase.BlockLogs()


class RDKitConformerError(RuntimeError):
    """Raised when RDKit conformer generation cannot produce coordinates."""

    def __init__(self, reason: str, detail: str) -> None:
        """Initialize the RDKit conformer failure."""
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class RDKitConformerResult:
    """Coordinates and metadata for one generated RDKit conformer."""

    coords: np.ndarray
    metadata: dict[str, object]


def single_atom_conformer(component_index: int, seed: int) -> RDKitConformerResult:
    """Return origin coordinates for one single-atom component."""
    return RDKitConformerResult(
        coords=np.zeros((1, 3), dtype=np.float64),
        metadata={
            "component_index": component_index,
            "num_atoms": 1,
            "seed": seed,
            "conformer_id": None,
            "used_random_coords": False,
            "uff_status": "skipped_single_atom",
        },
    )


def _mol_from_sdf(sdf_block: str) -> Any:
    """Build a sanitized RDKit molecule from one SDF block."""
    from rdkit import Chem

    block_logs = _block_rdkit_logs()
    with TemporaryDirectory() as tmp_dir:
        sdf_path = Path(tmp_dir) / "component.sdf"
        sdf_path.write_text(sdf_block)
        try:
            supplier = Chem.SDMolSupplier(
                str(sdf_path),
                sanitize=True,
                removeHs=False,
                strictParsing=True,
            )
            mol = supplier[0] if len(supplier) > 0 else None
        except (RuntimeError, ValueError, OSError) as exc:
            raise RDKitConformerError("rdkit_sdf_parse_failed", str(exc)) from exc
        finally:
            del block_logs

    if mol is None:
        raise RDKitConformerError("rdkit_sdf_parse_failed", "RDKit returned None.")
    return mol


def _validate_atom_order(mol: Any, expected_atomic_numbers: Sequence[int]) -> None:
    """Validate RDKit atom count and element order."""
    rdkit_numbers = [int(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    if len(expected_atomic_numbers) != len(rdkit_numbers):
        raise RDKitConformerError(
            "rdkit_atom_count_mismatch",
            (
                f"Expected atoms={len(expected_atomic_numbers)} "
                f"RDKit atoms={len(rdkit_numbers)}."
            ),
        )
    if list(expected_atomic_numbers) == rdkit_numbers:
        return

    mismatch_idx = next(
        idx
        for idx, (expected_value, rdkit_value) in enumerate(
            zip(expected_atomic_numbers, rdkit_numbers)
        )
        if expected_value != rdkit_value
    )
    raise RDKitConformerError(
        "rdkit_atom_order_mismatch",
        (
            f"Atom {mismatch_idx}: expected atomic number "
            f"{expected_atomic_numbers[mismatch_idx]} != RDKit "
            f"{rdkit_numbers[mismatch_idx]}."
        ),
    )


def _etkdg_options(seed: int, use_random_coords: bool = False) -> Any:
    """Build deterministic ETKDGv3 options."""
    from rdkit.Chem import AllChem

    options = AllChem.ETKDGv3()
    options.clearConfs = True
    options.randomSeed = seed
    options.useRandomCoords = use_random_coords
    return options


def _embed_molecule(mol: Any, seed: int) -> tuple[int, bool]:
    """Embed one RDKit molecule with ETKDG and one random-coordinate retry."""
    from rdkit.Chem import AllChem

    block_logs = _block_rdkit_logs()
    try:
        conf_id = int(AllChem.EmbedMolecule(mol, _etkdg_options(seed)))
        if conf_id != -1:
            return conf_id, False
        conf_id = int(
            AllChem.EmbedMolecule(mol, _etkdg_options(seed, use_random_coords=True))
        )
    except (RuntimeError, ValueError) as exc:
        raise RDKitConformerError("rdkit_embed_exception", str(exc)) from exc
    finally:
        del block_logs

    return conf_id, True


def _uff_optimize(mol: Any, conf_id: int, max_iters: int) -> int | str:
    """Run UFF optimization and return its status without raising."""
    from rdkit.Chem import AllChem

    block_logs = _block_rdkit_logs()
    try:
        return int(
            AllChem.UFFOptimizeMolecule(
                mol,
                confId=conf_id,
                maxIters=max_iters,
            )
        )
    except (RuntimeError, ValueError) as exc:
        return f"{exc.__class__.__name__}: {exc}"
    finally:
        del block_logs


def _conformer_coords(mol: Any, conf_id: int) -> np.ndarray:
    """Return conformer coordinates as an array."""
    try:
        conformer = mol.GetConformer(conf_id)
    except ValueError as exc:
        raise RDKitConformerError("rdkit_missing_conformer", str(exc)) from exc

    coords = []
    for atom_idx in range(mol.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_idx)
        coords.append([position.x, position.y, position.z])
    return np.array(coords, dtype=np.float64)


def generate_rdkit_conformer_from_sdf(
    sdf_block: str,
    expected_atomic_numbers: Sequence[int],
    component_index: int,
    seed: int,
    relax_with_uff: bool,
    uff_max_iters: int,
) -> RDKitConformerResult:
    """Generate one RDKit conformer from a component SDF block."""
    if len(expected_atomic_numbers) == 1:
        return single_atom_conformer(component_index, seed)

    mol = _mol_from_sdf(sdf_block)
    _validate_atom_order(mol, expected_atomic_numbers)
    mol.RemoveAllConformers()
    conf_id, used_random_coords = _embed_molecule(mol, seed)
    if conf_id == -1:
        raise RDKitConformerError("rdkit_embed_failed", "ETKDG returned -1.")

    uff_status: int | str = (
        _uff_optimize(mol, conf_id, uff_max_iters)
        if relax_with_uff
        else "skipped_disabled"
    )
    return RDKitConformerResult(
        coords=_conformer_coords(mol, conf_id),
        metadata={
            "component_index": component_index,
            "num_atoms": int(mol.GetNumAtoms()),
            "seed": seed,
            "conformer_id": conf_id,
            "used_random_coords": used_random_coords,
            "relax_with_uff": relax_with_uff,
            "uff_status": uff_status,
        },
    )
