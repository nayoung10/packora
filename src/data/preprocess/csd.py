"""Preprocessor for CSD molecular crystals."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.geometry import cellpar_to_cell

from src.data.constants import (
    BOND_TYPE_IDS,
    CSD_ATOM_CHIRALITY_IDS,
    CSD_BOND_STEREOCHEMISTRY_IDS,
    CSD_BOND_TYPE_IDS,
)
from src.data.preprocess.base import (
    BasePreprocessor,
    PreprocessContext,
    ProcessResult,
    RawEntryRef,
)
from src.data.preprocess.utils.rdkit import (
    RDKitConformerError,
    generate_rdkit_conformer_from_sdf,
    single_atom_conformer,
)
from src.data.types import Material, MoleculeConditioning

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ccdc.crystal import Crystal
    from ccdc.entry import Entry
    from ccdc.molecule import Atom, Bond, Molecule


class CSDPreprocessingError(RuntimeError):
    """Raised when one CSD entry cannot be converted safely."""


@dataclass(frozen=True)
class RDKitTemplateResult:
    """RDKit template coordinate generation output for one entry."""

    coords: np.ndarray
    present: bool
    failure_reason: str | None
    failure_detail: str | None
    component_results: list[dict[str, object]]


@dataclass(frozen=True)
class RDKitTemplateConfig:
    """Configuration for RDKit template conformer generation."""

    random_seed: int = 123
    relax_with_uff: bool = False
    uff_max_iters: int = 1000


@dataclass(frozen=True)
class CSDFilterConfig:
    """Filtering options for CSD entry selection."""

    date_cutoff: str | date | None = "2025-05-01"
    max_atoms: int | None = 512
    max_heavy_atoms: int | None = None
    max_r_factor: float | None = 9.0
    allow_powder: bool = False
    allow_polymeric: bool = False
    require_3d_coordinates: bool = True
    require_ambient_pressure: bool = True
    require_known_spacegroup: bool = True
    require_organic_or_organometallic: bool = True

    def __post_init__(self) -> None:
        """Normalize Hydra string dates."""
        if isinstance(self.date_cutoff, str):
            object.__setattr__(
                self,
                "date_cutoff",
                date.fromisoformat(self.date_cutoff),
            )


def _rdkit_template_config(
    config: RDKitTemplateConfig | Mapping[str, Any] | None,
) -> RDKitTemplateConfig:
    """Return normalized RDKit template configuration."""
    if config is None:
        return RDKitTemplateConfig()
    if isinstance(config, RDKitTemplateConfig):
        return config
    return RDKitTemplateConfig(**dict(config))


def _spacegroup_number(crystal: Crystal) -> int | None:
    """Return a standard International Tables space-group number."""
    try:
        spacegroup_number, _ = crystal.spacegroup_number_and_setting
    except RuntimeError:
        return None

    spacegroup_number = int(spacegroup_number)
    return None if spacegroup_number > 230 else spacegroup_number


def _spacegroup_symbol(crystal: Crystal) -> str | None:
    """Return the CSD space-group symbol when available."""
    try:
        return str(crystal.spacegroup_symbol)
    except RuntimeError:
        return None


def _conditioning_spacegroup(spacegroup_number: int | None) -> tuple[int, bool]:
    """Return persisted conditioning value and availability for a space group."""
    if spacegroup_number is None:
        return 0, False
    return int(spacegroup_number), True


def _atoms_from_components(molecules: Molecule) -> list[Atom]:
    """Return CSD atoms in component order."""
    return [atom for molecule in molecules.components for atom in molecule.atoms]


def _atom_labels_from_components(molecules: Molecule) -> list[str]:
    """Return CSD atom labels in component order."""
    return [
        str(getattr(atom, "label", atom_idx))
        for atom_idx, atom in enumerate(_atoms_from_components(molecules))
    ]


def _component_sizes(molecules: Molecule) -> list[int]:
    """Return atom counts for each packed CSD component."""
    return [len(molecule.atoms) for molecule in molecules.components]


def _component_smiles(molecules: Molecule) -> list[str | None]:
    """Return component SMILES strings in packed component order."""
    return [molecule.smiles for molecule in molecules.components]


def _component_rotatable_bond_counts(molecules: Molecule) -> list[int]:
    """Return rotatable-bond counts for each packed CSD component."""
    return [
        sum(1 for bond in molecule.bonds if _bond_rotatable_flag(bond))
        for molecule in molecules.components
    ]


def _flexibility_label(molecules: Molecule) -> str:
    """Return the rigid/flexible label from component rotatable-bond counts."""
    max_rotatable_bonds = max(_component_rotatable_bond_counts(molecules), default=0)
    return "rigid" if max_rotatable_bonds <= 3 else "flexible"


def _atom_arrays_from_components(molecules: Molecule) -> tuple[np.ndarray, np.ndarray]:
    """Return atomic numbers and Cartesian coordinates in component order."""
    atoms = _atoms_from_components(molecules)
    atomic_numbers = np.array([atom.atomic_number for atom in atoms], dtype=np.int32)
    cart_coords = np.array(
        [
            [atom.coordinates.x, atom.coordinates.y, atom.coordinates.z]
            for atom in atoms
        ],
        dtype=np.float64,
    )
    return atomic_numbers, cart_coords


def _component_atomic_numbers(component: Molecule) -> list[int]:
    """Return component atomic numbers in CCDC atom order."""
    return [int(atom.atomic_number) for atom in component.atoms]


def _component_sdf_block(component: Molecule) -> str:
    """Return one CCDC component as an SDF block."""
    try:
        return str(component.to_string("sdf"))
    except (RuntimeError, ValueError) as exc:
        raise RDKitConformerError("rdkit_sdf_export_failed", str(exc)) from exc


def _rdkit_template_coords_from_components(
    molecules: Molecule,
    config: RDKitTemplateConfig,
) -> RDKitTemplateResult:
    """Return RDKit template coordinates for all packed components."""
    total_atoms = sum(len(component.atoms) for component in molecules.components)
    component_coords: list[np.ndarray] = []
    component_results: list[dict[str, object]] = []

    for component_index, component in enumerate(molecules.components):
        try:
            atomic_numbers = _component_atomic_numbers(component)
            seed = config.random_seed + component_index
            if len(component.atoms) == 1:
                result = single_atom_conformer(component_index, seed)
            else:
                result = generate_rdkit_conformer_from_sdf(
                    _component_sdf_block(component),
                    atomic_numbers,
                    component_index,
                    seed,
                    config.relax_with_uff,
                    config.uff_max_iters,
                )
            component_coords.append(result.coords)
            component_results.append(result.metadata)
        except RDKitConformerError as exc:
            component_results.append(
                {
                    "component_index": component_index,
                    "num_atoms": len(component.atoms),
                    "reason": exc.reason,
                    "detail": exc.detail,
                }
            )
            return RDKitTemplateResult(
                coords=np.zeros((total_atoms, 3), dtype=np.float64),
                present=False,
                failure_reason=exc.reason,
                failure_detail=exc.detail,
                component_results=component_results,
            )

    coords = (
        np.concatenate(component_coords, axis=0)
        if component_coords
        else np.zeros((0, 3), dtype=np.float64)
    )
    return RDKitTemplateResult(
        coords=coords,
        present=True,
        failure_reason=None,
        failure_detail=None,
        component_results=component_results,
    )


def _cell_arrays_from_crystal(crystal: Crystal) -> tuple[np.ndarray, np.ndarray]:
    """Return lattice parameters and ASE-convention cell matrix."""
    lattice_parameters = np.array(
        [*crystal.cell_lengths, *crystal.cell_angles],
        dtype=np.float64,
    )
    ccdc_cell = np.array(
        crystal.fractional_to_orthogonal.rotation,
        dtype=np.float64,
    )
    ase_cell = cellpar_to_cell(lattice_parameters).astype(np.float64)

    if not np.allclose(ccdc_cell, ase_cell, atol=1e-8):
        raise CSDPreprocessingError("cell_mismatch_between_ccdc_and_ase")

    return lattice_parameters, ase_cell


def _csd_label_id(label: object, mapping: dict[str, int], fallback: int) -> int:
    """Map a CSD enum-like value to a compact integer ID."""
    return mapping.get(str(label), fallback)


def _component_atom_index(component: Molecule, atom: Atom) -> int:
    """Return the component-local index for a bond atom."""
    atom_index = int(atom.index)
    if 0 <= atom_index < len(component.atoms) and component.atoms[atom_index] == atom:
        return atom_index

    for fallback_index, component_atom in enumerate(component.atoms):
        if component_atom == atom:
            return fallback_index

    raise CSDPreprocessingError("bond_atom_not_in_component")


def _bond_type_id(bond: Bond) -> int:
    """Return the CSD-native integer label for one bond type."""
    return _csd_label_id(
        bond.bond_type,
        CSD_BOND_TYPE_IDS,
        BOND_TYPE_IDS["UNKNOWN"],
    )


def _atom_chirality_id(atom: Atom) -> int:
    """Return the CSD-native integer label for one atom chirality value."""
    return _csd_label_id(
        atom.chirality,
        CSD_ATOM_CHIRALITY_IDS,
        CSD_ATOM_CHIRALITY_IDS["Error"],
    )


def _bond_stereochemistry_id(bond: Bond) -> int:
    """Return the CSD-native integer label for one bond E/Z value."""
    return _csd_label_id(
        getattr(bond, "ez_stereochemistry", ""),
        CSD_BOND_STEREOCHEMISTRY_IDS,
        CSD_BOND_STEREOCHEMISTRY_IDS["Error"],
    )


def _bond_rotatable_flag(bond: Bond) -> bool:
    """Return whether CCDC considers one bond rotatable."""
    return bool(bond.is_rotatable)


def _bond_rotatable_flags(molecules: Molecule) -> list[bool]:
    """Return directed rotatable flags in CSD component bond order."""
    flags: list[bool] = []
    for molecule in molecules.components:
        for bond in molecule.bonds:
            is_rotatable = _bond_rotatable_flag(bond)
            flags.extend([is_rotatable, is_rotatable])
    return flags


def _failure_result(
    refcode: str,
    reason: str,
    stage: str | None,
    exc: Exception | None = None,
    skipped: bool = False,
) -> ProcessResult:
    """Build a failed ProcessResult with CSD debug metadata."""
    return ProcessResult(
        material=None,
        failure_reason=reason,
        skipped=skipped,
        refcode=refcode,
        failure_detail=None if exc is None else str(exc),
        stage=stage,
    )


class CSDPreprocessor(BasePreprocessor):
    """Preprocessor for CSD molecular crystals."""

    dataset_provided_splits = ("all",)

    def __init__(
        self,
        niggli: bool = True,
        filters: CSDFilterConfig | None = None,
        max_entries: int | None = None,
        rdkit_template: RDKitTemplateConfig | Mapping[str, Any] | None = None,
        deduplicate: Any | None = None,
        cache: Any | None = None,
        n_jobs: int = 1,
    ) -> None:
        """Initialize the CSD preprocessor."""
        super().__init__(primitive=False, niggli=niggli, n_jobs=n_jobs)
        self.filters = CSDFilterConfig() if filters is None else filters
        self.max_entries = max_entries
        self.rdkit_template = _rdkit_template_config(rdkit_template)
        self.deduplicate = deduplicate
        self.cache = cache

    def _entry_criteria(self, entry: Entry) -> Iterator[tuple[str, bool]]:
        """Yield entry-metadata filtering criteria."""
        filters = self.filters
        if filters.date_cutoff is not None:
            yield (
                "date_cutoff",
                entry.deposition_date is not None
                and entry.deposition_date <= filters.date_cutoff,
            )
        if filters.max_r_factor is not None:
            yield (
                "r_factor",
                entry.r_factor is not None
                and float(entry.r_factor) <= filters.max_r_factor,
            )
        if not filters.allow_powder:
            yield "powder", not bool(entry.is_powder_study)
        if not filters.allow_polymeric:
            yield "polymeric", not bool(entry.is_polymeric)
        if filters.require_3d_coordinates:
            yield "has_3d_coordinates", bool(entry.has_3d_structure)
        if filters.require_ambient_pressure:
            yield "ambient_pressure", entry.pressure is None
        if filters.require_organic_or_organometallic:
            yield (
                "organic_or_organometallic",
                bool(entry.is_organic) or bool(entry.is_organometallic),
            )

    def _crystal_molecule_criteria(
        self,
        spacegroup_number: int | None,
        molecules: Molecule,
    ) -> Iterator[tuple[str, bool]]:
        """Yield crystal and molecule filtering criteria."""
        filters = self.filters
        if filters.require_known_spacegroup:
            yield "known_spacegroup", spacegroup_number is not None
        if filters.max_atoms is not None:
            yield "max_atoms", len(molecules.atoms) <= filters.max_atoms

    def _filter_failure_reason(
        self,
        checks: Iterator[tuple[str, bool]],
    ) -> str | None:
        """Return the first failed filter reason."""
        for reason, passes in checks:
            if not passes:
                return reason
        return None

    def _standardize_bonds(self, molecules: Molecule) -> None:
        """Infer missing CSD bond types and standardize aromatic labels."""
        try:
            molecules.assign_bond_types(which="unknown")
            molecules.standardise_aromatic_bonds()
            molecules.standardise_delocalised_bonds()
        except (RuntimeError, ValueError) as exc:
            raise CSDPreprocessingError("csd_assign_bond_types_failed") from exc

    def _processed_crystal_and_molecules(
        self,
        entry: Entry,
    ) -> tuple[Crystal, Molecule]:
        """Return the Niggli-reduced crystal and packed CSD molecules."""
        from ccdc.crystal import Crystal

        crystal: Crystal = (
            Crystal.generate_reduced_crystal(entry.crystal)
            if self.niggli
            else entry.crystal
        )

        try:
            crystal.add_hydrogens(mode="missing", add_sites=True)
        except (RuntimeError, ValueError) as exc:
            raise CSDPreprocessingError("csd_add_hydrogens_failed") from exc

        molecules: Molecule = crystal.packing(inclusion="UniqueIncluded")
        self._standardize_bonds(molecules)
        return crystal, molecules

    def rotatable_flags_for_entry(self, entry: Entry) -> list[bool]:
        """Return directed CCDC rotatable-bond flags for one CSD entry."""
        _, molecules = self._processed_crystal_and_molecules(entry)
        return _bond_rotatable_flags(molecules)

    def _rdkit_template_result(self, molecules: Molecule) -> RDKitTemplateResult:
        """Generate RDKit template conditioning for packed molecules."""
        return _rdkit_template_coords_from_components(
            molecules,
            self.rdkit_template,
        )

    def _create_conditioning(
        self,
        molecules: Molecule,
        spacegroup_number: int | None,
        atomic_numbers: np.ndarray,
    ) -> tuple[MoleculeConditioning, dict[str, object], list[bool], dict[str, object]]:
        """Build MoleculeConditioning for one CSD entry."""
        senders: list[int] = []
        receivers: list[int] = []
        bond_types: list[int] = []
        bond_stereochemistry: list[int] = []
        bond_is_rotatable: list[bool] = []
        formal_charges: list[int] = []
        atom_chirality: list[int] = []
        membership: list[int] = []

        bond_type_counts: Counter[str] = Counter()
        stereo_counts: Counter[str] = Counter()
        chirality_counts: Counter[str] = Counter()
        atom_offset = 0

        for mol_idx, molecule in enumerate(molecules.components):
            for atom in molecule.atoms:
                formal_charges.append(int(atom.formal_charge))
                atom_chirality.append(_atom_chirality_id(atom))
                chirality_counts[str(atom.chirality) or "ACHIRAL"] += 1
                membership.append(mol_idx)

            for bond in molecule.bonds:
                begin = atom_offset + _component_atom_index(molecule, bond.atoms[0])
                end = atom_offset + _component_atom_index(molecule, bond.atoms[1])
                bond_type = _bond_type_id(bond)
                stereo = _bond_stereochemistry_id(bond)
                is_rotatable = _bond_rotatable_flag(bond)

                senders.extend([begin, end])
                receivers.extend([end, begin])
                bond_types.extend([bond_type, bond_type])
                bond_stereochemistry.extend([stereo, stereo])
                bond_is_rotatable.extend([is_rotatable, is_rotatable])
                bond_type_counts[str(bond.bond_type)] += 1
                stereo_counts[
                    str(getattr(bond, "ez_stereochemistry", "")) or "ACHIRAL"
                ] += 1

            atom_offset += len(molecule.atoms)

        rdkit_template = self._rdkit_template_result(molecules)
        conditioning_spacegroup, spacegroup_present = _conditioning_spacegroup(
            spacegroup_number
        )
        conditioning = MoleculeConditioning(
            atomic_numbers=atomic_numbers,
            bond_indices=np.array([senders, receivers], dtype=np.int32),
            bond_types=np.array(bond_types, dtype=np.int32),
            formal_charges=np.array(formal_charges, dtype=np.int32),
            template_coords=rdkit_template.coords,
            template_present=rdkit_template.present,
            membership=np.array(membership, dtype=np.int32),
            spacegroup_number=conditioning_spacegroup,
            spacegroup_present=spacegroup_present,
            atom_chirality=np.array(atom_chirality, dtype=np.int32),
            bond_stereochemistry=np.array(bond_stereochemistry, dtype=np.int32),
            stereochemistry_present=True,
            num_molecules=len(molecules.components),
        )
        return (
            conditioning,
            {
                "bond_type_counts": dict(bond_type_counts),
                "atom_chirality_counts": dict(chirality_counts),
                "bond_stereochemistry_counts": dict(stereo_counts),
                "unknown_bonds": int(bond_type_counts.get("Unknown", 0)),
                "template_present": rdkit_template.present,
            },
            bond_is_rotatable,
            {
                "present": rdkit_template.present,
                "failure_reason": rdkit_template.failure_reason,
                "failure_detail": rdkit_template.failure_detail,
                "random_seed": self.rdkit_template.random_seed,
                "relax_with_uff": self.rdkit_template.relax_with_uff,
                "uff_max_iters": self.rdkit_template.uff_max_iters,
                "components": rdkit_template.component_results,
            },
        )

    def _material_info(
        self,
        entry: Entry,
        crystal: Crystal,
        molecules: Molecule,
        context: PreprocessContext,
        conditioning_summary: dict[str, object],
        bond_is_rotatable: list[bool],
        rdkit_template: dict[str, object],
        spacegroup_number: int | None,
        spacegroup_symbol: str | None,
    ) -> dict[str, object]:
        """Build CSD metadata requested for persisted Material objects."""
        return {
            "dataset_name": context.dataset_name,
            "material_id": entry.identifier,
            "csd_refcode": entry.identifier,
            "csd_family": entry.identifier[:6].upper(),
            "chemical_name": entry.chemical_name,
            "formula": entry.formula,
            "is_organic": entry.is_organic,
            "is_organometallic": entry.is_organometallic,
            "has_disorder": entry.has_disorder,
            "disorder_details": entry.disorder_details,
            "pressure": entry.pressure,
            "temperature": entry.temperature,
            "deposition_date": (
                None
                if entry.deposition_date is None
                else entry.deposition_date.isoformat()
            ),
            "r_factor": entry.r_factor,
            "polymorph": entry.polymorph,
            "spacegroup": spacegroup_number,
            "spacegroup_number": spacegroup_number,
            "spacegroup_symbol": spacegroup_symbol,
            "z_value": crystal.z_value,
            "z_prime": crystal.z_prime,
            "num_atoms": len(molecules.atoms),
            "num_molecules": len(molecules.components),
            "csd_atom_labels": _atom_labels_from_components(molecules),
            "csd_component_sizes": _component_sizes(molecules),
            "csd_component_smiles": _component_smiles(molecules),
            "flexibility": _flexibility_label(molecules),
            "bond_is_rotatable": bond_is_rotatable,
            "conditioning_summary": conditioning_summary,
            "rdkit_template": rdkit_template,
        }

    def iter_raw_entries(self, context: PreprocessContext) -> list[RawEntryRef]:
        """Return CSD entry references for the requested split."""
        from ccdc import io

        if context.split not in self.dataset_provided_splits:
            raise ValueError(f"CSD only supports the all split, got {context.split!r}")

        reader = io.EntryReader("CSD")
        total = (
            len(reader)
            if self.max_entries is None
            else min(len(reader), self.max_entries)
        )
        logger.info("CSD %s split: scanning %d entries", context.split, total)
        return [
            RawEntryRef(key=reader.identifier(idx), source_split=str(context.split))
            for idx in range(total)
        ]

    def raw_entry_to_material(
        self,
        ref: RawEntryRef,
        context: PreprocessContext,
    ) -> ProcessResult:
        """Convert one CSD entry into a Material."""
        from ccdc import io

        refcode = ref.key
        try:
            entry = io.EntryReader("CSD").entry(ref.key)
            refcode = entry.identifier
            spacegroup_number = _spacegroup_number(entry.crystal)
            spacegroup_symbol = _spacegroup_symbol(entry.crystal)

            failure_reason = self._filter_failure_reason(self._entry_criteria(entry))
            if failure_reason:
                return _failure_result(
                    refcode,
                    failure_reason,
                    "filter",
                    skipped=True,
                )

            crystal, molecules = self._processed_crystal_and_molecules(entry)

            failure_reason = self._filter_failure_reason(
                self._crystal_molecule_criteria(spacegroup_number, molecules)
            )
            if failure_reason:
                return _failure_result(
                    refcode,
                    failure_reason,
                    "filter",
                    skipped=True,
                )

            if spacegroup_number is None and self.filters.require_known_spacegroup:
                return _failure_result(
                    refcode,
                    "known_spacegroup",
                    "filter",
                    skipped=True,
                )

            atomic_numbers, cart_coords = _atom_arrays_from_components(molecules)
            if np.any(atomic_numbers <= 0):
                return _failure_result(
                    refcode,
                    "unknown_atomic_numbers",
                    "filter",
                    skipped=True,
                )

            (
                conditioning,
                conditioning_summary,
                bond_is_rotatable,
                rdkit_template,
            ) = self._create_conditioning(
                molecules,
                spacegroup_number,
                atomic_numbers,
            )
            lattice_parameters, cell = _cell_arrays_from_crystal(crystal)
            frac_coords = cart_coords @ np.linalg.inv(cell)

            material = Material(
                cart_coords=cart_coords,
                frac_coords=frac_coords,
                lattice_parameters=lattice_parameters,
                cell=cell,
                conditioning=conditioning,
                info=self._material_info(
                    entry,
                    crystal,
                    molecules,
                    context,
                    conditioning_summary,
                    bond_is_rotatable,
                    rdkit_template,
                    spacegroup_number,
                    spacegroup_symbol,
                ),
            )
        except CSDPreprocessingError as exc:
            return _failure_result(refcode, str(exc), "conditioning", exc)
        except (RuntimeError, ValueError) as exc:
            return _failure_result(refcode, "csd_conversion_failed", None, exc)

        return ProcessResult(material=material, refcode=refcode)
