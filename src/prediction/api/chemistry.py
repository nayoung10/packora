"""Training-aligned SMILES featurization for Packora prediction."""

from __future__ import annotations

# RDKit must load before CCDC because both packages ship native chemistry libraries.
from rdkit import Chem

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from ccdc.molecule import Molecule

from src.data.preprocess.csd import CSDPreprocessor, RDKitTemplateConfig
from src.data.preprocess.utils.rdkit import _embed_molecule
from src.data.types import MoleculeConditioning
from src.prediction.api.prior import EmpiricalZPrior, ZDraw
from src.prediction.api.schemas import ComponentRequest, PredictionRequest


class ChemistryError(ValueError):
    """Raised when a molecular request cannot be featurized safely."""


@dataclass(frozen=True)
class ParsedComponent:
    """Hold one canonical CCDC molecule before stoichiometric expansion."""

    canonical_smiles: str
    molecule: Molecule
    atom_count: int
    requested_ratio: int


@dataclass(frozen=True)
class ResolvedComponent:
    """Describe one normalized formula-unit component."""

    canonical_smiles: str
    atom_count: int
    ratio: int


@dataclass(frozen=True)
class FeaturizedInput:
    """Contain the exact conditioning and provenance for one prediction."""

    conditioning: MoleculeConditioning
    components: tuple[ResolvedComponent, ...]
    z_value: int
    z_source: str
    formula_atom_count: int
    total_atom_count: int
    num_molecules: int
    conditioning_summary: dict[str, object]
    template_metadata: dict[str, object]
    z_draw: ZDraw | None


def _validated_rdkit_molecule(smiles: str, seed: int) -> Any:
    """Build one explicit-H 3D RDKit molecule from a connected SMILES."""
    if "." in smiles:
        raise ChemistryError(
            "Dot-disconnected SMILES are not accepted. Submit each component "
            "in a separate row with its own stoichiometric ratio."
        )
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ChemistryError(f"RDKit could not parse SMILES: {smiles!r}")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ChemistryError(
            "Each SMILES row must describe exactly one connected component."
        )
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise ChemistryError(f"RDKit could not canonicalize SMILES: {smiles!r}")
    molecule = Chem.AddHs(molecule)
    conformer_id, _ = _embed_molecule(molecule, int(seed))
    if conformer_id == -1:
        raise ChemistryError(
            f"RDKit ETKDG could not generate a molecular template for {smiles!r}."
        )
    return molecule


def _ccdc_component(component: ComponentRequest, seed: int) -> ParsedComponent:
    """Round-trip one explicit-H molecule through CCDC bond standardization."""
    rdkit_molecule = _validated_rdkit_molecule(component.smiles, seed)
    mol_block = Chem.MolToMolBlock(
        rdkit_molecule,
        includeStereo=True,
        kekulize=False,
        forceV3000=False,
    )
    try:
        molecule = Molecule.from_string(f"{mol_block}\n$$$$\n", format="sdf")
        molecule.assign_bond_types(which="unknown")
        molecule.standardise_aromatic_bonds()
        molecule.standardise_delocalised_bonds()
    except (RuntimeError, ValueError) as exc:
        raise ChemistryError(
            f"CCDC could not standardize component {component.smiles!r}: {exc}"
        ) from exc
    if len(molecule.components) != 1 or not molecule.atoms:
        raise ChemistryError(
            f"CCDC did not produce one connected component for {component.smiles!r}."
        )
    canonical_smiles = str(molecule.smiles or "").strip()
    if not canonical_smiles:
        raise ChemistryError(
            f"CCDC did not produce a canonical SMILES for {component.smiles!r}."
        )
    return ParsedComponent(
        canonical_smiles=canonical_smiles,
        molecule=molecule,
        atom_count=len(molecule.atoms),
        requested_ratio=int(component.ratio),
    )


def _merge_and_normalize(
    components: Sequence[ParsedComponent],
) -> tuple[list[ParsedComponent], list[int]]:
    """Merge duplicate canonical components and reduce ratios by their GCD."""
    merged: dict[str, ParsedComponent] = {}
    ratios: dict[str, int] = {}
    for component in components:
        key = component.canonical_smiles
        if key not in merged:
            merged[key] = component
            ratios[key] = 0
        ratios[key] += int(component.requested_ratio)
    divisor = math.gcd(*(ratios[key] for key in merged))
    normalized = [int(ratios[key] // divisor) for key in merged]
    return list(merged.values()), normalized


def _resolve_z(
    request: PredictionRequest,
    prior: EmpiricalZPrior | None,
    formula_atom_count: int,
    max_atoms: int,
    seed: int,
) -> tuple[int, str, ZDraw | None]:
    """Resolve explicit Z or draw from the conditioned empirical prior."""
    if request.z is not None:
        z_value = int(request.z)
        if formula_atom_count * z_value > max_atoms:
            raise ChemistryError(
                f"Explicit Z={z_value} produces {formula_atom_count * z_value} "
                f"atoms, above the {max_atoms}-atom prediction limit."
            )
        return z_value, "explicit", None
    if prior is None:
        raise ChemistryError("An empirical Z prior is required when Z is omitted.")
    try:
        draw = prior.draw(
            seed=int(seed),
            atoms_per_formula=formula_atom_count,
            max_atoms=max_atoms,
        )
    except ValueError as exc:
        raise ChemistryError(str(exc)) from exc
    return draw.value, "empirical_prior", draw


def featurize_request(
    request: PredictionRequest,
    prior: EmpiricalZPrior | None,
    max_atoms: int,
    seed: int,
) -> FeaturizedInput:
    """Convert a molecular request into training-aligned Packora conditioning."""
    parsed = [
        _ccdc_component(component, 123 + index)
        for index, component in enumerate(request.components)
    ]
    components, ratios = _merge_and_normalize(parsed)
    formula_atom_count = sum(
        component.atom_count * ratio
        for component, ratio in zip(components, ratios, strict=True)
    )
    z_value, z_source, z_draw = _resolve_z(
        request,
        prior,
        formula_atom_count,
        max_atoms,
        seed,
    )

    aggregate = Molecule("packora_input")
    resolved_components: list[ResolvedComponent] = []
    for component, ratio in zip(components, ratios, strict=True):
        resolved_components.append(
            ResolvedComponent(
                canonical_smiles=component.canonical_smiles,
                atom_count=component.atom_count,
                ratio=ratio,
            )
        )
        for _ in range(ratio * z_value):
            aggregate.add_molecule(component.molecule)

    atomic_numbers = np.asarray(
        [int(atom.atomic_number) for atom in aggregate.atoms],
        dtype=np.int32,
    )
    preprocessor = CSDPreprocessor(
        rdkit_template=RDKitTemplateConfig(
            random_seed=123,
            relax_with_uff=False,
            uff_max_iters=1000,
        ),
        n_jobs=1,
    )
    try:
        conditioning, summary, _, template_metadata = preprocessor._create_conditioning(  # noqa: SLF001
            aggregate,
            None,
            atomic_numbers,
        )
    except (RuntimeError, ValueError) as exc:
        raise ChemistryError(f"Training-time conditioning failed: {exc}") from exc
    if not conditioning.template_present:
        reason = template_metadata.get("failure_reason", "unknown error")
        detail = template_metadata.get("failure_detail")
        raise ChemistryError(
            f"Training-time RDKit template generation failed: {reason}"
            + ("" if detail is None else f" ({detail})")
        )
    total_atom_count = int(atomic_numbers.shape[0])
    if total_atom_count != formula_atom_count * z_value:
        raise ChemistryError("Expanded atom count does not match stoichiometry and Z.")
    if total_atom_count > max_atoms:
        raise ChemistryError(
            f"Expanded input has {total_atom_count} atoms, above the "
            f"{max_atoms}-atom prediction limit."
        )
    return FeaturizedInput(
        conditioning=conditioning,
        components=tuple(resolved_components),
        z_value=z_value,
        z_source=z_source,
        formula_atom_count=formula_atom_count,
        total_atom_count=total_atom_count,
        num_molecules=int(conditioning.num_molecules),
        conditioning_summary=summary,
        template_metadata=template_metadata,
        z_draw=z_draw,
    )
