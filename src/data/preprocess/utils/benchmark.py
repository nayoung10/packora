"""Benchmark-carving helpers for CSD preprocessing."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.types import Material


@dataclass(frozen=True)
class BenchmarkCsvConfig:
    """Configuration for one benchmark refcode CSV."""

    name: str
    path: Path
    refcode_column: str


@dataclass(frozen=True)
class ComponentFilterConfig:
    """Configuration for benchmark component-overlap eligibility."""

    min_component_heavy_atoms: int = 8
    ignore_solvents: bool = True


@dataclass(frozen=True)
class BenchmarkRefcode:
    """One benchmark refcode with source metadata."""

    refcode: str
    family: str
    source: str


@dataclass(frozen=True)
class FamilyExclusion:
    """One material excluded by benchmark CSD-family overlap."""

    refcode: str
    family: str
    reason: str
    benchmark_refcodes: list[str]
    benchmark_sources: list[str]


@dataclass(frozen=True)
class ComponentRecord:
    """One eligible CSD component used for overlap matching."""

    refcode: str
    family: str
    source: str
    component_index: int
    smiles: str
    heavy_atoms: int


@dataclass(frozen=True)
class ComponentSkip:
    """One CSD component skipped before overlap matching."""

    refcode: str
    family: str
    source: str
    component_index: int
    reason: str
    smiles: str | None = None
    heavy_atoms: int | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ComponentMatch:
    """One component match that excludes a candidate material."""

    smiles: str
    heavy_atoms: int
    benchmark_refcodes: list[str]
    benchmark_sources: list[str]


@dataclass(frozen=True)
class ComponentExclusion:
    """One material excluded by benchmark component overlap."""

    refcode: str
    family: str
    reason: str
    matches: list[ComponentMatch]


def refcodes_from_value(value: object) -> list[str]:
    """Return uppercase refcodes from a possibly semicolon-separated value."""
    refcodes: list[str] = []
    for item in str(value or "").split(";"):
        refcode = item.strip().upper()
        if refcode:
            refcodes.append(refcode)
    return refcodes


def load_benchmark_refcodes(
    csv_configs: Sequence[BenchmarkCsvConfig],
) -> list[BenchmarkRefcode]:
    """Load benchmark refcodes from configured CSV files."""
    records: list[BenchmarkRefcode] = []
    seen: set[tuple[str, str]] = set()
    for csv_config in csv_configs:
        with csv_config.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if csv_config.refcode_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"Benchmark CSV must contain {csv_config.refcode_column!r}: "
                    f"{csv_config.path}"
                )
            for row in reader:
                for refcode in refcodes_from_value(row[csv_config.refcode_column]):
                    key = (refcode, csv_config.name)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(
                        BenchmarkRefcode(
                            refcode=refcode,
                            family=refcode[:6],
                            source=csv_config.name,
                        )
                    )
    return records


def build_family_index(
    records: Sequence[BenchmarkRefcode],
) -> dict[str, list[BenchmarkRefcode]]:
    """Return benchmark refcodes grouped by six-letter CSD family."""
    index: dict[str, list[BenchmarkRefcode]] = {}
    for record in records:
        index.setdefault(record.family, []).append(record)
    return index


def material_refcode(material: Material) -> str:
    """Return the persisted CSD refcode for one material."""
    refcode = (material.info or {}).get("csd_refcode")
    return str(refcode) if refcode is not None else "unknown"


def material_family(material: Material) -> str:
    """Return the persisted CSD family for one material."""
    refcode = material_refcode(material)
    if not refcode:
        raise ValueError("Material is missing info['csd_refcode'].")
    return refcode[:6].upper()


def filter_materials_by_family(
    materials: Sequence[Material],
    family_index: Mapping[str, Sequence[BenchmarkRefcode]],
) -> tuple[list[Material], list[FamilyExclusion]]:
    """Remove materials whose CSD family belongs to the benchmark set."""
    kept: list[Material] = []
    exclusions: list[FamilyExclusion] = []
    for material in materials:
        family = material_family(material)
        benchmark_records = list(family_index.get(family, []))
        if not benchmark_records:
            kept.append(material)
            continue
        exclusions.append(
            FamilyExclusion(
                refcode=material_refcode(material),
                family=family,
                reason="benchmark_refcode_family",
                benchmark_refcodes=sorted(
                    {record.refcode for record in benchmark_records}
                ),
                benchmark_sources=sorted(
                    {record.source for record in benchmark_records}
                ),
            )
        )
    return kept, exclusions


def load_ccdc_solvent_smiles() -> set[str]:
    """Load exact CCDC solvent SMILES from the bundled solvent library."""
    from ccdc import io
    from ccdc.utilities import Resources

    solvent_smiles: set[str] = set()
    for solvent_file in Resources().get_ccdc_solvents_dir().glob("*.mol2"):
        molecule = io.MoleculeReader(str(solvent_file))[0]
        solvent_smiles.add(str(molecule.smiles))
    return solvent_smiles


def component_is_eligible(
    smiles: str | None,
    heavy_atoms: int,
    solvent_smiles: set[str],
    config: ComponentFilterConfig,
) -> tuple[bool, str | None]:
    """Return whether one component should be used for overlap matching."""
    if smiles is None or smiles == "":
        return False, "missing_smiles"
    if config.ignore_solvents and smiles in solvent_smiles:
        return False, "solvent"
    if heavy_atoms < config.min_component_heavy_atoms:
        return False, "min_component_heavy_atoms"
    return True, None


def ccdc_component_smiles(component: object) -> str | None:
    """Return the CCDC SMILES property for one molecular component."""
    value = component.smiles
    return None if value is None else str(value)


def _component_heavy_atom_counts(material: Material) -> dict[int, int]:
    """Return heavy-atom counts keyed by component membership id."""
    atomic_numbers = np.asarray(material.conditioning.atomic_numbers)
    membership = np.asarray(material.conditioning.membership)
    if atomic_numbers.shape[0] != membership.shape[0]:
        raise ValueError(
            f"Material {material_refcode(material)} has inconsistent membership."
        )
    counts: dict[int, int] = {}
    for component_index in sorted({int(value) for value in membership.tolist()}):
        mask = membership == component_index
        counts[component_index] = int(np.count_nonzero(atomic_numbers[mask] > 1))
    return counts


def material_component_records(
    material: Material,
    solvent_smiles: set[str],
    config: ComponentFilterConfig,
) -> tuple[list[ComponentRecord], list[ComponentSkip]]:
    """Return eligible component records and skipped diagnostics for a material."""
    info = material.info or {}
    raw_smiles = info.get("csd_component_smiles", [])
    if not isinstance(raw_smiles, Sequence) or isinstance(raw_smiles, str):
        raw_smiles = []
    heavy_counts = _component_heavy_atom_counts(material)
    records: list[ComponentRecord] = []
    skips: list[ComponentSkip] = []
    refcode = material_refcode(material)
    family = material_family(material)
    source = str(info.get("dataset_name", "candidate"))
    for component_index, value in enumerate(raw_smiles):
        smiles = None if value is None else str(value)
        heavy_atoms = heavy_counts.get(component_index, 0)
        eligible, reason = component_is_eligible(
            smiles,
            heavy_atoms,
            solvent_smiles,
            config,
        )
        if not eligible:
            skips.append(
                ComponentSkip(
                    refcode=refcode,
                    family=family,
                    source=source,
                    component_index=component_index,
                    reason=str(reason),
                    smiles=smiles,
                    heavy_atoms=heavy_atoms,
                )
            )
            continue
        records.append(
            ComponentRecord(
                refcode=refcode,
                family=family,
                source=source,
                component_index=component_index,
                smiles=str(smiles),
                heavy_atoms=heavy_atoms,
            )
        )
    return records, skips


def extract_benchmark_components_from_csd(
    records: Sequence[BenchmarkRefcode],
    solvent_smiles: set[str],
    config: ComponentFilterConfig,
    preprocessor: object,
) -> tuple[list[ComponentRecord], list[ComponentSkip]]:
    """Extract eligible benchmark component SMILES from CSD entries."""
    from ccdc import io

    reader = io.EntryReader("CSD")
    processor = getattr(preprocessor, "_processed_crystal_and_molecules")
    components: list[ComponentRecord] = []
    skips: list[ComponentSkip] = []
    seen: set[tuple[str, str, str]] = set()

    for record in records:
        try:
            entry = reader.entry(record.refcode)
            _, molecules = processor(entry)
        except (RuntimeError, ValueError) as exc:
            skips.append(
                ComponentSkip(
                    refcode=record.refcode,
                    family=record.family,
                    source=record.source,
                    component_index=-1,
                    reason="entry_load_failed",
                    detail=str(exc),
                )
            )
            continue

        for component_index, component in enumerate(molecules.components):
            try:
                smiles = ccdc_component_smiles(component)
            except (RuntimeError, ValueError) as exc:
                skips.append(
                    ComponentSkip(
                        refcode=record.refcode,
                        family=record.family,
                        source=record.source,
                        component_index=component_index,
                        reason="smiles_property_failed",
                        detail=str(exc),
                    )
                )
                continue
            heavy_atoms = sum(
                1 for atom in component.atoms if int(atom.atomic_number) > 1
            )
            eligible, reason = component_is_eligible(
                smiles,
                heavy_atoms,
                solvent_smiles,
                config,
            )
            if not eligible:
                skips.append(
                    ComponentSkip(
                        refcode=record.refcode,
                        family=record.family,
                        source=record.source,
                        component_index=component_index,
                        reason=str(reason),
                        smiles=smiles,
                        heavy_atoms=heavy_atoms,
                    )
                )
                continue
            eligible_smiles = str(smiles)
            key = (record.refcode, record.source, eligible_smiles)
            if key in seen:
                continue
            seen.add(key)
            components.append(
                ComponentRecord(
                    refcode=record.refcode,
                    family=record.family,
                    source=record.source,
                    component_index=component_index,
                    smiles=eligible_smiles,
                    heavy_atoms=heavy_atoms,
                )
            )
    return components, skips


def build_component_index(
    records: Sequence[ComponentRecord],
) -> dict[str, list[ComponentRecord]]:
    """Return eligible benchmark components grouped by exact CSD SMILES."""
    index: dict[str, list[ComponentRecord]] = {}
    for record in records:
        index.setdefault(record.smiles, []).append(record)
    return index


def filter_materials_by_components(
    materials: Sequence[Material],
    component_index: Mapping[str, Sequence[ComponentRecord]],
    solvent_smiles: set[str],
    config: ComponentFilterConfig,
) -> tuple[list[Material], list[ComponentExclusion], list[ComponentSkip]]:
    """Remove materials whose eligible CSD components overlap benchmark components."""
    kept: list[Material] = []
    exclusions: list[ComponentExclusion] = []
    all_skips: list[ComponentSkip] = []
    for material in materials:
        records, skips = material_component_records(material, solvent_smiles, config)
        all_skips.extend(skips)
        matches: list[ComponentMatch] = []
        seen_smiles: set[str] = set()
        for record in records:
            if record.smiles in seen_smiles:
                continue
            seen_smiles.add(record.smiles)
            benchmark_records = list(component_index.get(record.smiles, []))
            if not benchmark_records:
                continue
            matches.append(
                ComponentMatch(
                    smiles=record.smiles,
                    heavy_atoms=record.heavy_atoms,
                    benchmark_refcodes=sorted(
                        {benchmark.refcode for benchmark in benchmark_records}
                    ),
                    benchmark_sources=sorted(
                        {benchmark.source for benchmark in benchmark_records}
                    ),
                )
            )
        if not matches:
            kept.append(material)
            continue
        exclusions.append(
            ComponentExclusion(
                refcode=material_refcode(material),
                family=material_family(material),
                reason="benchmark_component",
                matches=matches,
            )
        )
    return kept, exclusions, all_skips


def records_to_dicts(records: Sequence[Any]) -> list[dict[str, Any]]:
    """Return dataclass records as JSON-serializable dictionaries."""
    return [asdict(record) for record in records]
