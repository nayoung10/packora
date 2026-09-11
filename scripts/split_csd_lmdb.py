"""Create CSD train/validation LMDB splits from an existing all.lmdb."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lmdb
import numpy as np

from src.data.preprocess.utils.benchmark import (
    ComponentFilterConfig,
    component_is_eligible,
    load_ccdc_solvent_smiles,
)
from src.data.preprocess.utils.lmdb import COMMIT_INTERVAL, LMDB_MAP_SIZE
from src.data.types import Material

logger = logging.getLogger(__name__)

DEFAULT_INPUT_LMDB = Path("data/csd/all.lmdb")
DEFAULT_VAL_FRACTION = 0.02
DEFAULT_SEED = 20250501
DEFAULT_MIN_COMPONENT_HEAVY_ATOMS = 8
DEFAULT_IGNORE_SOLVENTS = True
MANIFEST_VERSION = 2


@dataclass(frozen=True)
class MoleculeSplitConfig:
    """Configuration for molecule-overlap isolation."""

    min_component_heavy_atoms: int = DEFAULT_MIN_COMPONENT_HEAVY_ATOMS
    ignore_solvents: bool = DEFAULT_IGNORE_SOLVENTS
    molecule_key_source: str = "info.csd_component_smiles"


@dataclass(frozen=True)
class MaterialRow:
    """Lightweight metadata for one source LMDB row."""

    index: int
    refcode: str
    family: str
    num_atoms: int
    molecule_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class LmdbScanResult:
    """Metadata scanned from one source LMDB."""

    rows: list[MaterialRow]
    skipped_component_counts: Counter[str]


@dataclass(frozen=True)
class FamilyGroup:
    """One connected CSD-family component in the molecule graph."""

    group_id: int
    families: tuple[str, ...]
    entry_count: int
    molecule_keys: tuple[str, ...]


@dataclass(frozen=True)
class MoleculeFamilyGraph:
    """CSD-family graph induced by shared molecule keys."""

    groups: list[FamilyGroup]
    molecule_families: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class SplitSummary:
    """Summary of one proposed train/validation split."""

    total_entries: int
    train_entries: int
    val_entries: int
    total_families: int
    train_families: int
    val_families: int
    total_groups: int
    train_groups: int
    val_groups: int
    target_val_entries: int
    target_val_fraction: float
    actual_val_fraction: float
    seed: int


def _now_iso() -> str:
    """Return a UTC timestamp for manifest metadata."""
    return datetime.now(tz=timezone.utc).isoformat()


def _default_manifest_path(output_dir: Path) -> Path:
    """Return the default split manifest path for an output directory."""
    return output_dir / "splits" / "validation_manifest.json"


def _default_summary_path(output_dir: Path) -> Path:
    """Return the default split summary path for an output directory."""
    return output_dir / "splits" / "split_summary.json"


def _default_graph_path(output_dir: Path) -> Path:
    """Return the default molecule-family graph path for an output directory."""
    return output_dir / "splits" / "molecule_family_graph.json"


def _default_verification_path(output_dir: Path) -> Path:
    """Return the default split verification path for an output directory."""
    return output_dir / "splits" / "split_verification.json"


def _material_refcode(material: Material, index: int) -> str:
    """Return the normalized CSD refcode for one material."""
    refcode = (material.info or {}).get("csd_refcode")
    if refcode is None or str(refcode).strip() == "":
        raise ValueError(f"Material index {index} is missing info['csd_refcode'].")
    return str(refcode).strip().upper()


def refcode_family_from_material(material: Material, index: int) -> str:
    """Return the uppercase CSD family for one material."""
    return _material_refcode(material, index)[:6]


def _component_heavy_atom_counts(material: Material, index: int) -> dict[int, int]:
    """Return heavy-atom counts keyed by component membership id."""
    atomic_numbers = np.asarray(material.conditioning.atomic_numbers)
    membership = np.asarray(material.conditioning.membership)
    if atomic_numbers.shape[0] != membership.shape[0]:
        raise ValueError(f"Material index {index} has inconsistent membership.")
    if membership.size == 0:
        return {}
    if bool(np.any(membership < 0)):
        counts: dict[int, int] = {}
        for membership_id, atomic_number in zip(membership.tolist(), atomic_numbers):
            if int(atomic_number) > 1:
                key = int(membership_id)
                counts[key] = counts.get(key, 0) + 1
        return counts

    heavy_membership = membership[atomic_numbers > 1].astype(np.int64, copy=False)
    if heavy_membership.size == 0:
        return {}
    counts_array = np.bincount(heavy_membership)
    return {
        component_index: int(count)
        for component_index, count in enumerate(counts_array.tolist())
        if count > 0
    }


def molecule_keys_from_material(
    material: Material,
    index: int,
    solvent_smiles: set[str],
    config: MoleculeSplitConfig,
) -> tuple[tuple[str, ...], Counter[str]]:
    """Return eligible canonical molecule keys and skipped reasons."""
    raw_smiles = (material.info or {}).get("csd_component_smiles", [])
    if not isinstance(raw_smiles, Sequence) or isinstance(raw_smiles, str):
        return (), Counter({"invalid_component_smiles_metadata": 1})

    filter_config = ComponentFilterConfig(
        min_component_heavy_atoms=config.min_component_heavy_atoms,
        ignore_solvents=config.ignore_solvents,
    )
    heavy_counts = _component_heavy_atom_counts(material, index)
    molecule_keys: set[str] = set()
    skipped_counts: Counter[str] = Counter()
    for component_index, value in enumerate(raw_smiles):
        smiles = None if value is None else str(value)
        heavy_atoms = heavy_counts.get(component_index, 0)
        eligible, reason = component_is_eligible(
            smiles,
            heavy_atoms,
            solvent_smiles,
            filter_config,
        )
        if not eligible:
            skipped_counts[str(reason)] += 1
            continue
        molecule_keys.add(str(smiles))

    return tuple(sorted(molecule_keys)), skipped_counts


def row_from_material(
    index: int,
    material: Material,
    solvent_smiles: set[str] | None = None,
    config: MoleculeSplitConfig | None = None,
) -> tuple[MaterialRow, Counter[str]]:
    """Return lightweight row metadata for one material."""
    if config is None:
        config = MoleculeSplitConfig()
    if solvent_smiles is None:
        solvent_smiles = set()
    refcode = _material_refcode(material, index)
    molecule_keys, skipped_counts = molecule_keys_from_material(
        material,
        index,
        solvent_smiles,
        config,
    )
    return (
        MaterialRow(
            index=index,
            refcode=refcode,
            family=refcode[:6],
            num_atoms=int(material.conditioning.atomic_numbers.shape[0]),
            molecule_keys=molecule_keys,
        ),
        skipped_counts,
    )


def resolve_solvent_smiles(config: MoleculeSplitConfig) -> set[str]:
    """Return solvent SMILES used for molecule-overlap eligibility."""
    if not config.ignore_solvents:
        return set()
    return load_ccdc_solvent_smiles()


def read_lmdb_rows(
    lmdb_path: Path,
    config: MoleculeSplitConfig,
    log_interval: int = 100000,
) -> LmdbScanResult:
    """Read source LMDB row metadata needed for molecule-isolated splitting."""
    rows: list[MaterialRow] = []
    skipped_component_counts: Counter[str] = Counter()
    solvent_smiles = resolve_solvent_smiles(config)
    start_time = time.perf_counter()
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
    )
    try:
        with env.begin() as txn:
            total = int(txn.get(b"__len__").decode())
            for index in range(total):
                material: Material = pickle.loads(txn.get(f"{index:08d}".encode()))
                row, skipped_counts = row_from_material(
                    index,
                    material,
                    solvent_smiles,
                    config,
                )
                rows.append(row)
                skipped_component_counts.update(skipped_counts)
                if log_interval > 0 and (index + 1) % log_interval == 0:
                    elapsed = time.perf_counter() - start_time
                    rate = (index + 1) / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "Scanned %d/%d rows from %s (%.1f rows/s)",
                        index + 1,
                        total,
                        lmdb_path,
                        rate,
                    )
    finally:
        env.close()
    return LmdbScanResult(rows=rows, skipped_component_counts=skipped_component_counts)


def family_counts(rows: Sequence[MaterialRow]) -> Counter[str]:
    """Count source entries per CSD family."""
    return Counter(row.family for row in rows)


def _find(parent: dict[str, str], item: str) -> str:
    """Return the disjoint-set root for one item."""
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union(parent: dict[str, str], rank: dict[str, int], left: str, right: str) -> None:
    """Union two disjoint-set items."""
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return
    if rank[left_root] < rank[right_root]:
        left_root, right_root = right_root, left_root
    parent[right_root] = left_root
    if rank[left_root] == rank[right_root]:
        rank[left_root] += 1


def build_molecule_family_graph(rows: Sequence[MaterialRow]) -> MoleculeFamilyGraph:
    """Build connected CSD-family groups from shared molecule keys."""
    counts = family_counts(rows)
    parent = {family: family for family in counts}
    rank = {family: 0 for family in counts}
    molecule_family_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for molecule_key in row.molecule_keys:
            molecule_family_sets[molecule_key].add(row.family)

    for families in molecule_family_sets.values():
        ordered = sorted(families)
        for family in ordered[1:]:
            _union(parent, rank, ordered[0], family)

    root_families: dict[str, set[str]] = defaultdict(set)
    for family in counts:
        root_families[_find(parent, family)].add(family)

    root_molecules: dict[str, set[str]] = defaultdict(set)
    molecule_families: dict[str, tuple[str, ...]] = {}
    for molecule_key, families in molecule_family_sets.items():
        sorted_families = tuple(sorted(families))
        molecule_families[molecule_key] = sorted_families
        root_molecules[_find(parent, sorted_families[0])].add(molecule_key)

    groups: list[FamilyGroup] = []
    for group_id, (_, families) in enumerate(
        sorted(
            root_families.items(),
            key=lambda item: tuple(sorted(item[1])),
        )
    ):
        sorted_families = tuple(sorted(families))
        groups.append(
            FamilyGroup(
                group_id=group_id,
                families=sorted_families,
                entry_count=sum(counts[family] for family in sorted_families),
                molecule_keys=tuple(
                    sorted(root_molecules.get(_find(parent, sorted_families[0]), set()))
                ),
            )
        )
    return MoleculeFamilyGraph(groups=groups, molecule_families=molecule_families)


def select_validation_group_ids(
    groups: Sequence[FamilyGroup],
    total_entries: int,
    val_fraction: float,
    seed: int,
) -> set[int]:
    """Select connected groups until validation reaches the target count."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if total_entries == 0:
        raise ValueError("Cannot split an empty dataset.")

    target_entries = round(total_entries * val_fraction)
    rng = random.Random(seed)
    ordered_groups = sorted(groups, key=lambda group: group.families)
    rng.shuffle(ordered_groups)

    selected_group_ids: set[int] = set()
    selected_entries = 0
    for group in ordered_groups:
        if selected_entries >= target_entries:
            break
        selected_group_ids.add(group.group_id)
        selected_entries += group.entry_count
    return selected_group_ids


def val_families_from_groups(
    groups: Sequence[FamilyGroup],
    selected_group_ids: set[int],
) -> set[str]:
    """Return validation families contained in selected groups."""
    val_families: set[str] = set()
    for group in groups:
        if group.group_id in selected_group_ids:
            val_families.update(group.families)
    return val_families


def group_split(group: FamilyGroup, val_families: set[str]) -> str:
    """Return the split assignment for one connected family group."""
    group_families = set(group.families)
    val_count = len(group_families.intersection(val_families))
    if val_count == 0:
        return "train"
    if val_count == len(group_families):
        return "val"
    return "mixed"


def select_validation_families(
    counts: Counter[str],
    val_fraction: float,
    seed: int,
) -> set[str]:
    """Select whole standalone CSD families until validation reaches the target."""
    total_entries = sum(counts.values())
    groups = [
        FamilyGroup(
            group_id=group_id,
            families=(family,),
            entry_count=count,
            molecule_keys=(),
        )
        for group_id, (family, count) in enumerate(sorted(counts.items()))
    ]
    selected_group_ids = select_validation_group_ids(
        groups,
        total_entries,
        val_fraction,
        seed,
    )
    return val_families_from_groups(groups, selected_group_ids)


def split_summary(
    rows: Sequence[MaterialRow],
    val_families: set[str],
    val_fraction: float,
    seed: int,
    graph: MoleculeFamilyGraph | None = None,
) -> SplitSummary:
    """Return count metadata for one proposed split."""
    all_families = {row.family for row in rows}
    val_entries = sum(row.family in val_families for row in rows)
    train_entries = len(rows) - val_entries
    train_families = len(all_families - val_families)
    target_entries = round(len(rows) * val_fraction)
    if graph is None:
        total_groups = len(all_families)
        val_group_count = len(val_families)
        train_group_count = total_groups - val_group_count
    else:
        split_by_group = [group_split(group, val_families) for group in graph.groups]
        total_groups = len(graph.groups)
        val_group_count = sum(split == "val" for split in split_by_group)
        train_group_count = sum(split == "train" for split in split_by_group)
    actual_fraction = val_entries / len(rows) if rows else 0.0
    return SplitSummary(
        total_entries=len(rows),
        train_entries=train_entries,
        val_entries=val_entries,
        total_families=len(all_families),
        train_families=train_families,
        val_families=len(val_families),
        total_groups=total_groups,
        train_groups=train_group_count,
        val_groups=val_group_count,
        target_val_entries=target_entries,
        target_val_fraction=val_fraction,
        actual_val_fraction=actual_fraction,
        seed=seed,
    )


def molecule_split_settings(config: MoleculeSplitConfig) -> dict[str, Any]:
    """Return JSON-serializable molecule split settings."""
    return {
        "molecule_key_source": config.molecule_key_source,
        "min_component_heavy_atoms": config.min_component_heavy_atoms,
        "ignore_solvents": config.ignore_solvents,
    }


def manifest_payload(
    rows: Sequence[MaterialRow],
    val_families: set[str],
    summary: SplitSummary,
    input_lmdb: Path,
    config: MoleculeSplitConfig,
) -> dict[str, Any]:
    """Build the reusable validation split manifest payload."""
    val_family_set = set(val_families)
    return {
        "version": MANIFEST_VERSION,
        "created_at": _now_iso(),
        "input_lmdb": str(input_lmdb),
        "molecule_isolation": molecule_split_settings(config),
        "target_val_fraction": summary.target_val_fraction,
        "target_val_entries": summary.target_val_entries,
        "actual_val_fraction": summary.actual_val_fraction,
        "seed": summary.seed,
        "counts": {
            "all": summary.total_entries,
            "train": summary.train_entries,
            "val": summary.val_entries,
            "all_families": summary.total_families,
            "train_families": summary.train_families,
            "val_families": summary.val_families,
            "all_groups": summary.total_groups,
            "train_groups": summary.train_groups,
            "val_groups": summary.val_groups,
        },
        "val_families": sorted(val_family_set),
        "rows": [
            {
                "all_index": row.index,
                "refcode": row.refcode,
                "family": row.family,
                "split": "val" if row.family in val_family_set else "train",
            }
            for row in rows
        ],
    }


def load_manifest_families(
    path: Path,
    expected_settings: Mapping[str, Any] | None = None,
) -> set[str]:
    """Load validation CSD families from an existing manifest."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    version = payload.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"Manifest version {version!r} is not supported; "
            "rerun with --regenerate-manifest."
        )
    if expected_settings is not None:
        settings = payload.get("molecule_isolation")
        if settings != dict(expected_settings):
            raise ValueError(
                f"Manifest molecule settings do not match current settings: {path}"
            )
    families = payload.get("val_families")
    if not isinstance(families, list) or not families:
        raise ValueError(f"Manifest lacks a non-empty val_families list: {path}")
    return {str(family).strip().upper() for family in families}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with parent directories created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def graph_payload(
    graph: MoleculeFamilyGraph,
    val_families: set[str],
    config: MoleculeSplitConfig,
    skipped_component_counts: Counter[str],
) -> dict[str, Any]:
    """Build an audit payload for the molecule-family graph."""
    linked_molecules = {
        molecule_key: list(families)
        for molecule_key, families in sorted(graph.molecule_families.items())
        if len(families) > 1
    }
    return {
        "version": MANIFEST_VERSION,
        "created_at": _now_iso(),
        "molecule_isolation": molecule_split_settings(config),
        "counts": {
            "groups": len(graph.groups),
            "eligible_molecule_keys": len(graph.molecule_families),
            "linked_molecule_keys": len(linked_molecules),
        },
        "skipped_component_counts": dict(sorted(skipped_component_counts.items())),
        "groups": [
            {
                "group_id": group.group_id,
                "families": list(group.families),
                "entry_count": group.entry_count,
                "eligible_molecule_count": len(group.molecule_keys),
                "split": group_split(group, val_families),
            }
            for group in graph.groups
        ],
        "molecules": linked_molecules,
    }


def verification_payload(
    rows: Sequence[MaterialRow],
    val_families: set[str],
    val_fraction: float,
    skipped_component_counts: Counter[str],
) -> dict[str, Any]:
    """Build verification checks for a planned train/validation split."""
    train_families: set[str] = set()
    validation_families: set[str] = set()
    train_molecules: set[str] = set()
    validation_molecules: set[str] = set()
    train_count = 0
    val_count = 0

    for row in rows:
        if row.family in val_families:
            validation_families.add(row.family)
            validation_molecules.update(row.molecule_keys)
            val_count += 1
        else:
            train_families.add(row.family)
            train_molecules.update(row.molecule_keys)
            train_count += 1

    family_overlap = sorted(train_families.intersection(validation_families))
    molecule_overlap = sorted(train_molecules.intersection(validation_molecules))
    target_val_entries = round(len(rows) * val_fraction)
    checks = {
        "partition_counts": train_count + val_count == len(rows),
        "val_count_exceeds_target": val_count >= target_val_entries,
        "family_overlap_zero": len(family_overlap) == 0,
        "molecule_overlap_zero": len(molecule_overlap) == 0,
    }
    return {
        "version": MANIFEST_VERSION,
        "created_at": _now_iso(),
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "all": len(rows),
            "train": train_count,
            "val": val_count,
            "target_val_entries": target_val_entries,
            "train_families": len(train_families),
            "val_families": len(validation_families),
            "train_molecule_keys": len(train_molecules),
            "val_molecule_keys": len(validation_molecules),
            "family_overlap": len(family_overlap),
            "molecule_overlap": len(molecule_overlap),
        },
        "examples": {
            "family_overlap": family_overlap[:20],
            "molecule_overlap": molecule_overlap[:20],
        },
        "skipped_component_counts": dict(sorted(skipped_component_counts.items())),
    }


def require_verification_passed(payload: Mapping[str, Any]) -> None:
    """Raise when split verification did not pass."""
    if bool(payload["passed"]):
        return
    raise RuntimeError(
        "Split verification failed: "
        f"{json.dumps(payload.get('checks', {}), sort_keys=True)}"
    )


def _remove_existing_outputs(output_dir: Path, overwrite: bool) -> None:
    """Remove existing split outputs when overwrite is explicitly enabled."""
    paths = [
        output_dir / "train.lmdb",
        output_dir / "val.lmdb",
        output_dir / "train.num_atoms.npy",
        output_dir / "val.num_atoms.npy",
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Split outputs already exist: {formatted}")
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _put_material(
    txn: lmdb.Transaction,
    material: Material,
    write_count: int,
) -> None:
    """Write one material to an output LMDB transaction."""
    txn.put(f"{write_count:08d}".encode(), pickle.dumps(material))


def write_split_lmdbs(
    input_lmdb: Path,
    output_dir: Path,
    val_families: set[str],
    overwrite: bool,
    log_interval: int = 100000,
) -> tuple[int, int]:
    """Write train and validation LMDBs plus atom-count caches."""
    _remove_existing_outputs(output_dir, overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_env = lmdb.open(
        str(input_lmdb),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
    )
    train_env = lmdb.open(str(output_dir / "train.lmdb"), map_size=LMDB_MAP_SIZE)
    val_env = lmdb.open(str(output_dir / "val.lmdb"), map_size=LMDB_MAP_SIZE)
    train_txn = train_env.begin(write=True)
    val_txn = val_env.begin(write=True)
    train_num_atoms: list[int] = []
    val_num_atoms: list[int] = []
    train_count = 0
    val_count = 0
    start_time = time.perf_counter()

    try:
        with input_env.begin() as input_txn:
            total = int(input_txn.get(b"__len__").decode())
            for index in range(total):
                material: Material = pickle.loads(
                    input_txn.get(f"{index:08d}".encode())
                )
                family = refcode_family_from_material(material, index)
                num_atoms = int(material.conditioning.atomic_numbers.shape[0])

                if family in val_families:
                    _put_material(val_txn, material, val_count)
                    val_num_atoms.append(num_atoms)
                    val_count += 1
                    if val_count % COMMIT_INTERVAL == 0:
                        val_txn.commit()
                        val_txn = val_env.begin(write=True)
                else:
                    _put_material(train_txn, material, train_count)
                    train_num_atoms.append(num_atoms)
                    train_count += 1
                    if train_count % COMMIT_INTERVAL == 0:
                        train_txn.commit()
                        train_txn = train_env.begin(write=True)

                if log_interval > 0 and (index + 1) % log_interval == 0:
                    elapsed = time.perf_counter() - start_time
                    rate = (index + 1) / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "Wrote %d/%d source rows: train=%d val=%d (%.1f rows/s)",
                        index + 1,
                        total,
                        train_count,
                        val_count,
                        rate,
                    )

        train_txn.put(b"__len__", str(train_count).encode())
        val_txn.put(b"__len__", str(val_count).encode())
        train_txn.commit()
        val_txn.commit()
    finally:
        input_env.close()
        train_env.close()
        val_env.close()

    np.save(
        output_dir / "train.num_atoms.npy", np.asarray(train_num_atoms, dtype=np.int64)
    )
    np.save(output_dir / "val.num_atoms.npy", np.asarray(val_num_atoms, dtype=np.int64))
    return train_count, val_count


def resolve_val_families(
    rows: Sequence[MaterialRow],
    graph: MoleculeFamilyGraph,
    manifest_path: Path,
    val_fraction: float,
    seed: int,
    regenerate_manifest: bool,
    config: MoleculeSplitConfig,
) -> tuple[set[str], str]:
    """Return validation families and the source used to choose them."""
    if manifest_path.is_file() and not regenerate_manifest:
        return (
            load_manifest_families(
                manifest_path,
                expected_settings=molecule_split_settings(config),
            ),
            "manifest",
        )
    selected_group_ids = select_validation_group_ids(
        graph.groups,
        len(rows),
        val_fraction,
        seed,
    )
    return val_families_from_groups(graph.groups, selected_group_ids), "generated"


def build_summary_payload(
    summary: SplitSummary,
    input_lmdb: Path,
    output_dir: Path,
    manifest_path: Path,
    manifest_source: str,
    config: MoleculeSplitConfig,
    dry_run: bool,
    elapsed_seconds: Mapping[str, float | None] | None = None,
) -> dict[str, Any]:
    """Build a compact split summary payload."""
    payload = {
        "created_at": _now_iso(),
        "dry_run": dry_run,
        "input_lmdb": str(input_lmdb),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "manifest_source": manifest_source,
        "molecule_isolation": molecule_split_settings(config),
        "counts": {
            "all": summary.total_entries,
            "train": summary.train_entries,
            "val": summary.val_entries,
            "all_families": summary.total_families,
            "train_families": summary.train_families,
            "val_families": summary.val_families,
            "all_groups": summary.total_groups,
            "train_groups": summary.train_groups,
            "val_groups": summary.val_groups,
        },
        "target_val_entries": summary.target_val_entries,
        "target_val_fraction": summary.target_val_fraction,
        "actual_val_fraction": summary.actual_val_fraction,
        "seed": summary.seed,
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = dict(elapsed_seconds)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-lmdb", type=Path, default=DEFAULT_INPUT_LMDB)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--min-component-heavy-atoms",
        type=int,
        default=DEFAULT_MIN_COMPONENT_HEAVY_ATOMS,
    )
    parser.add_argument(
        "--include-solvents",
        action="store_true",
        help="Include solvent components when building molecule-overlap groups.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--regenerate-manifest", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CSD molecule-isolated LMDB splitter."""
    run_start_time = time.perf_counter()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    input_lmdb = args.input_lmdb.expanduser().resolve()
    output_dir = (
        input_lmdb.parent
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    manifest_path = (
        _default_manifest_path(output_dir)
        if args.manifest is None
        else args.manifest.expanduser().resolve()
    )
    split_config = MoleculeSplitConfig(
        min_component_heavy_atoms=int(args.min_component_heavy_atoms),
        ignore_solvents=not bool(args.include_solvents),
    )

    scan_start_time = time.perf_counter()
    scan = read_lmdb_rows(
        input_lmdb,
        split_config,
        log_interval=int(args.log_interval),
    )
    scan_elapsed = time.perf_counter() - scan_start_time
    rows = scan.rows

    graph_start_time = time.perf_counter()
    graph = build_molecule_family_graph(rows)
    graph_elapsed = time.perf_counter() - graph_start_time

    plan_start_time = time.perf_counter()
    val_families, manifest_source = resolve_val_families(
        rows,
        graph,
        manifest_path,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        regenerate_manifest=bool(args.regenerate_manifest),
        config=split_config,
    )
    summary = split_summary(
        rows,
        val_families,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        graph=graph,
    )
    planned_verification = verification_payload(
        rows,
        val_families,
        float(args.val_fraction),
        scan.skipped_component_counts,
    )
    require_verification_passed(planned_verification)
    plan_elapsed = time.perf_counter() - plan_start_time
    summary_payload = build_summary_payload(
        summary,
        input_lmdb,
        output_dir,
        manifest_path,
        manifest_source,
        split_config,
        dry_run=bool(args.dry_run),
        elapsed_seconds={
            "scan": scan_elapsed,
            "graph": graph_elapsed,
            "plan": plan_elapsed,
            "artifacts": None,
            "write": None,
            "total": time.perf_counter() - run_start_time,
        },
    )
    logger.info("Split summary:\n%s", json.dumps(summary_payload, indent=2))
    logger.info(
        "Split verification:\n%s",
        json.dumps(
            {
                "passed": planned_verification["passed"],
                "checks": planned_verification["checks"],
                "counts": planned_verification["counts"],
            },
            indent=2,
        ),
    )

    if args.dry_run:
        return 0

    artifact_start_time = time.perf_counter()
    if manifest_source == "generated" or args.regenerate_manifest:
        write_json(
            manifest_path,
            manifest_payload(
                rows,
                val_families,
                summary,
                input_lmdb,
                split_config,
            ),
        )
    write_json(
        _default_graph_path(output_dir),
        graph_payload(graph, val_families, split_config, scan.skipped_component_counts),
    )
    write_json(_default_verification_path(output_dir), planned_verification)
    artifact_elapsed = time.perf_counter() - artifact_start_time

    write_start_time = time.perf_counter()
    train_count, val_count = write_split_lmdbs(
        input_lmdb,
        output_dir,
        val_families,
        overwrite=bool(args.overwrite),
        log_interval=int(args.log_interval),
    )
    write_elapsed = time.perf_counter() - write_start_time
    if train_count != summary.train_entries or val_count != summary.val_entries:
        raise RuntimeError(
            "Written split counts do not match planned counts: "
            f"planned train={summary.train_entries} val={summary.val_entries}, "
            f"wrote train={train_count} val={val_count}."
        )
    total_elapsed = time.perf_counter() - run_start_time
    final_summary_payload = build_summary_payload(
        summary,
        input_lmdb,
        output_dir,
        manifest_path,
        manifest_source,
        split_config,
        dry_run=False,
        elapsed_seconds={
            "scan": scan_elapsed,
            "graph": graph_elapsed,
            "plan": plan_elapsed,
            "artifacts": artifact_elapsed,
            "write": write_elapsed,
            "total": total_elapsed,
        },
    )
    write_json(_default_summary_path(output_dir), final_summary_payload)
    logger.info(
        "Wrote train=%d and val=%d entries to %s", train_count, val_count, output_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
