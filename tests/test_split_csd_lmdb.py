import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts.split_csd_lmdb import (
    MANIFEST_VERSION,
    MaterialRow,
    MoleculeSplitConfig,
    build_molecule_family_graph,
    build_summary_payload,
    graph_payload,
    load_manifest_families,
    manifest_payload,
    molecule_keys_from_material,
    molecule_split_settings,
    select_validation_families,
    select_validation_group_ids,
    split_summary,
    val_families_from_groups,
    verification_payload,
    write_json,
)
from src.data.types import Material, MoleculeConditioning


def _rows() -> list[MaterialRow]:
    """Return a small molecule-linked split fixture."""
    return [
        MaterialRow(
            index=0,
            refcode="AAAAAA01",
            family="AAAAAA",
            num_atoms=10,
            molecule_keys=("X",),
        ),
        MaterialRow(
            index=1,
            refcode="AAAAAA02",
            family="AAAAAA",
            num_atoms=12,
            molecule_keys=("X",),
        ),
        MaterialRow(
            index=2,
            refcode="BBBBBB01",
            family="BBBBBB",
            num_atoms=14,
            molecule_keys=("Y",),
        ),
        MaterialRow(
            index=3,
            refcode="CCCCCC01",
            family="CCCCCC",
            num_atoms=16,
            molecule_keys=("X",),
        ),
        MaterialRow(
            index=4,
            refcode="DDDDDD01",
            family="DDDDDD",
            num_atoms=18,
            molecule_keys=("Z",),
        ),
    ]


def _material(
    refcode: str,
    component_smiles: list[str | None],
    component_atomic_numbers: list[list[int]],
) -> Material:
    """Build a minimal material with component SMILES metadata."""
    atomic_numbers = np.array(
        [number for component in component_atomic_numbers for number in component],
        dtype=np.int32,
    )
    membership = np.array(
        [
            component_index
            for component_index, component in enumerate(component_atomic_numbers)
            for _ in component
        ],
        dtype=np.int32,
    )
    num_atoms = int(atomic_numbers.shape[0])
    conditioning = MoleculeConditioning(
        atomic_numbers=atomic_numbers,
        bond_indices=np.zeros((2, 0), dtype=np.int32),
        bond_types=np.zeros(0, dtype=np.int32),
        formal_charges=np.zeros(num_atoms, dtype=np.int32),
        template_coords=np.zeros((num_atoms, 3), dtype=np.float64),
        template_present=False,
        membership=membership,
        spacegroup_number=1,
        spacegroup_present=True,
        atom_chirality=np.zeros(num_atoms, dtype=np.int32),
        bond_stereochemistry=np.zeros(0, dtype=np.int32),
        stereochemistry_present=True,
        num_molecules=len(component_smiles),
    )
    return Material(
        cart_coords=np.zeros((num_atoms, 3), dtype=np.float64),
        frac_coords=np.zeros((num_atoms, 3), dtype=np.float64),
        lattice_parameters=np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0]),
        cell=np.eye(3, dtype=np.float64) * 5.0,
        conditioning=conditioning,
        info={
            "dataset_name": "csd",
            "csd_refcode": refcode,
            "csd_component_smiles": component_smiles,
        },
    )


def test_select_validation_families_is_seed_deterministic() -> None:
    """Validation family selection is deterministic for one seed."""
    counts = Counter({"AAAAAA": 2, "BBBBBB": 1, "CCCCCC": 2, "DDDDDD": 1})

    first = select_validation_families(counts, val_fraction=0.33, seed=20250501)
    second = select_validation_families(counts, val_fraction=0.33, seed=20250501)

    assert first == second
    assert first


def test_build_molecule_family_graph_merges_shared_molecules() -> None:
    """Shared molecule keys merge different CSD families into one group."""
    graph = build_molecule_family_graph(_rows())

    linked_group = next(group for group in graph.groups if "AAAAAA" in group.families)

    assert linked_group.families == ("AAAAAA", "CCCCCC")
    assert linked_group.entry_count == 3
    assert graph.molecule_families["X"] == ("AAAAAA", "CCCCCC")


def test_select_validation_groups_exceeds_target_without_molecule_overlap() -> None:
    """Selected connected groups reach target size without molecule overlap."""
    rows = _rows()
    graph = build_molecule_family_graph(rows)
    selected_group_ids = select_validation_group_ids(
        graph.groups,
        total_entries=len(rows),
        val_fraction=0.4,
        seed=20250501,
    )
    val_families = val_families_from_groups(graph.groups, selected_group_ids)

    payload = verification_payload(rows, val_families, 0.4, Counter())

    assert payload["passed"]
    assert payload["checks"]["val_count_exceeds_target"]
    assert payload["counts"]["molecule_overlap"] == 0


def test_split_summary_counts_connected_groups() -> None:
    """Split summary reports molecule-connected group counts."""
    rows = _rows()
    graph = build_molecule_family_graph(rows)
    val_families = {"AAAAAA", "CCCCCC"}
    summary = split_summary(
        rows,
        val_families=val_families,
        val_fraction=0.4,
        seed=20250501,
        graph=graph,
    )

    assert summary.total_entries == 5
    assert summary.train_entries == 2
    assert summary.val_entries == 3
    assert summary.total_groups == 3
    assert summary.train_groups == 2
    assert summary.val_groups == 1


def test_summary_payload_records_elapsed_seconds(tmp_path: Path) -> None:
    """Split summary payload includes elapsed timing fields."""
    rows = _rows()
    graph = build_molecule_family_graph(rows)
    summary = split_summary(
        rows,
        val_families={"AAAAAA", "CCCCCC"},
        val_fraction=0.4,
        seed=20250501,
        graph=graph,
    )

    payload = build_summary_payload(
        summary,
        tmp_path / "all.lmdb",
        tmp_path,
        tmp_path / "splits" / "validation_manifest.json",
        "generated",
        MoleculeSplitConfig(),
        dry_run=False,
        elapsed_seconds={"scan": 1.0, "graph": 2.0, "total": 3.0},
    )

    assert payload["elapsed_seconds"] == {"scan": 1.0, "graph": 2.0, "total": 3.0}


def test_molecule_keys_skip_missing_small_and_solvent_components() -> None:
    """Molecule keys use preprocess component eligibility rules."""
    material = _material(
        "ABCDEF01",
        ["O", "CCCCCCCC", "[Cl-]", None],
        [[8], [6, 6, 6, 6, 6, 6, 6, 6], [17], [6, 6, 6, 6, 6, 6, 6, 6]],
    )

    keys, skipped = molecule_keys_from_material(
        material,
        0,
        {"O"},
        MoleculeSplitConfig(min_component_heavy_atoms=8, ignore_solvents=True),
    )

    assert keys == ("CCCCCCCC",)
    assert skipped == Counter(
        {
            "solvent": 1,
            "min_component_heavy_atoms": 1,
            "missing_smiles": 1,
        }
    )


def test_manifest_round_trip_stores_validation_families(tmp_path: Path) -> None:
    """Manifest loading reuses validation families as the source of truth."""
    rows = _rows()
    graph = build_molecule_family_graph(rows)
    config = MoleculeSplitConfig()
    summary = split_summary(
        rows,
        val_families={"AAAAAA", "CCCCCC"},
        val_fraction=0.4,
        seed=20250501,
        graph=graph,
    )
    payload = manifest_payload(
        rows,
        val_families={"AAAAAA", "CCCCCC"},
        summary=summary,
        input_lmdb=tmp_path / "all.lmdb",
        config=config,
    )
    path = tmp_path / "validation_manifest.json"

    write_json(path, payload)

    assert load_manifest_families(
        path,
        expected_settings=molecule_split_settings(config),
    ) == {"AAAAAA", "CCCCCC"}
    rows_payload = json.loads(path.read_text())["rows"]
    assert payload["version"] == MANIFEST_VERSION
    assert {row["split"] for row in rows_payload} == {"train", "val"}


def test_legacy_manifest_is_rejected(tmp_path: Path) -> None:
    """Legacy family-only manifests are not reused."""
    path = tmp_path / "validation_manifest.json"
    write_json(path, {"version": 1, "val_families": ["AAAAAA"]})

    with pytest.raises(ValueError, match="regenerate-manifest"):
        load_manifest_families(path)


def test_graph_payload_contains_audit_data() -> None:
    """Graph payload records linked molecules and skipped counts."""
    rows = _rows()
    graph = build_molecule_family_graph(rows)
    payload = graph_payload(
        graph,
        {"AAAAAA", "CCCCCC"},
        MoleculeSplitConfig(),
        Counter({"missing_smiles": 2}),
    )

    assert payload["molecules"] == {"X": ["AAAAAA", "CCCCCC"]}
    assert payload["skipped_component_counts"] == {"missing_smiles": 2}
    assert {group["split"] for group in payload["groups"]} == {"train", "val"}
