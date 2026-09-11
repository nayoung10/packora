"""Tests for public CSD manifest reconstruction helpers."""

from __future__ import annotations

import csv
import pickle
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import lmdb
import numpy as np
import pytest

from src.data.preprocess.base import PreprocessContext
from src.data.preprocess.csd_manifest import (
    NO_TEMPLATE_SOURCE,
    RECOVERY_SOURCE,
    STANDARD_SOURCE,
    CSDManifestPreprocessor,
    SplitCachePaths,
    assemble_manifest_lmdbs,
    load_csd_manifest,
    verify_manifest_lmdbs,
)
from src.data.types import Material, MoleculeConditioning


def _write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write a small public CSD manifest fixture."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "split"])
        writer.writerows(rows)


def _material(refcode: str, template_present: bool = True) -> Material:
    """Build a minimal cache-compatible material fixture."""
    conditioning = MoleculeConditioning(
        atomic_numbers=np.array([6, 1], dtype=np.int32),
        bond_indices=np.array([[0, 1], [1, 0]], dtype=np.int32),
        bond_types=np.array([1, 1], dtype=np.int32),
        formal_charges=np.zeros(2, dtype=np.int32),
        template_coords=np.zeros((2, 3), dtype=np.float64),
        template_present=template_present,
        membership=np.zeros(2, dtype=np.int32),
        spacegroup_number=1,
        spacegroup_present=True,
        atom_chirality=np.zeros(2, dtype=np.int32),
        bond_stereochemistry=np.zeros(2, dtype=np.int32),
        stereochemistry_present=True,
        num_molecules=1,
    )
    return Material(
        cart_coords=np.zeros((2, 3), dtype=np.float64),
        frac_coords=np.zeros((2, 3), dtype=np.float64),
        lattice_parameters=np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0]),
        cell=np.eye(3, dtype=np.float64) * 5.0,
        conditioning=conditioning,
        info={"csd_refcode": refcode, "dataset_name": "source"},
    )


def _write_cache(
    path: Path,
    rows: list[tuple[str, str, Material | None, str | None]],
) -> None:
    """Write intermediate-cache rows needed by assembly tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE entries (
            key TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            material_blob BLOB,
            refcode TEXT,
            failure_reason TEXT,
            failure_detail TEXT,
            stage TEXT,
            skipped INTEGER NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO entries VALUES (?, ?, ?, ?, ?, NULL, 'test', 0)",
        [
            (
                refcode,
                status,
                None if material is None else pickle.dumps(material),
                refcode,
                reason,
            )
            for refcode, status, material, reason in rows
        ],
    )
    conn.commit()
    conn.close()


def _read_materials(path: Path) -> list[Material]:
    """Read every material from one test LMDB."""
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            total = int(txn.get(b"__len__").decode())
            return [
                pickle.loads(txn.get(f"{index:08d}".encode())) for index in range(total)
            ]
    finally:
        env.close()


def test_manifest_accepts_arbitrary_safe_splits(tmp_path: Path) -> None:
    """Discover arbitrary split names and normalize refcodes."""
    path = tmp_path / "splits.csv"
    _write_manifest(path, [("bbbbbb01", "banana"), ("AAAAAA", "apple")])

    manifest = load_csd_manifest(path)

    assert manifest.splits == ("apple", "banana")
    assert manifest.refcodes("banana") == ("BBBBBB01",)


def test_manifest_loads_optional_truth_refcodes(tmp_path: Path) -> None:
    """Normalize optional benchmark truth groups while preserving their order."""
    path = tmp_path / "benchmark.csv"
    path.write_text(
        "id,split,truth_refcodes\nABCDEF,rigid,ABCDEF;ABCDEF02;ABCDEF01;ABCDEF02\n",
        encoding="utf-8",
    )

    manifest = load_csd_manifest(path)

    assert manifest.rows[0].truth_refcodes == (
        "ABCDEF",
        "ABCDEF02",
        "ABCDEF01",
    )


def test_manifest_loads_optional_flexibility_label(tmp_path: Path) -> None:
    """Normalize an authoritative optional benchmark flexibility label."""
    path = tmp_path / "benchmark.csv"
    path.write_text(
        "id,split,truth_refcodes,flexibility\nABCDEF,rigid,ABCDEF,RIGID\n",
        encoding="utf-8",
    )

    manifest = load_csd_manifest(path)

    assert manifest.rows[0].flexibility == "rigid"


def test_manifest_rejects_invalid_flexibility_label(tmp_path: Path) -> None:
    """Reject flexibility labels unsupported by benchmark evaluation."""
    path = tmp_path / "benchmark.csv"
    path.write_text(
        "id,split,flexibility\nABCDEF,rigid,unknown\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid flexibility"):
        load_csd_manifest(path)


def test_manifest_rejects_truth_group_with_different_primary_id(
    tmp_path: Path,
) -> None:
    """Require the converted target to be first in its benchmark truth group."""
    path = tmp_path / "benchmark.csv"
    path.write_text(
        "id,split,truth_refcodes\nABCDEF,rigid,ABCDEF01;ABCDEF\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="truth_refcodes must start with id"):
        load_csd_manifest(path)


@pytest.mark.parametrize("split", ["", "../train", "a/b", "a\\b", ".", ".."])
def test_manifest_rejects_unsafe_split_names(tmp_path: Path, split: str) -> None:
    """Reject split values that could escape the output directory."""
    path = tmp_path / "splits.csv"
    _write_manifest(path, [("AAAAAA", split)])

    with pytest.raises(ValueError, match="unsafe split name"):
        load_csd_manifest(path)


def test_manifest_rejects_duplicate_refcodes(tmp_path: Path) -> None:
    """Reject repeated refcodes even when assigned to different splits."""
    path = tmp_path / "splits.csv"
    _write_manifest(path, [("AAAAAA", "apple"), ("aaaaaa", "banana")])

    with pytest.raises(ValueError, match="repeats refcode"):
        load_csd_manifest(path)


def test_manifest_preprocessor_selects_requested_split(tmp_path: Path) -> None:
    """Select exact refcodes in manifest order for any safe split name."""
    path = tmp_path / "splits.csv"
    _write_manifest(
        path,
        [("BBBBBB", "banana"), ("AAAAAA", "banana"), ("CCCCCC", "apple")],
    )
    preprocessor = CSDManifestPreprocessor(
        manifest=load_csd_manifest(path),
        add_missing_hydrogens=True,
        preprocessing_source=STANDARD_SOURCE,
    )
    context = PreprocessContext(
        data_dir=tmp_path,
        dataset_name="fruit",
        split="banana",
        source_split="banana",
    )

    refs = preprocessor.iter_raw_entries(context)

    assert [ref.key for ref in refs] == ["BBBBBB", "AAAAAA"]


def test_manifest_preprocessor_can_bypass_timed_out_template(
    tmp_path: Path,
) -> None:
    """Keep an entry with unavailable template conditioning after a timeout."""
    path = tmp_path / "splits.csv"
    _write_manifest(path, [("AAAAAA", "train")])
    preprocessor = CSDManifestPreprocessor(
        manifest=load_csd_manifest(path),
        add_missing_hydrogens=False,
        preprocessing_source=NO_TEMPLATE_SOURCE,
        generate_rdkit_template=False,
    )

    result = preprocessor._rdkit_template_result(
        SimpleNamespace(atoms=[object(), object()])
    )

    assert result.coords.shape == (2, 3)
    assert not result.present
    assert result.failure_reason == "rdkit_template_timed_out"


def test_assembly_publishes_partial_arbitrary_splits(tmp_path: Path) -> None:
    """Write successes from all passes while auditing unresolved rows."""
    manifest_path = tmp_path / "splits.csv"
    _write_manifest(
        manifest_path,
        [
            ("BBBBBB", "banana"),
            ("AAAAAA", "banana"),
            ("DDDDDD", "apple"),
            ("CCCCCC", "apple"),
        ],
    )
    manifest = load_csd_manifest(manifest_path)
    cache_paths: dict[str, SplitCachePaths] = {}
    for split in manifest.splits:
        standard_path = tmp_path / "cache" / STANDARD_SOURCE / f"{split}.sqlite3"
        recovery_path = tmp_path / "cache" / RECOVERY_SOURCE / f"{split}.sqlite3"
        no_template_path = tmp_path / "cache" / NO_TEMPLATE_SOURCE / f"{split}.sqlite3"
        cache_paths[split] = SplitCachePaths(
            standard_path,
            recovery_path,
            no_template_path,
        )

    _write_cache(
        cache_paths["banana"].standard,
        [
            ("AAAAAA", "success", _material("AAAAAA", False), None),
            ("BBBBBB", "failed", None, "csd_add_hydrogens_failed"),
        ],
    )
    _write_cache(
        cache_paths["banana"].recovery,
        [("BBBBBB", "success", _material("BBBBBB", True), None)],
    )
    _write_cache(
        cache_paths["apple"].standard,
        [
            ("CCCCCC", "failed", None, "csd_conversion_failed"),
            ("DDDDDD", "timed_out", None, "timed_out"),
        ],
    )
    _write_cache(
        cache_paths["apple"].recovery,
        [
            ("CCCCCC", "failed", None, "csd_conversion_failed"),
            ("DDDDDD", "timed_out", None, "timed_out"),
        ],
    )
    _write_cache(
        cache_paths["apple"].no_template_recovery,
        [("DDDDDD", "success", _material("DDDDDD", False), None)],
    )
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    summaries, failures = assemble_manifest_lmdbs(
        manifest,
        cache_paths,
        staging_dir,
        dataset_name="fruit",
    )
    verification = verify_manifest_lmdbs(manifest, staging_dir, failures)

    assert summaries == {
        "apple": {
            "expected": 2,
            "standard_success": 0,
            "recovery_success": 0,
            "no_template_recovery_success": 1,
            "failed": 1,
            "written": 1,
        },
        "banana": {
            "expected": 2,
            "standard_success": 1,
            "recovery_success": 1,
            "no_template_recovery_success": 0,
            "failed": 0,
            "written": 2,
        },
    }
    assert [(failure.refcode, failure.split) for failure in failures] == [
        ("CCCCCC", "apple"),
    ]
    assert verification["passed"]
    assert not verification["complete"]
    apple = _read_materials(staging_dir / "apple.lmdb")
    assert [material.info["csd_refcode"] for material in apple] == ["DDDDDD"]
    assert apple[0].info["preprocessing_source"] == NO_TEMPLATE_SOURCE
    assert not apple[0].conditioning.template_present

    banana = _read_materials(staging_dir / "banana.lmdb")
    assert [material.info["csd_refcode"] for material in banana] == [
        "BBBBBB",
        "AAAAAA",
    ]
    assert banana[0].conditioning.template_present
    assert banana[0].info["preprocessing_source"] == RECOVERY_SOURCE
    assert banana[0].info["hydrogen_policy"] == "explicit_only"
    assert banana[1].info["template_policy"] == "unavailable"
    assert banana[1].info["preprocessing_source"] == STANDARD_SOURCE


def test_assembly_persists_optional_benchmark_truth_group(tmp_path: Path) -> None:
    """Attach optional benchmark truth metadata to the assembled material."""
    manifest_path = tmp_path / "rigid.csv"
    manifest_path.write_text(
        "id,split,truth_refcodes\nABCDEF,rigid,ABCDEF;ABCDEF01\n",
        encoding="utf-8",
    )
    manifest = load_csd_manifest(manifest_path)
    standard_path = tmp_path / "cache" / STANDARD_SOURCE / "rigid.sqlite3"
    recovery_path = tmp_path / "cache" / RECOVERY_SOURCE / "rigid.sqlite3"
    no_template_path = tmp_path / "cache" / NO_TEMPLATE_SOURCE / "rigid.sqlite3"
    _write_cache(
        standard_path,
        [("ABCDEF", "success", _material("ABCDEF"), None)],
    )
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    _, failures = assemble_manifest_lmdbs(
        manifest,
        {
            "rigid": SplitCachePaths(
                standard_path,
                recovery_path,
                no_template_path,
            )
        },
        staging_dir,
        dataset_name="benchmarks",
    )

    assert not failures
    material = _read_materials(staging_dir / "rigid.lmdb")[0]
    assert material.info["benchmark_truth_refcodes"] == ["ABCDEF", "ABCDEF01"]
    assert material.info["benchmark_truth_group"] == "ABCDEF;ABCDEF01"


def test_assembly_overrides_computed_flexibility_from_manifest(tmp_path: Path) -> None:
    """Use the authoritative manifest label for benchmark evaluation metadata."""
    manifest_path = tmp_path / "rigid.csv"
    manifest_path.write_text(
        "id,split,truth_refcodes,flexibility\nABCDEF,rigid,ABCDEF,rigid\n",
        encoding="utf-8",
    )
    manifest = load_csd_manifest(manifest_path)
    material = _material("ABCDEF")
    material.info["flexibility"] = "flexible"
    standard_path = tmp_path / "cache" / STANDARD_SOURCE / "rigid.sqlite3"
    _write_cache(standard_path, [("ABCDEF", "success", material, None)])
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    _, failures = assemble_manifest_lmdbs(
        manifest,
        {
            "rigid": SplitCachePaths(
                standard_path,
                tmp_path / "recovery.sqlite3",
                tmp_path / "no_template.sqlite3",
            )
        },
        staging_dir,
        dataset_name="benchmarks",
    )

    assert not failures
    material = _read_materials(staging_dir / "rigid.lmdb")[0]
    assert material.info["flexibility"] == "rigid"
