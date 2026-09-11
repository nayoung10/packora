"""Build model-ready CSD LMDB splits from a public refcode manifest."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
import pickle
import shutil
import sqlite3
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lmdb
import numpy as np

from src.data.preprocess.base import PreprocessContext, RawEntryRef
from src.data.preprocess.csd import (
    CSDFilterConfig,
    CSDPreprocessor,
    CSDPreprocessingError,
    RDKitTemplateConfig,
    RDKitTemplateResult,
)
from src.data.preprocess.utils.cache import cache_source_split
from src.data.preprocess.utils.config import PreprocessCacheConfig
from src.data.preprocess.utils.lmdb import COMMIT_INTERVAL, LMDB_MAP_SIZE
from src.data.types import Material

if TYPE_CHECKING:
    from ccdc.crystal import Crystal
    from ccdc.entry import Entry
    from ccdc.molecule import Molecule

logger = logging.getLogger(__name__)

STANDARD_SOURCE = "standard"
RECOVERY_SOURCE = "explicit_hydrogen_recovery"
NO_TEMPLATE_SOURCE = "explicit_hydrogen_no_template_recovery"


@dataclass(frozen=True)
class CSDManifestRow:
    """One normalized CSD refcode and output-split assignment."""

    refcode: str
    split: str
    truth_refcodes: tuple[str, ...] = ()
    flexibility: str | None = None


@dataclass(frozen=True)
class CSDManifest:
    """Validated public CSD split manifest."""

    path: Path
    rows: tuple[CSDManifestRow, ...]
    sha256: str

    @property
    def splits(self) -> tuple[str, ...]:
        """Return all split names in deterministic order."""
        return tuple(sorted({row.split for row in self.rows}))

    def refcodes(self, split: str) -> tuple[str, ...]:
        """Return refcodes assigned to one split in manifest order."""
        return tuple(row.refcode for row in self.rows if row.split == split)

    def rows_for_split(self, split: str) -> tuple[CSDManifestRow, ...]:
        """Return rows assigned to one split in manifest order."""
        return tuple(row for row in self.rows if row.split == split)

    def subset(self, refcodes: set[str]) -> CSDManifest:
        """Return a manifest containing only selected refcodes."""
        return CSDManifest(
            path=self.path,
            rows=tuple(row for row in self.rows if row.refcode in refcodes),
            sha256=self.sha256,
        )


@dataclass(frozen=True)
class CacheEntryResult:
    """Final status metadata for one intermediate cache row."""

    status: str
    failure_reason: str | None
    failure_detail: str | None
    stage: str | None


@dataclass(frozen=True)
class BuildFailure:
    """One manifest row unresolved by every preprocessing pass."""

    refcode: str
    split: str
    standard: CacheEntryResult | None
    recovery: CacheEntryResult | None
    no_template_recovery: CacheEntryResult | None


@dataclass(frozen=True)
class SplitCachePaths:
    """Intermediate preprocessing cache paths for one split."""

    standard: Path
    recovery: Path
    no_template_recovery: Path


@dataclass
class _SplitWriter:
    """Open LMDB transaction and aligned metadata buffers."""

    env: lmdb.Environment
    txn: lmdb.Transaction
    count: int
    num_atoms: list[int]
    families: list[str]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_split_name(split: str, path: Path, row_number: int) -> None:
    """Reject split values that cannot safely become output filenames."""
    if (
        not split
        or split != split.strip()
        or split in {".", ".."}
        or "/" in split
        or "\\" in split
        or "\x00" in split
    ):
        raise ValueError(
            f"Manifest {path} has unsafe split name {split!r} on row {row_number}."
        )


def load_csd_manifest(path: Path) -> CSDManifest:
    """Load and validate an id/split CSD manifest."""
    resolved_path = path.expanduser().resolve()
    rows: list[CSDManifestRow] = []
    seen_refcodes: dict[str, str] = {}
    with resolved_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"id", "split"}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"Manifest {resolved_path} must contain columns "
                f"{sorted(required_columns)}."
            )
        for row_number, row in enumerate(reader, start=2):
            refcode = str(row["id"]).strip().upper()
            split = str(row["split"])
            if not refcode:
                raise ValueError(
                    f"Manifest {resolved_path} has an empty id on row {row_number}."
                )
            _validate_split_name(split, resolved_path, row_number)
            if refcode in seen_refcodes:
                raise ValueError(
                    f"Manifest {resolved_path} repeats refcode {refcode!r}; "
                    f"first split={seen_refcodes[refcode]!r}, row={row_number}."
                )
            seen_refcodes[refcode] = split
            truth_refcodes = tuple(
                dict.fromkeys(
                    value.strip().upper()
                    for value in str(row.get("truth_refcodes") or "").split(";")
                    if value.strip()
                )
            )
            if truth_refcodes and truth_refcodes[0] != refcode:
                raise ValueError(
                    f"Manifest {resolved_path} truth_refcodes must start with "
                    f"id {refcode!r} on row {row_number}."
                )
            flexibility = str(row.get("flexibility") or "").strip().lower() or None
            if flexibility not in {None, "rigid", "flexible"}:
                raise ValueError(
                    f"Manifest {resolved_path} has invalid flexibility "
                    f"{flexibility!r} on row {row_number}."
                )
            rows.append(
                CSDManifestRow(
                    refcode=refcode,
                    split=split,
                    truth_refcodes=truth_refcodes,
                    flexibility=flexibility,
                )
            )

    if not rows:
        raise ValueError(f"Manifest {resolved_path} contains no data rows.")
    return CSDManifest(
        path=resolved_path,
        rows=tuple(rows),
        sha256=sha256_file(resolved_path),
    )


class CSDManifestPreprocessor(CSDPreprocessor):
    """Convert exactly the refcodes selected by one public manifest."""

    def __init__(
        self,
        manifest: CSDManifest,
        add_missing_hydrogens: bool,
        preprocessing_source: str,
        remove_unknown_atoms: bool = True,
        generate_rdkit_template: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize manifest selection and hydrogen policy."""
        super().__init__(**kwargs)
        self.manifest = manifest
        self.add_missing_hydrogens = add_missing_hydrogens
        self.preprocessing_source = preprocessing_source
        self.remove_unknown_atoms = remove_unknown_atoms
        self.generate_rdkit_template = generate_rdkit_template
        self._removed_unknown_atom_count = 0

    def iter_raw_entries(self, context: PreprocessContext) -> list[RawEntryRef]:
        """Return exact manifest entries for the requested split."""
        if context.split is None:
            raise ValueError("Manifest-driven CSD preprocessing requires a split.")
        split = str(context.split)
        refcodes = list(self.manifest.refcodes(split))
        if not refcodes:
            raise ValueError(
                f"Manifest {self.manifest.path} has no entries for split {split!r}."
            )
        if self.max_entries is not None:
            refcodes = refcodes[: self.max_entries]
        logger.info("CSD manifest split %s: loading %d entries", split, len(refcodes))
        return [RawEntryRef(key=refcode, source_split=split) for refcode in refcodes]

    def _processed_crystal_and_molecules(
        self,
        entry: Entry,
    ) -> tuple[Crystal, Molecule]:
        """Prepare packed molecules under the configured hydrogen policy."""
        from ccdc.crystal import Crystal

        crystal: Crystal = (
            Crystal.generate_reduced_crystal(entry.crystal)
            if self.niggli
            else entry.crystal
        )
        if self.add_missing_hydrogens:
            try:
                crystal.add_hydrogens(mode="missing", add_sites=True)
            except (RuntimeError, ValueError) as exc:
                raise CSDPreprocessingError("csd_add_hydrogens_failed") from exc

        molecules: Molecule = crystal.packing(inclusion="UniqueIncluded")
        unknown_atom_count = sum(
            int(atom.atomic_number) <= 0 for atom in molecules.atoms
        )
        self._removed_unknown_atom_count = 0
        if unknown_atom_count and self.remove_unknown_atoms:
            molecules.remove_unknown_atoms()
            self._removed_unknown_atom_count = unknown_atom_count
        self._standardize_bonds(molecules)
        return crystal, molecules

    def _rdkit_template_result(self, molecules: Molecule) -> RDKitTemplateResult:
        """Generate a template or mark it unavailable after a timeout."""
        if self.generate_rdkit_template:
            return super()._rdkit_template_result(molecules)
        return RDKitTemplateResult(
            coords=np.zeros((len(molecules.atoms), 3), dtype=np.float64),
            present=False,
            failure_reason="rdkit_template_timed_out",
            failure_detail=None,
            component_results=[],
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
        """Add public reconstruction provenance to CSD metadata."""
        info = super()._material_info(
            entry,
            crystal,
            molecules,
            context,
            conditioning_summary,
            bond_is_rotatable,
            rdkit_template,
            spacegroup_number,
            spacegroup_symbol,
        )
        info.update(
            {
                "manifest_split": str(context.split),
                "preprocessing_source": self.preprocessing_source,
                "hydrogen_policy": (
                    "add_missing" if self.add_missing_hydrogens else "explicit_only"
                ),
                "template_policy": (
                    "rdkit"
                    if bool(conditioning_summary.get("template_present"))
                    else "unavailable"
                ),
                "removed_unknown_atom_count": self._removed_unknown_atom_count,
            }
        )
        return info


def _manifest_filters() -> CSDFilterConfig:
    """Return permissive filters because the manifest defines membership."""
    return CSDFilterConfig(
        date_cutoff=None,
        max_atoms=None,
        max_r_factor=None,
        allow_powder=True,
        allow_polymeric=True,
        require_3d_coordinates=False,
        require_ambient_pressure=False,
        require_known_spacegroup=False,
        require_organic_or_organometallic=False,
    )


def _cache_config(source: str, timeout_seconds: float) -> PreprocessCacheConfig:
    """Return one resumable cache configuration for a preprocessing pass."""
    return PreprocessCacheConfig(
        enabled=True,
        cache_dir_name=f"_preprocess_cache/{source}",
        commit_interval=COMMIT_INTERVAL,
        retry_failed=False,
        entry_timeout_seconds=timeout_seconds,
    )


def _cache_path(
    output_dir: Path,
    source: str,
    split: str,
) -> Path:
    """Return one pass-specific intermediate SQLite path."""
    return output_dir / "_preprocess_cache" / source / f"{split}.sqlite3"


def _run_cache_pass(
    manifest: CSDManifest,
    output_dir: Path,
    split: str,
    source: str,
    add_missing_hydrogens: bool,
    generate_rdkit_template: bool,
    n_jobs: int,
    timeout_seconds: float,
) -> Path:
    """Populate one resumable manifest cache and preserve its report."""
    config = _cache_config(source, timeout_seconds)
    pass_n_jobs = min(n_jobs, len(manifest.refcodes(split)))
    preprocessor = CSDManifestPreprocessor(
        manifest=manifest,
        add_missing_hydrogens=add_missing_hydrogens,
        preprocessing_source=source,
        generate_rdkit_template=generate_rdkit_template,
        niggli=True,
        filters=_manifest_filters(),
        rdkit_template=RDKitTemplateConfig(
            random_seed=123,
            relax_with_uff=False,
            uff_max_iters=1000,
        ),
        cache=config,
        n_jobs=pass_n_jobs,
    )
    cache_source_split(
        preprocessor,
        output_dir.parent,
        output_dir.name,
        split,
        config,
    )
    default_report = output_dir / f"preprocessing_report_{split}.json"
    source_report = output_dir / f"preprocessing_report_{split}_{source}.json"
    if default_report.is_file():
        default_report.replace(source_report)
    return _cache_path(output_dir, source, split)


def _read_cache_results(sqlite_path: Path) -> dict[str, CacheEntryResult]:
    """Read status metadata for every intermediate cache row."""
    if not sqlite_path.is_file():
        return {}
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT key, status, failure_reason, failure_detail, stage
            FROM entries
            ORDER BY key
            """
        )
        return {
            str(key).strip().upper(): CacheEntryResult(
                status=str(status),
                failure_reason=(None if reason is None else str(reason)),
                failure_detail=(None if detail is None else str(detail)),
                stage=None if stage is None else str(stage),
            )
            for key, status, reason, detail, stage in rows
        }
    finally:
        conn.close()


def _successful_cache_rows(
    sqlite_path: Path,
    eligible_refcodes: set[str],
    source: str,
) -> Iterator[tuple[str, bytes, str]]:
    """Yield eligible successful cache blobs sorted by refcode."""
    if not sqlite_path.is_file():
        return
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT key, material_blob
            FROM entries INDEXED BY sqlite_autoindex_entries_1
            WHERE status = 'success'
            ORDER BY key
            """
        )
        for key, material_blob in rows:
            refcode = str(key).strip().upper()
            if refcode in eligible_refcodes:
                if material_blob is None:
                    raise ValueError(f"Successful cache row {refcode} has no blob.")
                yield refcode, bytes(material_blob), source
    finally:
        conn.close()


def _open_split_writer(lmdb_path: Path) -> _SplitWriter:
    """Open one LMDB writer and aligned metadata buffers."""
    env = lmdb.open(str(lmdb_path), map_size=LMDB_MAP_SIZE)
    return _SplitWriter(
        env=env,
        txn=env.begin(write=True),
        count=0,
        num_atoms=[],
        families=[],
    )


def _normalize_material(
    material_blob: bytes,
    dataset_name: str,
    manifest_row: CSDManifestRow,
    source: str,
) -> Material:
    """Load one cached material and normalize public-build provenance."""
    material: Material = pickle.loads(material_blob)
    info = dict(material.info or {})
    refcode = manifest_row.refcode
    info.update(
        {
            "dataset_name": dataset_name,
            "material_id": refcode,
            "csd_refcode": refcode,
            "csd_family": refcode[:6],
            "manifest_split": manifest_row.split,
            "preprocessing_source": source,
            "hydrogen_policy": (
                "add_missing" if source == STANDARD_SOURCE else "explicit_only"
            ),
            "template_policy": (
                "rdkit" if material.conditioning.template_present else "unavailable"
            ),
        }
    )
    if manifest_row.truth_refcodes:
        info.update(
            {
                "benchmark_truth_refcodes": list(manifest_row.truth_refcodes),
                "benchmark_truth_group": ";".join(manifest_row.truth_refcodes),
            }
        )
    if manifest_row.flexibility is not None:
        info["flexibility"] = manifest_row.flexibility
    material.info = info
    return material


def _write_material(
    writer: _SplitWriter,
    material: Material,
    refcode: str,
) -> None:
    """Write one material and update aligned metadata buffers."""
    writer.txn.put(f"{writer.count:08d}".encode(), pickle.dumps(material))
    writer.count += 1
    writer.num_atoms.append(int(material.conditioning.atomic_numbers.shape[0]))
    writer.families.append(refcode[:6])
    if writer.count % COMMIT_INTERVAL == 0:
        writer.txn.commit()
        writer.txn = writer.env.begin(write=True)


def _close_split_writer(
    writer: _SplitWriter,
    staging_dir: Path,
    split: str,
) -> None:
    """Finalize one LMDB and its aligned metadata caches."""
    writer.txn.put(b"__len__", str(writer.count).encode())
    writer.txn.commit()
    writer.env.close()
    np.save(
        staging_dir / f"{split}.num_atoms.npy",
        np.asarray(writer.num_atoms, dtype=np.int64),
    )
    np.save(
        staging_dir / f"{split}.csd_families.npy",
        np.asarray(writer.families, dtype="<U6"),
    )


def _failure_for_refcode(
    refcode: str,
    split: str,
    standard_results: Mapping[str, CacheEntryResult],
    recovery_results: Mapping[str, CacheEntryResult],
    no_template_results: Mapping[str, CacheEntryResult],
) -> BuildFailure:
    """Build one unresolved-refcode audit record."""
    return BuildFailure(
        refcode=refcode,
        split=split,
        standard=standard_results.get(refcode),
        recovery=recovery_results.get(refcode),
        no_template_recovery=no_template_results.get(refcode),
    )


def assemble_manifest_lmdbs(
    manifest: CSDManifest,
    cache_paths: Mapping[str, SplitCachePaths],
    staging_dir: Path,
    dataset_name: str,
) -> tuple[dict[str, dict[str, int]], list[BuildFailure]]:
    """Assemble successful cache rows into one LMDB per manifest split."""
    split_summaries: dict[str, dict[str, int]] = {}
    failures: list[BuildFailure] = []
    for split in manifest.splits:
        manifest_rows = manifest.rows_for_split(split)
        eligible_refcodes = {row.refcode for row in manifest_rows}
        paths = cache_paths[split]
        standard_results = _read_cache_results(paths.standard)
        recovery_results = _read_cache_results(paths.recovery)
        no_template_results = _read_cache_results(paths.no_template_recovery)
        recovery_refcodes = {
            refcode
            for refcode in eligible_refcodes
            if standard_results.get(refcode) is None
            or standard_results[refcode].status != "success"
        }
        no_template_refcodes = {
            refcode
            for refcode in recovery_refcodes
            if recovery_results.get(refcode) is not None
            and recovery_results[refcode].status == "timed_out"
        }
        cached_rows = itertools.chain(
            _successful_cache_rows(paths.standard, eligible_refcodes, STANDARD_SOURCE),
            _successful_cache_rows(
                paths.recovery,
                recovery_refcodes,
                RECOVERY_SOURCE,
            ),
            _successful_cache_rows(
                paths.no_template_recovery,
                no_template_refcodes,
                NO_TEMPLATE_SOURCE,
            ),
        )
        material_by_refcode: dict[str, tuple[bytes, str]] = {}
        for refcode, material_blob, source in cached_rows:
            if refcode in material_by_refcode:
                raise ValueError(
                    f"Multiple successful cache sources contain {refcode}."
                )
            material_by_refcode[refcode] = (material_blob, source)
        writer = _open_split_writer(staging_dir / f"{split}.lmdb")
        seen: set[str] = set()
        sources: Counter[str] = Counter()
        try:
            for manifest_row in manifest_rows:
                refcode = manifest_row.refcode
                cached_material = material_by_refcode.get(refcode)
                if cached_material is None:
                    continue
                material_blob, source = cached_material
                material = _normalize_material(
                    material_blob,
                    dataset_name,
                    manifest_row,
                    source,
                )
                atomic_numbers = np.asarray(material.conditioning.atomic_numbers)
                if atomic_numbers.size == 0 or bool(np.any(atomic_numbers <= 0)):
                    raise ValueError(f"Material {refcode} has invalid atomic numbers.")
                _write_material(writer, material, refcode)
                seen.add(refcode)
                sources[source] += 1
            _close_split_writer(writer, staging_dir, split)
        except Exception:
            try:
                writer.txn.abort()
            except lmdb.Error:
                pass
            writer.env.close()
            raise

        missing = [row.refcode for row in manifest_rows if row.refcode not in seen]
        failures.extend(
            _failure_for_refcode(
                refcode,
                split,
                standard_results,
                recovery_results,
                no_template_results,
            )
            for refcode in missing
        )
        split_summaries[split] = {
            "expected": len(eligible_refcodes),
            "standard_success": int(sources[STANDARD_SOURCE]),
            "recovery_success": int(sources[RECOVERY_SOURCE]),
            "no_template_recovery_success": int(sources[NO_TEMPLATE_SOURCE]),
            "failed": len(missing),
            "written": len(seen),
        }
    return split_summaries, failures


def _scan_lmdb(
    lmdb_path: Path,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Read refcodes, atom counts, and integrity failures from one LMDB."""
    refcodes: list[str] = []
    atom_counts: list[int] = []
    failures: list[str] = []
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        with env.begin() as txn:
            length_value = txn.get(b"__len__")
            if length_value is None:
                raise ValueError(f"LMDB has no __len__ metadata: {lmdb_path}")
            total = int(length_value.decode())
            for index in range(total):
                value = txn.get(f"{index:08d}".encode())
                if value is None:
                    failures.append(f"missing_lmdb_key:{index}")
                    continue
                material: Material = pickle.loads(value)
                info = material.info or {}
                refcode = str(info.get("csd_refcode", "")).strip().upper()
                if not refcode:
                    failures.append(f"missing_refcode:{index}")
                    continue
                atomic_numbers = np.asarray(material.conditioning.atomic_numbers)
                if atomic_numbers.size == 0 or bool(np.any(atomic_numbers <= 0)):
                    failures.append(f"invalid_atomic_numbers:{refcode}")
                if info.get("manifest_split") != lmdb_path.stem:
                    failures.append(f"invalid_manifest_split:{refcode}")
                if info.get("preprocessing_source") not in {
                    STANDARD_SOURCE,
                    RECOVERY_SOURCE,
                    NO_TEMPLATE_SOURCE,
                }:
                    failures.append(f"invalid_preprocessing_source:{refcode}")
                refcodes.append(refcode)
                atom_counts.append(int(atomic_numbers.shape[0]))
    finally:
        env.close()
    return refcodes, np.asarray(atom_counts, dtype=np.int64), failures


def verify_manifest_lmdbs(
    manifest: CSDManifest,
    staging_dir: Path,
    build_failures: Sequence[BuildFailure],
) -> dict[str, Any]:
    """Verify staged LMDB membership and aligned metadata caches."""
    failed_by_split: dict[str, set[str]] = {
        split: {failure.refcode for failure in build_failures if failure.split == split}
        for split in manifest.splits
    }
    split_payloads: dict[str, Any] = {}
    passed = True
    observed_sets: dict[str, set[str]] = {}
    for split in manifest.splits:
        refcodes, atom_counts, integrity_failures = _scan_lmdb(
            staging_dir / f"{split}.lmdb"
        )
        observed = set(refcodes)
        expected_refcodes = [
            refcode
            for refcode in manifest.refcodes(split)
            if refcode not in failed_by_split[split]
        ]
        expected = set(expected_refcodes)
        cached_atoms = np.load(
            staging_dir / f"{split}.num_atoms.npy",
            allow_pickle=False,
        )
        cached_families = np.load(
            staging_dir / f"{split}.csd_families.npy",
            allow_pickle=False,
        )
        duplicates = len(refcodes) - len(observed)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        atom_cache_matches = bool(np.array_equal(cached_atoms, atom_counts))
        family_cache_matches = bool(
            np.array_equal(
                cached_families.astype(str),
                np.asarray([refcode[:6] for refcode in refcodes]),
            )
        )
        order_matches = refcodes == expected_refcodes
        split_passed = not (
            integrity_failures
            or duplicates
            or missing
            or extra
            or not atom_cache_matches
            or not family_cache_matches
            or not order_matches
        )
        passed = passed and split_passed
        observed_sets[split] = observed
        split_payloads[split] = {
            "passed": split_passed,
            "expected": len(manifest.refcodes(split)),
            "written": len(refcodes),
            "failed": len(failed_by_split[split]),
            "duplicates": duplicates,
            "missing_count": len(missing),
            "extra_count": len(extra),
            "integrity_failure_count": len(integrity_failures),
            "atom_cache_matches": atom_cache_matches,
            "family_cache_matches": family_cache_matches,
            "order_matches": order_matches,
            "examples": {
                "missing": missing[:20],
                "extra": extra[:20],
                "integrity_failures": integrity_failures[:20],
            },
        }

    overlap_count = 0
    split_names = list(manifest.splits)
    for split_index, split in enumerate(split_names):
        for other_split in split_names[split_index + 1 :]:
            overlap_count += len(observed_sets[split] & observed_sets[other_split])
    passed = passed and overlap_count == 0
    return {
        "passed": passed,
        "complete": not build_failures,
        "manifest_sha256": manifest.sha256,
        "splits": split_payloads,
        "cross_split_overlap_count": overlap_count,
        "failed_refcode_count": len(build_failures),
    }


def _final_output_names(manifest: CSDManifest) -> list[str]:
    """Return all final split artifact names."""
    return [
        name
        for split in manifest.splits
        for name in (
            f"{split}.lmdb",
            f"{split}.num_atoms.npy",
            f"{split}.csd_families.npy",
        )
    ]


def _publish_staging_outputs(
    manifest: CSDManifest,
    staging_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    """Publish verified staged outputs into the dataset directory."""
    names = _final_output_names(manifest)
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Final outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for name in names:
        (staging_dir / name).replace(output_dir / name)
    staging_dir.rmdir()


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically with parent directories created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _result_columns(prefix: str) -> list[str]:
    """Return failure CSV columns for one preprocessing pass."""
    return [
        f"{prefix}_status",
        f"{prefix}_reason",
        f"{prefix}_detail",
        f"{prefix}_stage",
    ]


def _result_values(result: CacheEntryResult | None) -> list[str]:
    """Return CSV-safe status values for one preprocessing pass."""
    if result is None:
        return ["missing", "", "", ""]
    return [
        result.status,
        result.failure_reason or "",
        result.failure_detail or "",
        result.stage or "",
    ]


def _write_failures_csv(path: Path, failures: Sequence[BuildFailure]) -> None:
    """Write unresolved manifest rows and all pass outcomes."""
    fieldnames = [
        "id",
        "split",
        *_result_columns("standard"),
        *_result_columns("recovery"),
        *_result_columns("no_template_recovery"),
    ]
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for failure in sorted(failures, key=lambda item: (item.split, item.refcode)):
            writer.writerow(
                [
                    failure.refcode,
                    failure.split,
                    *_result_values(failure.standard),
                    *_result_values(failure.recovery),
                    *_result_values(failure.no_template_recovery),
                ]
            )
    temporary_path.replace(path)


def _database_entry_count() -> int:
    """Return the number of entries in the installed CSD release."""
    from ccdc import io

    return int(len(io.EntryReader("CSD")))


def build_csd_dataset(
    manifest_path: Path,
    output_dir: Path,
    n_jobs: int = 64,
    entry_timeout_seconds: float = 180.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build every split in a public CSD manifest."""
    if n_jobs <= 0:
        raise ValueError(f"n_jobs must be positive, got {n_jobs}.")
    if entry_timeout_seconds <= 0:
        raise ValueError(
            f"entry_timeout_seconds must be positive, got {entry_timeout_seconds}."
        )
    manifest = load_csd_manifest(manifest_path)
    csd_entry_count = _database_entry_count()
    resolved_output_dir = output_dir.expanduser().resolve()
    if not resolved_output_dir.name:
        raise ValueError(f"Output directory must have a dataset name: {output_dir}")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        resolved_output_dir / name
        for name in _final_output_names(manifest)
        if (resolved_output_dir / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Final outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    cache_paths: dict[str, SplitCachePaths] = {}
    for split in manifest.splits:
        standard_path = _run_cache_pass(
            manifest,
            resolved_output_dir,
            split,
            STANDARD_SOURCE,
            add_missing_hydrogens=True,
            generate_rdkit_template=True,
            n_jobs=n_jobs,
            timeout_seconds=entry_timeout_seconds,
        )
        standard_results = _read_cache_results(standard_path)
        recovery_refcodes = {
            refcode
            for refcode in manifest.refcodes(split)
            if standard_results.get(refcode) is None
            or standard_results[refcode].status != "success"
        }
        recovery_path = _cache_path(resolved_output_dir, RECOVERY_SOURCE, split)
        if recovery_refcodes:
            recovery_path = _run_cache_pass(
                manifest.subset(recovery_refcodes),
                resolved_output_dir,
                split,
                RECOVERY_SOURCE,
                add_missing_hydrogens=False,
                generate_rdkit_template=True,
                n_jobs=n_jobs,
                timeout_seconds=entry_timeout_seconds,
            )
        recovery_results = _read_cache_results(recovery_path)
        no_template_refcodes = {
            refcode
            for refcode in recovery_refcodes
            if recovery_results.get(refcode) is not None
            and recovery_results[refcode].status == "timed_out"
        }
        no_template_path = _cache_path(
            resolved_output_dir,
            NO_TEMPLATE_SOURCE,
            split,
        )
        if no_template_refcodes:
            no_template_path = _run_cache_pass(
                manifest.subset(no_template_refcodes),
                resolved_output_dir,
                split,
                NO_TEMPLATE_SOURCE,
                add_missing_hydrogens=False,
                generate_rdkit_template=False,
                n_jobs=n_jobs,
                timeout_seconds=entry_timeout_seconds,
            )
        cache_paths[split] = SplitCachePaths(
            standard=standard_path,
            recovery=recovery_path,
            no_template_recovery=no_template_path,
        )

    staging_dir = resolved_output_dir / ".lmdb_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()
    try:
        split_summaries, failures = assemble_manifest_lmdbs(
            manifest,
            cache_paths,
            staging_dir,
            resolved_output_dir.name,
        )
        verification = verify_manifest_lmdbs(manifest, staging_dir, failures)
        if not verification["passed"]:
            raise RuntimeError(
                f"Staged CSD dataset failed integrity verification: {verification}"
            )
        _publish_staging_outputs(
            manifest,
            staging_dir,
            resolved_output_dir,
            overwrite,
        )
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    total_expected = sum(summary["expected"] for summary in split_summaries.values())
    total_written = sum(summary["written"] for summary in split_summaries.values())
    dataset_manifest = {
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "output_dir": str(resolved_output_dir),
        "dataset_name": resolved_output_dir.name,
        "splits": split_summaries,
        "total_expected": total_expected,
        "total_written": total_written,
        "total_failed": len(failures),
        "complete": not failures,
        "n_jobs": n_jobs,
        "entry_timeout_seconds": entry_timeout_seconds,
        "csd_entry_count": csd_entry_count,
        "filters_applied": False,
        "deduplication_applied": False,
        "benchmark_carving_applied": False,
        "rdkit_template": {
            "attempted_for_standard": True,
            "attempted_for_recovery": True,
            "disabled_only_after_recovery_timeout": True,
            "random_seed": 123,
            "relax_with_uff": False,
            "uff_max_iters": 1000,
        },
    }
    _write_failures_csv(resolved_output_dir / "failed_refcodes.csv", failures)
    _write_json(resolved_output_dir / "verification.json", verification)
    _write_json(resolved_output_dir / "dataset_manifest.json", dataset_manifest)
    return dataset_manifest
