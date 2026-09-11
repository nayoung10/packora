"""Official OXtal molecular crystal structure prediction metrics."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from ccdc import io as csdio
from ccdc.crystal import Crystal, PackingSimilarity
from ccdc.io import CrystalReader

BENCHMARK_MANIFEST_NAMES = ("rigid", "flexible")
DEFAULT_PACKING_SIZES = (1, 15)
MATCH_FRACTION = 0.5

_ENTRY_READER: csdio.EntryReader | None = None


@dataclass(frozen=True)
class OXtalComparison:
    """Container for one official OXtal sample comparison."""

    csd_refcode: str
    best_true_refcode: str
    clash: bool
    passed: bool
    errors: tuple[str, ...]
    nmatched_1: int
    rmsd_1: float | None
    nmatched_15: int
    rmsd_15: float | None

    @property
    def matched_molecules(self) -> int:
        """Return the size-15 matched molecule count for legacy callers."""
        return int(self.nmatched_15)


def _normalize_refcode(value: object) -> str:
    """Return an uppercase CSD refcode or raise a useful error."""
    refcode = str(value).strip().upper()
    if not refcode:
        raise ValueError("Official OXtal evaluation requires a non-empty csd_refcode.")
    return refcode


def _finite_or_none(value: object) -> float | None:
    """Return a finite float or None."""
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _mean_bool(values: Sequence[bool]) -> float:
    """Return the arithmetic mean of boolean indicator values."""
    if not values:
        raise ValueError("Cannot average an empty indicator sequence.")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _unique_in_order(values: Iterable[str]) -> list[str]:
    """Return unique strings in first-seen order."""
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def load_truth_map(path: Path | None = None) -> dict[str, list[str]]:
    """Load polymorph groups from the downloaded public benchmark manifests."""
    truth_path = (
        path or Path(os.environ.get("PACKORA_DATA_ROOT", "data")) / "csv_manifests"
    )
    paths = (
        [truth_path / f"{name}.csv" for name in BENCHMARK_MANIFEST_NAMES]
        if truth_path.is_dir() or path is None
        else [truth_path]
    )
    mapping: dict[str, list[str]] = {}
    for manifest_path in paths:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Benchmark manifest not found: {manifest_path}. Download "
                "nayoung10/Packora-data and set PACKORA_DATA_ROOT."
            )
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "id" not in (reader.fieldnames or []):
                raise ValueError(
                    f"Truth manifest must contain an id column: {manifest_path}"
                )
            for row in reader:
                primary = _normalize_refcode(row["id"])
                raw = str(row.get("truth_refcodes") or primary).strip() or primary
                refs = _unique_in_order(
                    _normalize_refcode(item) for item in raw.split(";") if item.strip()
                )
                if primary not in refs:
                    raise ValueError(f"Truth group omits primary refcode {primary}")
                for ref in refs:
                    if ref in mapping and set(mapping[ref]) != set(refs):
                        raise ValueError(f"Conflicting truth groups for {ref}")
                    mapping[ref] = list(refs)
    return mapping


def truth_set_for_refcode(
    csd_refcode: str,
    truth_map: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Return the official truth set, falling back to singleton official behavior."""
    refcode = _normalize_refcode(csd_refcode)
    mapping = load_truth_map() if truth_map is None else truth_map
    refs = mapping.get(refcode)
    if refs is None:
        return [refcode]
    return [_normalize_refcode(ref) for ref in refs]


def _get_entry_reader(
    entry_reader: csdio.EntryReader | None = None,
) -> csdio.EntryReader:
    """Return a per-process CSD EntryReader."""
    if entry_reader is not None:
        return entry_reader

    global _ENTRY_READER
    if _ENTRY_READER is None:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        _ENTRY_READER = csdio.EntryReader()
    return _ENTRY_READER


def has_collision(crystal: Crystal, clash_cutoff: float = 0.7) -> bool:
    """Return official OXtal heavy-atom steric clash status."""
    overlaps: list[float] = []
    try:
        for contact in crystal.molecule.contacts(
            distance_range=(-5.0, 0.0),
            only_strongest=False,
        ):
            atom_a, atom_b = contact.atoms
            if atom_a.atomic_symbol == "H" or atom_b.atomic_symbol == "H":
                continue
            overlap = max(
                0.0,
                float(atom_a.vdw_radius)
                + float(atom_b.vdw_radius)
                - float(contact.length),
            )
            if overlap > 0.0:
                overlaps.append(overlap)
    except Exception:
        overlaps = []
    return bool(any(overlap >= float(clash_cutoff) for overlap in overlaps))


def _safe_cif_filename(identifier: str) -> str:
    """Return a filesystem-safe temporary CIF file name."""
    safe = "".join(
        character if character.isalnum() or character in {"_", "-", "."} else "_"
        for character in str(identifier)
    ).strip("._")
    return f"{safe or 'query'}.cif"


def load_crystal_from_cif_text(cif_text: str, identifier: str = "query") -> Crystal:
    """Load one CIF string through CCDC CrystalReader."""
    with tempfile.TemporaryDirectory(prefix="packora_oxtal_") as tmp_dir:
        cif_path = Path(tmp_dir) / _safe_cif_filename(identifier)
        cif_path.write_text(cif_text, encoding="utf-8")
        with CrystalReader(str(cif_path)) as reader:
            if len(reader) == 0:
                raise ValueError(f"No crystals found in temporary CIF: {cif_path}")
            return reader[0]


def make_packing_similarity(
    packing_shell_size: int,
    distance_tolerance: float = 0.5,
    angle_tolerance: float = 75.0,
    timeout_ms: int = 10000,
    allow_molecular_differences: bool = False,
) -> PackingSimilarity:
    """Build a COMPACK engine with official OXtal settings for one shell size."""
    engine = PackingSimilarity()
    settings = engine.settings
    size = int(packing_shell_size)
    settings.packing_shell_size = size
    if size == 1:
        settings.distance_tolerance = 0.2
        settings.angle_tolerance = 20
        settings.match_entire_packing_shell = True
    else:
        settings.distance_tolerance = float(distance_tolerance)
        settings.angle_tolerance = float(angle_tolerance)
        settings.match_entire_packing_shell = False
    settings.timeout_ms = int(timeout_ms)
    settings.allow_molecular_differences = bool(allow_molecular_differences)
    settings.ignore_hydrogen_counts = True
    settings.ignore_hydrogen_positions = True
    settings.ignore_bond_counts = True
    settings.ignore_bond_types = True
    settings.allow_artificial_inversion = True
    return engine


def _compare_size(
    target: Crystal,
    sample: Crystal,
    packing_shell_size: int,
    distance_tolerance: float,
    angle_tolerance: float,
    timeout_ms: int,
    allow_molecular_differences: bool,
) -> tuple[int, float | None]:
    """Compare one target/sample pair at one official packing shell size."""
    engine = make_packing_similarity(
        packing_shell_size=int(packing_shell_size),
        distance_tolerance=float(distance_tolerance),
        angle_tolerance=float(angle_tolerance),
        timeout_ms=int(timeout_ms),
        allow_molecular_differences=bool(allow_molecular_differences),
    )
    result = engine.compare(target, sample)
    if result is None:
        return 0, None
    return int(result.nmatched_molecules), _finite_or_none(result.rmsd)


def _empty_results_by_size(packing_sizes: Sequence[int]) -> dict[int, dict[str, Any]]:
    """Return zero-match results for all requested packing sizes."""
    return {int(size): {"nmatched": 0, "rmsd": None} for size in packing_sizes}


def _candidate_rank(
    result_by_size: Mapping[int, Mapping[str, Any]],
    largest_packing_size: int,
    need_matches: int,
) -> tuple[int, float, int]:
    """Return the official OXtal rank tuple for one target candidate."""
    largest = result_by_size[largest_packing_size]
    nmatched = int(largest["nmatched"])
    rmsd = _finite_or_none(largest["rmsd"])
    passed = nmatched >= int(need_matches)
    return (
        0 if passed else 1,
        float(rmsd) if passed and rmsd is not None else float("inf"),
        -nmatched,
    )


def compare_packing(
    sample: Crystal,
    csd_refcode: str,
    truth_map: Mapping[str, Sequence[str]] | None = None,
    entry_reader: csdio.EntryReader | None = None,
    packing_sizes: Sequence[int] = DEFAULT_PACKING_SIZES,
    distance_tolerance: float = 0.5,
    angle_tolerance: float = 75.0,
    timeout_ms: int = 10000,
    allow_molecular_differences: bool = False,
    clash_cutoff: float = 0.7,
) -> OXtalComparison:
    """Compare one CrystalReader-loaded sample against official OXtal truth crystals."""
    refcode = _normalize_refcode(csd_refcode)
    sizes = tuple(int(size) for size in packing_sizes)
    largest_size = max(sizes)
    need_matches = int(math.ceil(largest_size * MATCH_FRACTION))
    truth_refs = truth_set_for_refcode(refcode, truth_map=truth_map)
    reader = _get_entry_reader(entry_reader)

    best_refcode = ""
    best_results = _empty_results_by_size(sizes)
    best_passed = False
    errors: list[str] = []
    query = sample
    clash = has_collision(query, clash_cutoff=float(clash_cutoff))

    for truth_refcode in truth_refs:
        current_results = _empty_results_by_size(sizes)
        for size in sizes:
            try:
                target = reader.entry(truth_refcode).crystal
                nmatched, rmsd = _compare_size(
                    target=target,
                    sample=query,
                    packing_shell_size=size,
                    distance_tolerance=float(distance_tolerance),
                    angle_tolerance=float(angle_tolerance),
                    timeout_ms=int(timeout_ms),
                    allow_molecular_differences=bool(allow_molecular_differences),
                )
            except Exception as exc:
                errors.append(f"truth_load_error:{truth_refcode}:{size}:{exc}")
                nmatched, rmsd = 0, None
            current_results[size] = {"nmatched": int(nmatched), "rmsd": rmsd}

        current_rank = _candidate_rank(current_results, largest_size, need_matches)
        best_rank = _candidate_rank(best_results, largest_size, need_matches)
        if current_rank < best_rank:
            best_refcode = truth_refcode
            best_results = current_results
            best_passed = bool(best_results[largest_size]["nmatched"] >= need_matches)

    return OXtalComparison(
        csd_refcode=refcode,
        best_true_refcode=best_refcode,
        clash=bool(clash),
        passed=bool(best_passed),
        errors=tuple(errors),
        nmatched_1=int(best_results.get(1, {"nmatched": 0})["nmatched"]),
        rmsd_1=_finite_or_none(best_results.get(1, {"rmsd": None})["rmsd"]),
        nmatched_15=int(best_results.get(15, {"nmatched": 0})["nmatched"]),
        rmsd_15=_finite_or_none(best_results.get(15, {"rmsd": None})["rmsd"]),
    )


def compute_conformer_rmsd_1(
    sample: Crystal,
    csd_refcode: str,
    truth_map: Mapping[str, Sequence[str]] | None = None,
    entry_reader: csdio.EntryReader | None = None,
) -> float | None:
    """Return official OXtal COMPACK size-1 RMSD for one sample."""
    return compare_packing(
        sample=sample,
        csd_refcode=csd_refcode,
        truth_map=truth_map,
        entry_reader=entry_reader,
    ).rmsd_1


def _is_recovered(row: Mapping[str, Any], threshold: float = 0.5) -> bool:
    """Return official OXtal recovery indicator for one row."""
    rmsd_1 = _finite_or_none(row.get("rmsd_1"))
    return bool(
        rmsd_1 is not None and rmsd_1 < float(threshold) and not bool(row["clash"])
    )


def _is_match(row: Mapping[str, Any], threshold: float = 2.0) -> bool:
    """Return official OXtal match-rate indicator for one row."""
    rmsd_15 = _finite_or_none(row.get("rmsd_15"))
    return bool(
        bool(row["passed"])
        and rmsd_15 is not None
        and rmsd_15 < float(threshold)
        and not bool(row["clash"])
    )


def aggregate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate official OXtal rows with Packora's public metric names."""
    if not rows:
        raise ValueError("Cannot aggregate empty official OXtal rows.")

    codes = _unique_in_order(_normalize_refcode(row["csd_refcode"]) for row in rows)
    n_filtered = len(rows)
    n_dataset = len(codes)
    passed_codes = {
        _normalize_refcode(row["csd_refcode"]) for row in rows if bool(row["passed"])
    }
    recovered_codes = {
        _normalize_refcode(row["csd_refcode"]) for row in rows if _is_recovered(row)
    }
    matched_codes = {
        _normalize_refcode(row["csd_refcode"]) for row in rows if _is_match(row)
    }

    return {
        "Col_S": float(sum(bool(row["clash"]) for row in rows) / n_filtered),
        "Pac_S": float(sum(bool(row["passed"]) for row in rows) / n_filtered),
        "Pac_C": float(len(passed_codes) / n_dataset),
        "Rec_S": float(sum(_is_recovered(row) for row in rows) / n_filtered),
        "Rec_C": float(len(recovered_codes) / n_dataset),
        "Sol_C": float(len(matched_codes) / n_dataset),
    }


def _validate_inputs(
    target_crystals: Sequence[Crystal],
    samples_by_target: Sequence[Sequence[Crystal]],
    target_refcodes: Sequence[str] | None,
) -> list[str]:
    """Validate grouped official OXtal evaluation inputs."""
    if not target_crystals:
        raise ValueError("target_crystals must contain at least one target entry.")
    if len(target_crystals) != len(samples_by_target):
        raise ValueError(
            "samples_by_target must have one sample sequence per target entry."
        )
    if target_refcodes is None:
        raise ValueError("Official OXtal evaluation requires target_refcodes.")
    if len(target_refcodes) != len(samples_by_target):
        raise ValueError("target_refcodes must have one CSD refcode per target entry.")
    for target_index, samples in enumerate(samples_by_target):
        if not samples:
            raise ValueError(
                f"samples_by_target[{target_index}] must contain at least one sample."
            )
    return [_normalize_refcode(refcode) for refcode in target_refcodes]


def evaluate(
    target_crystals: Sequence[Crystal],
    samples_by_target: Sequence[Sequence[Crystal]],
    target_refcodes: Sequence[str] | None = None,
    truth_map_path: Path | None = None,
    packing_sizes: Sequence[int] = DEFAULT_PACKING_SIZES,
    distance_tolerance: float = 0.5,
    angle_tolerance: float = 75.0,
    timeout_ms: int = 10000,
    allow_molecular_differences: bool = False,
    clash_cutoff: float = 0.7,
    entry_reader: csdio.EntryReader | None = None,
) -> dict[str, object]:
    """Evaluate official OXtal metrics for samples grouped by CSD refcode."""
    refcodes = _validate_inputs(target_crystals, samples_by_target, target_refcodes)
    truth_map = load_truth_map(truth_map_path)
    reader = _get_entry_reader(entry_reader)

    per_sample: list[dict[str, object]] = []
    per_target: list[dict[str, object]] = []

    for target_index, refcode in enumerate(refcodes):
        target_rows: list[dict[str, object]] = []
        for sample_index, sample in enumerate(samples_by_target[target_index]):
            comparison = compare_packing(
                sample=sample,
                csd_refcode=refcode,
                truth_map=truth_map,
                entry_reader=reader,
                packing_sizes=packing_sizes,
                distance_tolerance=float(distance_tolerance),
                angle_tolerance=float(angle_tolerance),
                timeout_ms=int(timeout_ms),
                allow_molecular_differences=bool(allow_molecular_differences),
                clash_cutoff=float(clash_cutoff),
            )
            row: dict[str, object] = {
                "target_index": int(target_index),
                "target_identifier": refcode,
                "csd_refcode": comparison.csd_refcode,
                "sample_index": int(sample_index),
                "best_true_refcode": comparison.best_true_refcode,
                "clash": bool(comparison.clash),
                "passed": bool(comparison.passed),
                "errors": ";".join(comparison.errors),
                "nmatched_1": int(comparison.nmatched_1),
                "rmsd_1": comparison.rmsd_1,
                "nmatched_15": int(comparison.nmatched_15),
                "rmsd_15": comparison.rmsd_15,
            }
            row["recovered"] = _is_recovered(row)
            row["matched"] = _is_match(row)
            per_sample.append(row)
            target_rows.append(row)

        per_target.append(
            {
                "target_index": int(target_index),
                "target_identifier": refcode,
                "csd_refcode": refcode,
                "num_samples": int(len(target_rows)),
                "passed": bool(any(bool(row["passed"]) for row in target_rows)),
                "recovered": bool(any(bool(row["recovered"]) for row in target_rows)),
                "matched": bool(any(bool(row["matched"]) for row in target_rows)),
            },
        )

    return {
        "summary": aggregate_summary(per_sample),
        "per_sample": per_sample,
        "per_target": per_target,
    }
