import json
import logging
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import spglib
from ase import Atoms
from ase.build import niggli_reduce
from joblib import Parallel, delayed
from tqdm import tqdm

from src.data.types import Material

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawEntryRef:
    """Lightweight locator for one raw source entry."""

    key: str
    source_split: Optional[str]
    payload: dict[str, object] = field(default_factory=dict)
    group_key: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessContext:
    """Shared context for one preprocessing run."""

    data_dir: Path
    dataset_name: str
    split: Optional[str]
    source_split: Optional[str] = None


@dataclass(frozen=True)
class ProcessResult:
    """Result returned by one raw-entry worker."""

    material: Optional[Material]
    failure_reason: Optional[str] = None
    skipped: bool = False
    refcode: Optional[str] = None
    failure_detail: Optional[str] = None
    stage: Optional[str] = None


@dataclass(frozen=True)
class PreprocessingReport:
    """Summary written after a preprocessing run."""

    data_dir: Path
    dataset_name: str
    split: str
    total_input: int
    yielded: int
    success_count: int
    skipped_by_reason: dict[str, int]
    failures: dict[str, list[str]]
    skipped_crystals_by_reason: dict[str, list[str]] = field(default_factory=dict)
    failure_details: list[dict[str, object]] = field(default_factory=list)
    skipped_details: list[dict[str, object]] = field(default_factory=list)


def _process_raw_entry(
    preprocessor: "BasePreprocessor",
    ref: RawEntryRef,
    context: PreprocessContext,
) -> tuple[RawEntryRef, ProcessResult]:
    """Return ref with result so unordered workers keep report attribution correct."""
    return ref, preprocessor.raw_entry_to_material(ref, context)


class BasePreprocessor(ABC):
    """Abstract base for dataset preprocessors."""

    dataset_provided_splits: tuple[str, ...] = ()

    def __init__(
        self, primitive: bool = True, niggli: bool = True, n_jobs: int = 1,
    ) -> None:
        """Initialize shared preprocessing options."""
        self.primitive = primitive
        self.niggli = niggli
        self.n_jobs = n_jobs

    def iter_materials(
        self,
        data_dir: Path,
        dataset_name: str,
        split: Optional[str] = None,
    ) -> Iterator[Material]:
        """Yield Materials using the raw-entry template path."""
        context = PreprocessContext(
            data_dir=data_dir,
            dataset_name=dataset_name,
            split=split,
            source_split=split,
        )
        refs = self.iter_raw_entries(context)
        results = Parallel(n_jobs=self.n_jobs, return_as="generator_unordered")(
            delayed(_process_raw_entry)(self, ref, context) for ref in refs
        )

        failures: dict[str, list[str]] = defaultdict(list)
        skipped_by_reason: dict[str, int] = defaultdict(int)
        skipped_crystals_by_reason: dict[str, list[str]] = defaultdict(list)
        failure_details: list[dict[str, object]] = []
        skipped_details: list[dict[str, object]] = []
        yielded = 0
        success_count = 0
        desc = dataset_name if split is None else f"{dataset_name}/{split}"

        try:
            for ref, result in tqdm(results, desc=desc, total=len(refs)):
                reason = result.failure_reason or "unknown_error"
                result_refcode = getattr(result, "refcode", None) or ref.key
                detail = {
                    "ref_key": ref.key,
                    "refcode": result_refcode,
                    "reason": reason,
                    "stage": getattr(result, "stage", None),
                    "detail": getattr(result, "failure_detail", None),
                    "skipped": bool(result.skipped),
                }
                if result.material is None:
                    if result.skipped:
                        skipped_by_reason[reason] += 1
                        skipped_crystals_by_reason[reason].append(result_refcode)
                        skipped_details.append(detail)
                    else:
                        failures[reason].append(result_refcode)
                        failure_details.append(detail)
                    continue

                yielded += 1
                if result.failure_reason is None:
                    success_count += 1
                else:
                    failures[result.failure_reason].append(result_refcode)
                    failure_details.append(detail)
                yield result.material
        finally:
            if split is not None:
                self.write_preprocessing_report(
                    PreprocessingReport(
                        data_dir=data_dir,
                        dataset_name=dataset_name,
                        split=split,
                        total_input=len(refs),
                        yielded=yielded,
                        success_count=success_count,
                        skipped_by_reason=dict(skipped_by_reason),
                        failures=dict(failures),
                        skipped_crystals_by_reason=dict(skipped_crystals_by_reason),
                        failure_details=failure_details,
                        skipped_details=skipped_details,
                    )
                )

    def iter_raw_entries(self, context: PreprocessContext) -> list[RawEntryRef]:
        """Return raw entry references for the requested run."""
        raise NotImplementedError

    def raw_entry_to_material(
        self, ref: RawEntryRef, context: PreprocessContext,
    ) -> ProcessResult:
        """Convert one raw entry reference into a Material."""
        raise NotImplementedError

    def standardize_atoms(self, atoms: Atoms) -> Atoms:
        """Apply primitive and Niggli standardization."""
        if self.primitive:
            cell = (
                atoms.get_cell().array,
                atoms.get_scaled_positions(),
                atoms.get_atomic_numbers(),
            )
            primitive = spglib.find_primitive(cell)
            if primitive is None:
                raise ValueError("spglib could not find primitive cell.")
            lattice, positions, numbers = primitive
            atoms = Atoms(
                numbers=numbers,
                scaled_positions=positions,
                cell=lattice,
                pbc=True,
            )
        if self.niggli:
            niggli_reduce(atoms)
        return atoms

    def load_key_list_split(self, splits_path: Path, split: str) -> set[str]:
        """Load a predefined split key list from JSON."""
        with open(splits_path, "r") as f:
            payload = json.load(f)
        return set(payload[split])

    def filter_refs_by_keys(
        self, refs: list[RawEntryRef], keys: set[str],
    ) -> list[RawEntryRef]:
        """Filter raw entry refs by canonical key."""
        return [ref for ref in refs if ref.key in keys]

    def write_preprocessing_report(self, report: PreprocessingReport) -> None:
        """Write a JSON summary of preprocessing outcomes."""
        failure_total = sum(len(v) for v in report.failures.values())
        skipped_total = sum(report.skipped_by_reason.values())
        payload = {
            "split": report.split,
            "total_input": report.total_input,
            "yielded": report.yielded,
            "skipped_total": skipped_total,
            "skipped_by_reason": report.skipped_by_reason,
            "skipped_crystals_by_reason": report.skipped_crystals_by_reason,
            "skipped_details": report.skipped_details,
            "skipped_max_atoms": report.skipped_by_reason.get("max_atoms", 0),
            "conditioning_success": report.success_count,
            "conditioning_failures_total": failure_total,
            "failure_breakdown": {
                key: len(value) for key, value in report.failures.items()
            },
            "failed_crystals_by_reason": report.failures,
            "failure_details": report.failure_details,
        }
        report_path = (
            report.data_dir / report.dataset_name
            / f"preprocessing_report_{report.split}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            "Preprocessing report saved to %s (success=%d, failures=%d, skipped=%d)",
            report_path, report.success_count, failure_total, skipped_total,
        )
