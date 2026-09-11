"""Evaluate saved CSP predictions with official OXtal metrics."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from eval.oxtal import _is_match, _is_recovered, aggregate_summary
from eval.oxtal import compare_packing, has_collision
from src.models.callbacks.validation_metrics import (
    StructurePayload,
    _payload_to_crystal,
)
from src.prediction.io import PredictionBundle, load_prediction_bundle

DEFAULT_WORKERS = 32
DEFAULT_PARALLEL_MODE = "sample-chunk"
DEFAULT_SAMPLES_PER_TASK = 1
DEFAULT_SKIP_CLASH = True
DEFAULT_CACHE_FLUSH_ROWS = 10_000
CACHE_FILENAME = "evaluation_cache.sqlite3"
CACHE_SCHEMA_VERSION = 1
CACHE_LOGIC_VERSION = 1
FLEXIBILITY_LABELS = ("rigid", "flexible")
DEFAULT_PROTOCOL = "oxtal"
PROTOCOLS = (DEFAULT_PROTOCOL, "clari")

_WORKER_BUNDLE: PredictionBundle | None = None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline crystal evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--parallel-mode",
        choices=("target", "sample-chunk"),
        default=DEFAULT_PARALLEL_MODE,
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=DEFAULT_SAMPLES_PER_TASK,
    )
    parser.add_argument(
        "--skip-clash",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SKIP_CLASH,
    )
    parser.add_argument(
        "--cache-results",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cache-flush-rows",
        type=int,
        default=DEFAULT_CACHE_FLUSH_ROWS,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if int(args.workers) < 1:
        parser.error("--workers must be >= 1.")
    if int(args.samples_per_task) < 1:
        parser.error("--samples-per-task must be >= 1.")
    if int(args.cache_flush_rows) < 1:
        parser.error("--cache-flush-rows must be >= 1.")
    return args


def _to_jsonable(value: Any) -> Any:
    """Convert nested values into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object if it exists."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows as newline-delimited JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(dict(row)), sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    """Write a two-column summary CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "value"))
        writer.writeheader()
        for key in sorted(summary):
            writer.writerow({"metric": key, "value": _to_jsonable(summary[key])})
    temporary.replace(path)


def _remove_cache_files(path: Path) -> None:
    """Remove one SQLite cache and its transient auxiliary files."""
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)


def _prediction_identity(
    predictions_path: Path,
    manifest: Mapping[str, Any],
    verification: Mapping[str, Any],
    bundle: PredictionBundle,
) -> dict[str, Any]:
    """Build a stable identity for one prediction bundle."""
    digest = verification.get("predictions_sha256") or manifest.get(
        "predictions_sha256"
    )
    key_digest = hashlib.sha256()
    for dataset_index, sample_index in zip(
        bundle.dataset_indices.tolist(),
        bundle.sample_indices.tolist(),
        strict=True,
    ):
        key_digest.update(f"{int(dataset_index)}:{int(sample_index)}\n".encode())

    identity: dict[str, Any] = {
        "rows": int(bundle.dataset_indices.shape[0]),
        "row_keys_sha256": key_digest.hexdigest(),
    }
    if digest:
        identity["predictions_sha256"] = str(digest)
    else:
        stat = predictions_path.stat()
        identity.update(
            {
                "resolved_path": str(predictions_path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return identity


def _cache_signature(
    predictions_path: Path,
    manifest: Mapping[str, Any],
    verification: Mapping[str, Any],
    bundle: PredictionBundle,
    protocol: str,
    skip_clash: bool,
) -> dict[str, Any]:
    """Build the semantic compatibility signature for cached rows."""
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "logic_version": CACHE_LOGIC_VERSION,
        "prediction": _prediction_identity(
            predictions_path,
            manifest,
            verification,
            bundle,
        ),
        "protocol": str(protocol),
        "skip_clash": bool(skip_clash),
    }


class EvaluationResultCache:
    """Persist raw evaluation rows in crash-safe SQLite transactions."""

    def __init__(
        self,
        path: Path,
        signature: Mapping[str, Any],
        flush_rows: int,
    ) -> None:
        """Open or initialize a compatible result cache."""
        self.path = path
        self.flush_rows = int(flush_rows)
        if self.flush_rows < 1:
            raise ValueError("flush_rows must be >= 1.")
        self._pending: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        try:
            self._initialize(dict(signature))
        except Exception:
            self._connection.close()
            raise

    def _initialize(self, signature: dict[str, Any]) -> None:
        """Create the cache schema and validate its semantic signature."""
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS results ("
            "dataset_index INTEGER NOT NULL, "
            "sample_index INTEGER NOT NULL, "
            "payload TEXT NOT NULL, "
            "PRIMARY KEY (dataset_index, sample_index))"
        )
        encoded = json.dumps(_to_jsonable(signature), sort_keys=True)
        stored = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'signature'"
        ).fetchone()
        if stored is None:
            self._connection.execute(
                "INSERT INTO metadata (key, value) VALUES ('signature', ?)",
                (encoded,),
            )
            self._connection.commit()
        elif str(stored[0]) != encoded:
            raise ValueError(
                "Evaluation cache is incompatible with this run; "
                "rerun with --overwrite to discard it."
            )

    def load_rows(self) -> list[dict[str, Any]]:
        """Load committed rows in stable target/sample order."""
        records = self._connection.execute(
            "SELECT payload FROM results ORDER BY dataset_index, sample_index"
        ).fetchall()
        rows = [json.loads(str(record[0])) for record in records]
        if any(not isinstance(row, dict) for row in rows):
            raise TypeError(f"Expected JSON objects in evaluation cache {self.path}.")
        return rows

    def add_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Buffer completed rows and commit each full flush chunk."""
        self._pending.extend(dict(row) for row in rows)
        while len(self._pending) >= self.flush_rows:
            chunk = self._pending[: self.flush_rows]
            del self._pending[: self.flush_rows]
            self._commit(chunk)

    def _commit(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Commit one result chunk as a single SQLite transaction."""
        values = [
            (
                int(row["dataset_index"]),
                int(row["sample_index"]),
                json.dumps(_to_jsonable(dict(row)), sort_keys=True),
            )
            for row in rows
        ]
        with self._connection:
            self._connection.executemany(
                "INSERT INTO results (dataset_index, sample_index, payload) "
                "VALUES (?, ?, ?)",
                values,
            )

    def flush(self) -> None:
        """Commit all currently buffered result rows."""
        if not self._pending:
            return
        rows = self._pending
        self._pending = []
        self._commit(rows)

    def close(self) -> None:
        """Flush buffered results and close the SQLite connection."""
        try:
            self.flush()
        finally:
            self._connection.close()


def _write_done_marker(path: Path) -> None:
    """Write a completion marker for downstream tools."""
    _write_json(
        path,
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
        },
    )


def _metadata_for_row(bundle: PredictionBundle, row_index: int) -> dict[str, Any]:
    """Return metadata values for one prediction row."""
    return {
        key: values[row_index]
        for key, values in bundle.metadata.items()
        if row_index < len(values)
    }


def _csd_refcode_for_row(bundle: PredictionBundle, row_index: int) -> str:
    """Return the required CSD refcode for one prediction row."""
    metadata = _metadata_for_row(bundle, row_index)
    value = metadata.get("csd_refcode") or metadata.get("material_id")
    refcode = "" if value is None else str(value).strip().upper()
    if not refcode:
        raise ValueError(
            f"Missing csd_refcode metadata for prediction row {row_index}."
        )
    return refcode


def _metadata_flexibility(metadata: Mapping[str, Any], row_id: object) -> str:
    """Return the stored rigid/flexible label from one prediction row."""
    value = metadata.get("flexibility")
    label = "" if value is None else str(value).strip().lower()
    if label not in FLEXIBILITY_LABELS:
        raise ValueError(
            f"Missing or invalid flexibility metadata for prediction row {row_id}."
        )
    return label


def _normalize_refcode(value: object) -> str:
    """Return a normalized CSD refcode string."""
    return str(value or "").strip().upper()


def _unique_refcodes(values: Sequence[object]) -> list[str]:
    """Return normalized non-empty refcodes in first-seen order."""
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        refcode = _normalize_refcode(value)
        if refcode and refcode not in seen:
            seen.add(refcode)
            output.append(refcode)
    return output


def _coerce_refcode_group(value: object) -> list[str]:
    """Return refcodes from list-like or semicolon-separated metadata."""
    if value is None:
        return []
    if isinstance(value, str):
        return _unique_refcodes(value.split(";"))
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _unique_refcodes(list(value))
    return _unique_refcodes([value])


def _benchmark_truth_refcodes(
    metadata: Mapping[str, Any],
    fallback_refcode: str,
) -> list[str]:
    """Return benchmark truth refcodes stored in prediction metadata."""
    refcodes = _coerce_refcode_group(metadata.get("benchmark_truth_refcodes"))
    if not refcodes:
        refcodes = _coerce_refcode_group(metadata.get("benchmark_truth_group"))
    if not refcodes:
        refcodes = [_normalize_refcode(fallback_refcode)]
    return refcodes


def _truth_map_for_target(
    target: Mapping[str, Any],
) -> dict[str, list[str]] | None:
    """Build an OXtal truth map for a grouped benchmark target."""
    truth_refcodes = _coerce_refcode_group(target.get("truth_refcodes"))
    if not truth_refcodes:
        return None
    return {refcode: list(truth_refcodes) for refcode in truth_refcodes}


def _payload_from_bundle(
    bundle: PredictionBundle,
    row_index: int,
    source: str,
    identifier: str,
) -> StructurePayload:
    """Build one unbatched structure payload from a prediction bundle."""
    tensor_payload = bundle.ref if source == "ref" else bundle.pred
    bond_adj = bundle.ref.get("bond_adj")
    return StructurePayload(
        coords=tensor_payload["cart_coords"][row_index],
        lattice=tensor_payload["lattice"][row_index],
        atomic_numbers=tensor_payload["atomic_numbers"][row_index],
        atom_mask=tensor_payload["atom_mask"][row_index],
        bond_adj=None if bond_adj is None else bond_adj[row_index],
        identifier=identifier,
    )


def _group_bundle_rows(bundle: PredictionBundle) -> list[dict[str, Any]]:
    """Group prediction rows by source dataset index."""
    grouped: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for row_index, dataset_index in enumerate(bundle.dataset_indices.tolist()):
        target_id = int(dataset_index)
        sample_index = int(bundle.sample_indices[row_index])
        if target_id not in grouped:
            refcode = _csd_refcode_for_row(bundle, row_index)
            metadata = _metadata_for_row(bundle, row_index)
            truth_refcodes = _benchmark_truth_refcodes(metadata, refcode)
            grouped[target_id] = {
                "dataset_index": target_id,
                "csd_refcode": refcode,
                "truth_refcodes": truth_refcodes,
                "target_row": int(row_index),
                "samples": OrderedDict(),
            }
        grouped[target_id]["samples"][sample_index] = int(row_index)

    targets: list[dict[str, Any]] = []
    for target in grouped.values():
        target["samples"] = OrderedDict(sorted(target["samples"].items()))
        targets.append(target)
    return targets


def _target_context(
    manifest: Mapping[str, Any],
    bundle: PredictionBundle,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Build shared output context for one target."""
    row_index = int(target["target_row"])
    metadata = _metadata_for_row(bundle, row_index)
    refcode = str(target["csd_refcode"])
    context: dict[str, Any] = {
        "dataset_index": int(target["dataset_index"]),
        "target_identifier": refcode,
        "csd_refcode": refcode,
        "material_id": metadata.get("material_id"),
        "flexibility": _metadata_flexibility(metadata, row_index),
        "benchmark_truth_refcodes": list(target.get("truth_refcodes", [refcode])),
        "benchmark_truth_group": ";".join(
            str(value) for value in target.get("truth_refcodes", [refcode])
        ),
    }
    for key in ("epoch", "global_step"):
        if key in manifest:
            context[key] = int(manifest[key])
    return context


def _failed_row(
    target: Mapping[str, Any],
    sample_index: int,
    row_id: int,
    error: str,
    clash: bool = True,
) -> dict[str, Any]:
    """Return one incorrect sample row with an audit error."""
    refcode = str(target["csd_refcode"])
    return {
        "target_index": 0,
        "target_identifier": refcode,
        "dataset_index": int(target["dataset_index"]),
        "csd_refcode": refcode,
        "sample_index": int(sample_index),
        "row_id": int(row_id),
        "best_true_refcode": "",
        "clash": bool(clash),
        "passed": False,
        "errors": str(error),
        "nmatched_1": 0,
        "rmsd_1": None,
        "nmatched_15": 0,
        "rmsd_15": None,
        "recovered": False,
        "matched": False,
    }


def _evaluate_sample(
    bundle: PredictionBundle,
    target: Mapping[str, Any],
    sample_index: int,
    row_id: int,
    skip_clash: bool = DEFAULT_SKIP_CLASH,
) -> dict[str, Any]:
    """Evaluate one generated sample with official OXtal logic."""
    refcode = str(target["csd_refcode"])
    sample_payload = _payload_from_bundle(
        bundle=bundle,
        row_index=int(row_id),
        source="pred",
        identifier=f"{refcode}_sample_{int(sample_index)}",
    )
    try:
        sample_crystal = _payload_to_crystal(sample_payload)
    except Exception as exc:
        return _failed_row(
            target=target,
            sample_index=int(sample_index),
            row_id=int(row_id),
            error=f"sample_conversion_error:{exc}",
        )

    try:
        if bool(skip_clash) and has_collision(sample_crystal):
            return _failed_row(
                target=target,
                sample_index=int(sample_index),
                row_id=int(row_id),
                error="skipped_clash_comparison",
                clash=True,
            )
    except Exception as exc:
        return _failed_row(
            target=target,
            sample_index=int(sample_index),
            row_id=int(row_id),
            error=f"clash_detection_error:{exc}",
        )

    try:
        comparison = compare_packing(
            sample=sample_crystal,
            csd_refcode=refcode,
            truth_map=_truth_map_for_target(target),
        )
    except Exception as exc:
        return _failed_row(
            target=target,
            sample_index=int(sample_index),
            row_id=int(row_id),
            error=f"oxtal_compare_error:{exc}",
        )

    row: dict[str, Any] = {
        "target_index": 0,
        "target_identifier": refcode,
        "dataset_index": int(target["dataset_index"]),
        "csd_refcode": comparison.csd_refcode,
        "sample_index": int(sample_index),
        "row_id": int(row_id),
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
    return row


def _evaluate_target(
    bundle: PredictionBundle,
    target: Mapping[str, Any],
    skip_clash: bool = DEFAULT_SKIP_CLASH,
) -> dict[str, Any]:
    """Evaluate all samples for one target."""
    rows = [
        _evaluate_sample(
            bundle=bundle,
            target=target,
            sample_index=int(sample_index),
            row_id=int(row_id),
            skip_clash=bool(skip_clash),
        )
        for sample_index, row_id in target["samples"].items()
    ]
    return {
        "dataset_index": int(target["dataset_index"]),
        "rows": rows,
    }


def _evaluate_sample_chunk(
    bundle: PredictionBundle,
    target: Mapping[str, Any],
    sample_items: Sequence[tuple[int, int]],
    skip_clash: bool = DEFAULT_SKIP_CLASH,
) -> list[dict[str, Any]]:
    """Evaluate a chunk of samples from one target."""
    return [
        _evaluate_sample(
            bundle=bundle,
            target=target,
            sample_index=int(sample_index),
            row_id=int(row_id),
            skip_clash=bool(skip_clash),
        )
        for sample_index, row_id in sample_items
    ]


def _init_worker(predictions_path: Path) -> None:
    """Load the prediction bundle once per spawned worker process."""
    global _WORKER_BUNDLE
    if _WORKER_BUNDLE is None:
        _WORKER_BUNDLE = load_prediction_bundle(predictions_path)


def _worker_bundle(predictions_path: Path) -> PredictionBundle:
    """Return the worker-local prediction bundle."""
    if _WORKER_BUNDLE is not None:
        return _WORKER_BUNDLE
    return load_prediction_bundle(predictions_path)


def _evaluate_target_worker(
    predictions_path: Path,
    target: Mapping[str, Any],
    skip_clash: bool = DEFAULT_SKIP_CLASH,
) -> dict[str, Any]:
    """Evaluate one target using the worker-local prediction bundle."""
    return _evaluate_target(
        bundle=_worker_bundle(predictions_path),
        target=target,
        skip_clash=bool(skip_clash),
    )


def _evaluate_sample_chunk_worker(
    predictions_path: Path,
    target: Mapping[str, Any],
    sample_items: Sequence[tuple[int, int]],
    skip_clash: bool = DEFAULT_SKIP_CLASH,
) -> list[dict[str, Any]]:
    """Evaluate one sample chunk using the worker-local prediction bundle."""
    return _evaluate_sample_chunk(
        bundle=_worker_bundle(predictions_path),
        target=target,
        sample_items=sample_items,
        skip_clash=bool(skip_clash),
    )


def _sample_chunk_tasks(
    grouped_targets: Sequence[Mapping[str, Any]],
    samples_per_task: int,
) -> list[tuple[Mapping[str, Any], list[tuple[int, int]]]]:
    """Split target samples into chunked evaluation tasks."""
    tasks: list[tuple[Mapping[str, Any], list[tuple[int, int]]]] = []
    chunk_size = int(samples_per_task)
    for target in grouped_targets:
        sample_items = [
            (int(sample_index), int(row_id))
            for sample_index, row_id in target["samples"].items()
        ]
        for start in range(0, len(sample_items), chunk_size):
            tasks.append((target, sample_items[start : start + chunk_size]))
    return tasks


def _evaluate_targets(
    predictions_dir: Path,
    bundle: PredictionBundle,
    workers: int,
    show_progress: bool,
    parallel_mode: str = DEFAULT_PARALLEL_MODE,
    samples_per_task: int = DEFAULT_SAMPLES_PER_TASK,
    skip_clash: bool = DEFAULT_SKIP_CLASH,
    initial_rows: Sequence[Mapping[str, Any]] = (),
    result_sink: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all prediction rows with optional process parallelism."""
    global _WORKER_BUNDLE
    _WORKER_BUNDLE = bundle

    grouped_targets = _group_bundle_rows(bundle)
    expected_keys = {
        (int(dataset_index), int(bundle.sample_indices[row_index]))
        for row_index, dataset_index in enumerate(bundle.dataset_indices.tolist())
    }
    cached_rows = [dict(row) for row in initial_rows]
    cached_keys = {
        (int(row["dataset_index"]), int(row["sample_index"])) for row in cached_rows
    }
    if len(cached_keys) != len(cached_rows):
        raise ValueError("Evaluation cache contains duplicate sample keys.")
    unknown_cached_keys = cached_keys.difference(expected_keys)
    if unknown_cached_keys:
        raise ValueError(
            f"Evaluation cache contains unknown sample keys: "
            f"{sorted(unknown_cached_keys)[:5]}."
        )

    pending_targets: list[dict[str, Any]] = []
    for target in grouped_targets:
        dataset_index = int(target["dataset_index"])
        pending_samples = OrderedDict(
            (int(sample_index), int(row_id))
            for sample_index, row_id in target["samples"].items()
            if (dataset_index, int(sample_index)) not in cached_keys
        )
        if pending_samples:
            pending_targets.append({**target, "samples": pending_samples})

    predictions_path = predictions_dir / "predictions.pt"
    mode = str(parallel_mode)
    if mode not in {"target", "sample-chunk"}:
        raise ValueError(f"Unknown parallel_mode: {parallel_mode}")

    tasks = (
        []
        if mode == "target"
        else _sample_chunk_tasks(pending_targets, int(samples_per_task))
    )
    total_tasks = len(pending_targets) if mode == "target" else len(tasks)
    worker_count = min(int(workers), max(1, total_tasks))
    progress = tqdm(
        total=total_tasks,
        desc="oxtal targets" if mode == "target" else "oxtal sample chunks",
        leave=False,
        disable=not show_progress,
    )
    new_rows: list[dict[str, Any]] = []
    new_keys: set[tuple[int, int]] = set()

    def record_rows(rows: Sequence[Mapping[str, Any]]) -> None:
        """Validate, persist, and collect newly completed rows."""
        completed = [dict(row) for row in rows]
        for row in completed:
            key = (int(row["dataset_index"]), int(row["sample_index"]))
            if key not in expected_keys:
                raise ValueError(f"Evaluation returned unknown sample key: {key}.")
            if key in cached_keys or key in new_keys:
                raise ValueError(f"Evaluation returned duplicate sample key: {key}.")
            new_keys.add(key)
        if result_sink is not None:
            result_sink(completed)
        new_rows.extend(completed)

    with progress:
        if mode == "target":
            if worker_count <= 1:
                for target in pending_targets:
                    result = _evaluate_target(
                        bundle=bundle,
                        target=target,
                        skip_clash=bool(skip_clash),
                    )
                    record_rows(result["rows"])
                    progress.update(1)
            else:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_init_worker,
                    initargs=(predictions_path,),
                ) as executor:
                    futures = {
                        executor.submit(
                            _evaluate_target_worker,
                            predictions_path,
                            target,
                            bool(skip_clash),
                        ): target
                        for target in pending_targets
                    }
                    for future in as_completed(futures):
                        record_rows(future.result()["rows"])
                        progress.update(1)
        elif worker_count <= 1:
            for target, sample_items in tasks:
                rows = _evaluate_sample_chunk(
                    bundle=bundle,
                    target=target,
                    sample_items=sample_items,
                    skip_clash=bool(skip_clash),
                )
                record_rows(rows)
                progress.update(1)
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_worker,
                initargs=(predictions_path,),
            ) as executor:
                futures = {
                    executor.submit(
                        _evaluate_sample_chunk_worker,
                        predictions_path,
                        target,
                        sample_items,
                        bool(skip_clash),
                    ): (target, sample_items)
                    for target, sample_items in tasks
                }
                for future in as_completed(futures):
                    record_rows(future.result())
                    progress.update(1)

    rows = sorted(
        [*cached_rows, *new_rows],
        key=lambda row: (int(row["dataset_index"]), int(row["sample_index"])),
    )
    if len(rows) != int(bundle.dataset_indices.shape[0]):
        raise ValueError(
            f"Expected {int(bundle.dataset_indices.shape[0])} OXtal rows, got {len(rows)}.",
        )
    observed_keys = {
        (int(row["dataset_index"]), int(row["sample_index"])) for row in rows
    }
    if observed_keys != expected_keys:
        raise ValueError("Evaluated OXtal row keys do not match the prediction bundle.")
    return rows


def _row_key(row: Mapping[str, Any]) -> tuple[int, int]:
    """Return the stable target/sample key for one metric row."""
    return (int(row["dataset_index"]), int(row["sample_index"]))


def _row_ids_by_key(bundle: PredictionBundle) -> dict[tuple[int, int], int]:
    """Return canonical bundle row ids keyed by target/sample id."""
    return {
        (int(dataset_index), int(bundle.sample_indices[row_index])): int(row_index)
        for row_index, dataset_index in enumerate(bundle.dataset_indices.tolist())
    }


def _target_contexts_by_dataset(
    manifest: Mapping[str, Any],
    bundle: PredictionBundle,
) -> dict[int, dict[str, Any]]:
    """Return source context rows keyed by dataset index."""
    return {
        int(target["dataset_index"]): _target_context(manifest, bundle, target)
        for target in _group_bundle_rows(bundle)
    }


def _target_indices_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """Return stable target positions from evaluated rows."""
    dataset_indices = sorted({int(row["dataset_index"]) for row in rows})
    return {
        dataset_index: target_index
        for target_index, dataset_index in enumerate(dataset_indices)
    }


def _finalize_per_sample_rows(
    rows: Sequence[Mapping[str, Any]],
    row_ids: Mapping[tuple[int, int], int],
    target_contexts: Mapping[int, Mapping[str, Any]],
    target_indices: Mapping[int, int],
) -> list[dict[str, Any]]:
    """Attach canonical row ids, target indices, and target context."""
    output: list[dict[str, Any]] = []
    for row in rows:
        key = _row_key(row)
        dataset_index = int(row["dataset_index"])
        finalized = {
            **dict(row),
            **dict(target_contexts[dataset_index]),
            "target_index": int(target_indices[dataset_index]),
            "sample_index": int(row["sample_index"]),
            "row_id": int(row_ids[key]),
        }
        output.append(finalized)
    return output


def _attach_flexibility(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize stored flexibility metadata on metric rows."""
    output: list[dict[str, Any]] = []
    for row in rows:
        row_id = int(row.get("row_id", row["dataset_index"]))
        label = _metadata_flexibility(row, row_id)
        output.append({**dict(row), "flexibility": label})
    return output


def _group_rows_by_dataset(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Group metric rows by validation dataset index."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["dataset_index"]), []).append(dict(row))
    return grouped


def _aggregate_oxtal_per_target_rows(
    rows: Sequence[Mapping[str, Any]],
    target_contexts: Mapping[int, Mapping[str, Any]],
    target_indices: Mapping[int, int],
) -> list[dict[str, Any]]:
    """Aggregate OXtal sample rows into target-level rows."""
    grouped = _group_rows_by_dataset(rows)
    output: list[dict[str, Any]] = []
    for dataset_index in sorted(grouped, key=lambda key: target_indices[key]):
        target_rows = grouped[dataset_index]
        context = dict(target_contexts[dataset_index])
        output.append(
            {
                "target_index": int(target_indices[dataset_index]),
                "target_identifier": str(context["target_identifier"]),
                "csd_refcode": str(context["csd_refcode"]),
                "dataset_index": int(dataset_index),
                "num_samples": int(len(target_rows)),
                "passed": bool(any(bool(row["passed"]) for row in target_rows)),
                "recovered": bool(any(bool(row["recovered"]) for row in target_rows)),
                "matched": bool(any(bool(row["matched"]) for row in target_rows)),
                **context,
            },
        )
    return output


def _filter_flexibility_rows(
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    """Return rows belonging to one strict flexibility label."""
    return [dict(row) for row in rows if str(row.get("flexibility")) == str(label)]


def _prefixed_summary(prefix: str, summary: Mapping[str, float]) -> dict[str, float]:
    """Prefix all metric names in one summary payload."""
    return {f"{prefix}{key}": float(value) for key, value in summary.items()}


def _aggregate_flexibility_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate rigid/flexible OXtal summaries."""
    summary: dict[str, Any] = {}
    for label in FLEXIBILITY_LABELS:
        label_rows = _filter_flexibility_rows(rows, label)
        if not label_rows:
            continue
        prefix = f"{label}_"
        summary[f"{prefix}num_targets"] = int(
            len({str(row["csd_refcode"]) for row in label_rows}),
        )
        summary[f"{prefix}num_samples"] = int(len(label_rows))
        summary.update(_prefixed_summary(prefix, aggregate_summary(label_rows)))
    return summary


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    elapsed_seconds: float,
    worker_count: int,
    parallel_mode: str,
    samples_per_task: int,
    skip_clash: bool,
) -> dict[str, Any]:
    """Build the public summary payload."""
    summary: dict[str, Any] = {
        **aggregate_summary(rows),
        **_aggregate_flexibility_summary(rows),
        "worker_count": int(worker_count),
        "parallel_mode": str(parallel_mode),
        "samples_per_task": int(samples_per_task),
        "skip_clash": bool(skip_clash),
        "total_time_seconds": float(elapsed_seconds),
        "elapsed_seconds": float(elapsed_seconds),
        "num_targets": int(len({int(row["dataset_index"]) for row in rows})),
        "num_samples": int(len(rows)),
    }
    for key in ("split", "epoch", "global_step"):
        if key in manifest:
            summary[key] = manifest[key]
    return summary


def evaluate_predictions_dir(
    predictions_dir: Path,
    output_dir: Path | None = None,
    workers: int = DEFAULT_WORKERS,
    parallel_mode: str = DEFAULT_PARALLEL_MODE,
    samples_per_task: int = DEFAULT_SAMPLES_PER_TASK,
    skip_clash: bool = DEFAULT_SKIP_CLASH,
    show_progress: bool = True,
    overwrite: bool = False,
    protocol: str = DEFAULT_PROTOCOL,
    cache_results: bool = False,
    cache_flush_rows: int = DEFAULT_CACHE_FLUSH_ROWS,
) -> dict[str, Any]:
    """Evaluate one prediction directory and write protocol artifacts."""
    selected_protocol = str(protocol)
    if selected_protocol not in PROTOCOLS:
        raise ValueError(f"Unknown evaluation protocol: {selected_protocol}.")
    if output_dir is not None:
        eval_dir = output_dir
    elif selected_protocol == "clari":
        eval_dir = predictions_dir / "eval" / "clari"
    else:
        eval_dir = predictions_dir / "eval"
    done_path = eval_dir / "eval.done"
    cache_path = eval_dir / CACHE_FILENAME
    if done_path.is_file() and not bool(overwrite):
        _remove_cache_files(cache_path)
        return {
            "status": "skipped",
            "output_dir": str(eval_dir),
        }
    if bool(overwrite):
        _remove_cache_files(cache_path)

    predictions_path = predictions_dir / "predictions.pt"
    bundle = load_prediction_bundle(predictions_path)
    manifest = _read_json(predictions_dir / "manifest.json")
    verification = _read_json(predictions_dir / "verification.json")

    cache: EvaluationResultCache | None = None
    cached_rows: list[dict[str, Any]] = []
    if bool(cache_results):
        cache = EvaluationResultCache(
            path=cache_path,
            signature=_cache_signature(
                predictions_path=predictions_path,
                manifest=manifest,
                verification=verification,
                bundle=bundle,
                protocol=selected_protocol,
                skip_clash=bool(skip_clash),
            ),
            flush_rows=int(cache_flush_rows),
        )
        try:
            cached_rows = cache.load_rows()
        except Exception:
            cache.close()
            raise
        print(
            f"Evaluation cache: reused {len(cached_rows)}/"
            f"{int(bundle.dataset_indices.shape[0])} samples",
            flush=True,
        )

    grouped_targets = _group_bundle_rows(bundle)
    cached_keys = {
        (int(row["dataset_index"]), int(row["sample_index"])) for row in cached_rows
    }
    pending_counts = [
        sum(
            (int(target["dataset_index"]), int(sample_index)) not in cached_keys
            for sample_index in target["samples"]
        )
        for target in grouped_targets
    ]
    if str(parallel_mode) == "target":
        total_tasks = sum(count > 0 for count in pending_counts)
    else:
        total_tasks = sum(
            (count + int(samples_per_task) - 1) // int(samples_per_task)
            for count in pending_counts
        )
    worker_count = min(int(workers), max(1, total_tasks))
    started = perf_counter()
    try:
        rows = _evaluate_targets(
            predictions_dir=predictions_dir,
            bundle=bundle,
            workers=worker_count,
            show_progress=bool(show_progress),
            parallel_mode=str(parallel_mode),
            samples_per_task=int(samples_per_task),
            skip_clash=bool(skip_clash),
            initial_rows=cached_rows,
            result_sink=None if cache is None else cache.add_rows,
        )
    finally:
        if cache is not None:
            cache.close()
    elapsed_seconds = float(perf_counter() - started)

    row_ids = _row_ids_by_key(bundle)
    target_contexts = _target_contexts_by_dataset(manifest, bundle)
    target_indices = _target_indices_from_rows(rows)
    per_sample = _finalize_per_sample_rows(
        rows=rows,
        row_ids=row_ids,
        target_contexts=target_contexts,
        target_indices=target_indices,
    )
    per_sample = _attach_flexibility(per_sample)
    per_target = _aggregate_oxtal_per_target_rows(
        rows=per_sample,
        target_contexts=target_contexts,
        target_indices=target_indices,
    )
    if selected_protocol == "clari":
        from eval import clari as clari_protocol

        artifact = clari_protocol.build_correctness_artifact(per_sample)
        clari_evaluation = clari_protocol.evaluate_clari(artifact)
        summary = {
            **clari_evaluation.summary,
            "worker_count": int(worker_count),
            "parallel_mode": str(parallel_mode),
            "samples_per_task": int(samples_per_task),
            "skip_clash": bool(skip_clash),
            "structural_evaluation_seconds": elapsed_seconds,
        }
        for key in ("split", "epoch", "global_step"):
            if key in manifest:
                summary[key] = manifest[key]
        clari_protocol.write_correctness_artifact(
            eval_dir / "correctness.npz",
            artifact,
        )
        clari_protocol.write_replicate_artifact(
            eval_dir / "bootstrap_replicates.npz",
            clari_evaluation,
        )
        public_metrics = clari_protocol.flat_clari_metrics(summary)
        total_elapsed_seconds = float(perf_counter() - started)
        summary["elapsed_seconds"] = total_elapsed_seconds
        summary["total_time_seconds"] = total_elapsed_seconds
    else:
        summary = _build_summary(
            rows=per_sample,
            manifest=manifest,
            elapsed_seconds=elapsed_seconds,
            worker_count=worker_count,
            parallel_mode=str(parallel_mode),
            samples_per_task=int(samples_per_task),
            skip_clash=bool(skip_clash),
        )
        public_metrics = {
            key: value
            for key, value in summary.items()
            if isinstance(value, (float, int)) and not str(key).endswith("_seconds")
        }
    completed_samples = int(len(per_sample))
    reused_samples = int(len(cached_rows))
    evaluated_samples_this_run = completed_samples - reused_samples
    throughput_samples = (
        evaluated_samples_this_run if bool(cache_results) else completed_samples
    )
    cache_timing = (
        {
            "cache_flush_rows": int(cache_flush_rows),
            "reused_samples": reused_samples,
            "evaluated_samples_this_run": evaluated_samples_this_run,
        }
        if bool(cache_results)
        else {}
    )
    if selected_protocol == "clari":
        timing = {
            "protocol": selected_protocol,
            "predictions_dir": str(predictions_dir),
            "output_dir": str(eval_dir),
            "worker_count": int(worker_count),
            "parallel_mode": str(parallel_mode),
            "samples_per_task": int(samples_per_task),
            "skip_clash": bool(skip_clash),
            "completed_samples": completed_samples,
            **cache_timing,
            "structural_evaluation_seconds": elapsed_seconds,
            "elapsed_seconds": total_elapsed_seconds,
            "total_time_seconds": total_elapsed_seconds,
            "average_samples_per_second": float(
                throughput_samples / total_elapsed_seconds
            )
            if total_elapsed_seconds > 0
            else None,
        }
    else:
        timing = {
            "predictions_dir": str(predictions_dir),
            "output_dir": str(eval_dir),
            "worker_count": int(worker_count),
            "parallel_mode": str(parallel_mode),
            "samples_per_task": int(samples_per_task),
            "skip_clash": bool(skip_clash),
            "completed_samples": completed_samples,
            **cache_timing,
            "elapsed_seconds": elapsed_seconds,
            "total_time_seconds": elapsed_seconds,
            "average_samples_per_second": float(throughput_samples / elapsed_seconds)
            if elapsed_seconds > 0
            else None,
        }
    metrics_payload = {
        "metrics": public_metrics,
        "per_unit_metrics": {},
        "skipped_metrics": {},
    }

    _write_jsonl(eval_dir / "oxtal_per_sample.jsonl", per_sample)
    _write_jsonl(eval_dir / "oxtal_per_target.jsonl", per_target)
    _write_json(eval_dir / "summary.json", summary)
    _write_summary_csv(eval_dir / "summary.csv", summary)
    _write_json(eval_dir / "metrics.json", metrics_payload)
    _write_json(eval_dir / "timing.json", timing)
    _write_done_marker(eval_dir / "eval.done")
    _remove_cache_files(cache_path)
    return {
        "summary": summary,
        "timing": timing,
        "per_sample": per_sample,
        "per_target": per_target,
    }


def main() -> None:
    """Execute offline crystal evaluation from a saved prediction bundle."""
    args = parse_args()
    payload = evaluate_predictions_dir(
        predictions_dir=args.predictions_dir,
        output_dir=args.output_dir,
        workers=int(args.workers),
        parallel_mode=str(args.parallel_mode),
        samples_per_task=int(args.samples_per_task),
        skip_clash=bool(args.skip_clash),
        show_progress=not bool(args.no_progress),
        overwrite=bool(args.overwrite),
        protocol=str(args.protocol),
        cache_results=bool(args.cache_results),
        cache_flush_rows=int(args.cache_flush_rows),
    )
    if payload.get("status") == "skipped":
        print(f"Evaluation already done: {payload['output_dir']}")
        return
    print(json.dumps(_to_jsonable(payload["summary"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
