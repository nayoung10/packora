"""Compute OXtal correctness matrices and pass@k metrics from predictions."""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from tqdm import tqdm

from src.prediction.io import PredictionBundle, load_prediction_bundle

FLEXIBILITY_LABELS = ("rigid", "flexible")
METRIC_FIELDS: dict[str, str] = {
    "Pac_C": "passed",
    "Rec_C": "recovered",
    "Sol_C": "matched",
}
GROUPS = ("all", *FLEXIBILITY_LABELS)
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_530
DEFAULT_PARALLEL_MODE = "sample-chunk"
DEFAULT_SAMPLES_PER_TASK = 1
DEFAULT_SKIP_CLASH = True
_WORKER_BUNDLE: PredictionBundle | None = None


@dataclass(frozen=True)
class CorrectnessArtifact:
    """Container for reusable per-target correctness matrices."""

    target_ids: np.ndarray
    csd_refcodes: np.ndarray
    flexibilities: np.ndarray
    sample_indices: np.ndarray
    matrices: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ks", type=int, nargs="*", default=None)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--reuse-correctness", action="store_true")
    parser.add_argument("--no-rows", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--parallel-mode",
        choices=("target", "sample-chunk"),
        default=DEFAULT_PARALLEL_MODE,
    )
    parser.add_argument(
        "--samples-per-task", type=int, default=DEFAULT_SAMPLES_PER_TASK
    )
    parser.add_argument(
        "--skip-clash",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SKIP_CLASH,
    )
    args = parser.parse_args()
    if int(args.workers) < 1:
        parser.error("--workers must be >= 1.")
    if int(args.bootstrap_samples) < 0:
        parser.error("--bootstrap-samples must be >= 0.")
    if int(args.samples_per_task) < 1:
        parser.error("--samples-per-task must be >= 1.")
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(dict(payload)), handle, indent=2)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(dict(row)), sort_keys=True))
            handle.write("\n")


def _read_manifest(predictions_dir: Path) -> dict[str, Any]:
    """Read an optional generation manifest."""
    manifest_path = predictions_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    with manifest_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {manifest_path}.")
    return payload


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
    refcode = str(metadata.get("csd_refcode") or "").strip().upper()
    if not refcode:
        raise ValueError(
            f"Missing csd_refcode metadata for prediction row {row_index}."
        )
    return refcode


def _flexibility_for_row(bundle: PredictionBundle, row_index: int) -> str:
    """Return strict rigid/flexible metadata for one prediction row."""
    metadata = _metadata_for_row(bundle, row_index)
    value = metadata.get("flexibility")
    label = "" if value is None else str(value).strip().lower()
    if label not in FLEXIBILITY_LABELS:
        raise ValueError(
            f"Missing or invalid flexibility metadata for prediction row {row_index}."
        )
    return label


def _payload_from_bundle(
    bundle: PredictionBundle,
    row_index: int,
    source: str,
    identifier: str,
) -> Any:
    """Build one unbatched structure payload from a prediction bundle."""
    from src.models.callbacks.validation_metrics import StructurePayload

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
            grouped[target_id] = {
                "dataset_index": target_id,
                "csd_refcode": refcode,
                "flexibility": _flexibility_for_row(bundle, row_index),
                "target_row": int(row_index),
                "samples": OrderedDict(),
            }
        grouped[target_id]["samples"][sample_index] = int(row_index)

    targets: list[dict[str, Any]] = []
    for target in grouped.values():
        target["samples"] = OrderedDict(sorted(target["samples"].items()))
        targets.append(target)
    return targets


def _init_worker(predictions_path: Path) -> None:
    """Load the prediction bundle once per worker process."""
    global _WORKER_BUNDLE
    _WORKER_BUNDLE = load_prediction_bundle(predictions_path)


def _worker_bundle(predictions_path: Path) -> PredictionBundle:
    """Return the worker-local prediction bundle."""
    if _WORKER_BUNDLE is not None:
        return _WORKER_BUNDLE
    return load_prediction_bundle(predictions_path)


def _failed_row(
    target: Mapping[str, Any],
    sample_index: int,
    row_id: int,
    error: str,
) -> dict[str, Any]:
    """Return one incorrect sample row with an audit error."""
    return {
        "target_index": 0,
        "target_identifier": str(target["csd_refcode"]),
        "dataset_index": int(target["dataset_index"]),
        "csd_refcode": str(target["csd_refcode"]),
        "flexibility": str(target["flexibility"]),
        "sample_index": int(sample_index),
        "row_id": int(row_id),
        "best_true_refcode": "",
        "clash": True,
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
    """Evaluate one generated sample with skip-on-clash OXtal logic."""
    from eval.oxtal import _is_match, _is_recovered, compare_packing, has_collision
    from src.models.callbacks.validation_metrics import _payload_to_crystal

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

    if bool(skip_clash) and has_collision(sample_crystal):
        return _failed_row(
            target=target,
            sample_index=int(sample_index),
            row_id=int(row_id),
            error="skipped_clash_comparison",
        )

    try:
        comparison = compare_packing(sample=sample_crystal, csd_refcode=refcode)
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
        "flexibility": str(target["flexibility"]),
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
    """Evaluate OXtal rows for one target and all its samples."""
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


def _evaluate_target_worker(
    predictions_path: Path,
    target: Mapping[str, Any],
    skip_clash: bool = DEFAULT_SKIP_CLASH,
) -> dict[str, Any]:
    """Evaluate one target using the worker-local prediction bundle."""
    return _evaluate_target(
        _worker_bundle(predictions_path), target, skip_clash=bool(skip_clash)
    )


def _evaluate_sample_chunk_worker(
    predictions_path: Path,
    target: Mapping[str, Any],
    sample_items: Sequence[tuple[int, int]],
    skip_clash: bool = DEFAULT_SKIP_CLASH,
) -> list[dict[str, Any]]:
    """Evaluate one sample chunk using the worker-local prediction bundle."""
    return _evaluate_sample_chunk(
        _worker_bundle(predictions_path),
        target,
        sample_items,
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
) -> list[dict[str, Any]]:
    """Evaluate OXtal rows for all targets with optional process parallelism."""
    grouped_targets = _group_bundle_rows(bundle)
    predictions_path = predictions_dir / "predictions.pt"
    mode = str(parallel_mode)
    if mode not in {"target", "sample-chunk"}:
        raise ValueError(f"Unknown parallel_mode: {parallel_mode}")

    total_tasks = (
        len(grouped_targets)
        if mode == "target"
        else len(_sample_chunk_tasks(grouped_targets, int(samples_per_task)))
    )
    worker_count = min(int(workers), max(1, total_tasks))
    results: list[dict[str, Any]] = []
    chunk_results: list[list[dict[str, Any]]] = []
    progress = tqdm(
        total=total_tasks,
        desc="oxtal targets" if mode == "target" else "oxtal sample chunks",
        leave=False,
        disable=not show_progress,
    )
    with progress:
        if mode == "target":
            if worker_count <= 1:
                for target in grouped_targets:
                    results.append(
                        _evaluate_target(bundle, target, skip_clash=bool(skip_clash))
                    )
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
                        for target in grouped_targets
                    }
                    for future in as_completed(futures):
                        results.append(future.result())
                        progress.update(1)
        else:
            tasks = _sample_chunk_tasks(grouped_targets, int(samples_per_task))
            if worker_count <= 1:
                for target, sample_items in tasks:
                    chunk_results.append(
                        _evaluate_sample_chunk(
                            bundle,
                            target,
                            sample_items,
                            skip_clash=bool(skip_clash),
                        )
                    )
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
                        chunk_results.append(future.result())
                        progress.update(1)

    if mode == "target":
        rows = [
            row
            for result in sorted(results, key=lambda item: int(item["dataset_index"]))
            for row in result["rows"]
        ]
    else:
        rows = [row for chunk in chunk_results for row in chunk]
        rows = sorted(
            rows,
            key=lambda row: (int(row["dataset_index"]), int(row["sample_index"])),
        )
    if len(rows) != int(bundle.dataset_indices.shape[0]):
        raise ValueError(
            f"Expected {int(bundle.dataset_indices.shape[0])} OXtal rows, got {len(rows)}.",
        )
    return rows


def stable_pass_at_k(num_samples: int, correct_count: int, k: int) -> float:
    """Return the unbiased pass@k estimate for one target."""
    n = int(num_samples)
    c = int(correct_count)
    sample_count = int(k)
    if c < 0 or c > n:
        raise ValueError(f"correct_count must be between 0 and {n}, got {c}.")
    if sample_count < 1 or sample_count > n:
        raise ValueError(f"k must be between 1 and {n}, got {sample_count}.")
    if n - c < sample_count:
        return 1.0
    terms = 1.0 - sample_count / np.arange(n - c + 1, n + 1, dtype=np.float64)
    return float(1.0 - np.prod(terms))


def _validate_ks(ks: Sequence[int] | None, num_samples: int) -> list[int]:
    """Return validated k values."""
    if ks is None or len(ks) == 0:
        return list(range(1, int(num_samples) + 1))
    output = sorted({int(k) for k in ks})
    invalid = [k for k in output if k < 1 or k > int(num_samples)]
    if invalid:
        raise ValueError(f"k values must be in [1, {num_samples}], found {invalid}.")
    return output


def _build_correctness_artifact(
    rows: Sequence[Mapping[str, Any]],
) -> CorrectnessArtifact:
    """Build target-by-sample correctness matrices from OXtal rows."""
    if not rows:
        raise ValueError("Cannot build correctness matrices without rows.")

    targets: OrderedDict[int, dict[str, Any]] = OrderedDict()
    samples_by_target: dict[int, list[int]] = {}
    for row in rows:
        target_id = int(row["dataset_index"])
        if target_id not in targets:
            targets[target_id] = {
                "csd_refcode": str(row["csd_refcode"]),
                "flexibility": str(row["flexibility"]),
            }
        samples_by_target.setdefault(target_id, []).append(int(row["sample_index"]))

    sample_lists = {
        target_id: sorted(set(sample_indices))
        for target_id, sample_indices in samples_by_target.items()
    }
    first_samples = next(iter(sample_lists.values()))
    mismatched = {
        target_id: sample_indices
        for target_id, sample_indices in sample_lists.items()
        if sample_indices != first_samples
    }
    if mismatched:
        raise ValueError(
            f"pass@k requires a uniform sample count per target; mismatched={mismatched}.",
        )

    target_ids = np.asarray(list(targets.keys()), dtype=np.int64)
    target_pos = {
        int(target_id): pos for pos, target_id in enumerate(target_ids.tolist())
    }
    sample_indices = np.asarray(first_samples, dtype=np.int64)
    sample_pos = {
        int(sample_index): pos for pos, sample_index in enumerate(first_samples)
    }
    matrices = {
        metric: np.zeros((len(target_ids), len(sample_indices)), dtype=np.bool_)
        for metric in METRIC_FIELDS
    }
    for row in rows:
        row_target = target_pos[int(row["dataset_index"])]
        row_sample = sample_pos[int(row["sample_index"])]
        for metric, field_name in METRIC_FIELDS.items():
            matrices[metric][row_target, row_sample] = bool(row[field_name])

    return CorrectnessArtifact(
        target_ids=target_ids,
        csd_refcodes=np.asarray(
            [
                str(targets[int(target_id)]["csd_refcode"])
                for target_id in target_ids.tolist()
            ],
            dtype="U32",
        ),
        flexibilities=np.asarray(
            [
                str(targets[int(target_id)]["flexibility"])
                for target_id in target_ids.tolist()
            ],
            dtype="U16",
        ),
        sample_indices=sample_indices,
        matrices=matrices,
    )


def build_target_stats(rows: Sequence[Mapping[str, Any]]) -> CorrectnessArtifact:
    """Return a reusable correctness artifact from evaluated OXtal rows."""
    return _build_correctness_artifact(rows)


def _write_correctness_artifact(path: Path, artifact: CorrectnessArtifact) -> None:
    """Write correctness matrices and metadata to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        target_ids=artifact.target_ids,
        csd_refcodes=artifact.csd_refcodes,
        flexibilities=artifact.flexibilities,
        sample_indices=artifact.sample_indices,
        **artifact.matrices,
    )
    _write_json(
        path.with_suffix(".json"),
        {
            "path": str(path),
            "metrics": list(METRIC_FIELDS),
            "shape": {
                "targets": int(artifact.target_ids.shape[0]),
                "samples": int(artifact.sample_indices.shape[0]),
            },
            "target_ids": artifact.target_ids,
            "csd_refcodes": artifact.csd_refcodes,
            "flexibilities": artifact.flexibilities,
            "sample_indices": artifact.sample_indices,
        },
    )


def _load_correctness_artifact(path: Path) -> CorrectnessArtifact:
    """Read correctness matrices and metadata from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"Correctness artifact not found: {path}")
    payload = np.load(path, allow_pickle=False)
    return CorrectnessArtifact(
        target_ids=np.asarray(payload["target_ids"], dtype=np.int64),
        csd_refcodes=np.asarray(payload["csd_refcodes"], dtype="U32"),
        flexibilities=np.asarray(payload["flexibilities"], dtype="U16"),
        sample_indices=np.asarray(payload["sample_indices"], dtype=np.int64),
        matrices={
            metric: np.asarray(payload[metric], dtype=np.bool_)
            for metric in METRIC_FIELDS
        },
    )


def _bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> tuple[float | None, float | None]:
    """Return a target-bootstrap 95 percent confidence interval."""
    sample_count = int(bootstrap_samples)
    if sample_count <= 0:
        return None, None
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty value array.")
    if values.size == 1:
        scalar = float(values[0])
        return scalar, scalar
    indices = rng.integers(0, values.size, size=(sample_count, values.size))
    means = values[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(means, [0.025, 0.975])
    return float(ci_low), float(ci_high)


def pass_at_k_records(
    artifact: CorrectnessArtifact,
    ks: Sequence[int] | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Return plot-ready pass@k records from correctness matrices."""
    num_samples = int(artifact.sample_indices.shape[0])
    k_values = _validate_ks(ks, num_samples)
    pass_prob = {
        metric: {
            correct_count: {
                k: stable_pass_at_k(
                    num_samples=num_samples,
                    correct_count=correct_count,
                    k=k,
                )
                for k in k_values
            }
            for correct_count in range(num_samples + 1)
        }
        for metric in METRIC_FIELDS
    }
    rng = np.random.default_rng(int(bootstrap_seed))
    records: list[dict[str, Any]] = []
    for group in GROUPS:
        if group == "all":
            group_mask = np.ones(artifact.flexibilities.shape[0], dtype=np.bool_)
        else:
            group_mask = artifact.flexibilities == group
        if not bool(np.any(group_mask)):
            continue
        for metric, matrix in artifact.matrices.items():
            correct_counts = matrix[group_mask].sum(axis=1).astype(np.int64)
            for k in k_values:
                values = np.asarray(
                    [
                        pass_prob[metric][int(correct_count)][int(k)]
                        for correct_count in correct_counts
                    ],
                    dtype=np.float64,
                )
                ci_low, ci_high = _bootstrap_ci(values, rng, int(bootstrap_samples))
                records.append(
                    {
                        "group": group,
                        "metric": metric,
                        "k": int(k),
                        "value": float(values.mean()),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "num_targets": int(values.shape[0]),
                        "num_samples": int(num_samples),
                        "bootstrap_samples": int(bootstrap_samples),
                    },
                )
    return records


def _matrix_summary(artifact: CorrectnessArtifact) -> dict[str, Any]:
    """Return a compact summary of correctness matrices."""
    return {
        "num_targets": int(artifact.target_ids.shape[0]),
        "num_samples_per_target": int(artifact.sample_indices.shape[0]),
        "groups": {
            group: int(np.sum(artifact.flexibilities == group))
            for group in FLEXIBILITY_LABELS
        },
        "correct_counts": {
            metric: {
                "total": int(matrix.sum()),
                "per_group": {
                    group: int(matrix[artifact.flexibilities == group].sum())
                    for group in FLEXIBILITY_LABELS
                },
            }
            for metric, matrix in artifact.matrices.items()
        },
    }


def evaluate_predictions_dir(
    predictions_dir: Path,
    workers: int = 1,
    ks: Sequence[int] | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    reuse_correctness: bool = False,
    write_rows: bool = True,
    show_progress: bool = True,
    parallel_mode: str = DEFAULT_PARALLEL_MODE,
    samples_per_task: int = DEFAULT_SAMPLES_PER_TASK,
    skip_clash: bool = DEFAULT_SKIP_CLASH,
) -> dict[str, Any]:
    """Compute OXtal pass@k payload from one predictions directory."""
    eval_dir = predictions_dir / "eval"
    correctness_path = eval_dir / "correctness.npz"
    rows_path = eval_dir / "oxtal_rows.jsonl"
    if bool(reuse_correctness):
        artifact = _load_correctness_artifact(correctness_path)
        rows: list[dict[str, Any]] = []
    else:
        bundle = load_prediction_bundle(predictions_dir / "predictions.pt")
        rows = _evaluate_targets(
            predictions_dir=predictions_dir,
            bundle=bundle,
            workers=int(workers),
            show_progress=bool(show_progress),
            parallel_mode=str(parallel_mode),
            samples_per_task=int(samples_per_task),
            skip_clash=bool(skip_clash),
        )
        artifact = _build_correctness_artifact(rows)
        _write_correctness_artifact(correctness_path, artifact)
        if bool(write_rows):
            _write_jsonl(rows_path, rows)

    num_samples = int(artifact.sample_indices.shape[0])
    num_targets = int(artifact.target_ids.shape[0])
    if str(parallel_mode) == "target":
        total_tasks = num_targets
    else:
        total_tasks = int(np.ceil(num_targets * num_samples / int(samples_per_task)))
    k_values = _validate_ks(ks, num_samples)
    manifest = _read_manifest(predictions_dir)
    payload: dict[str, Any] = {
        "predictions_dir": str(predictions_dir),
        "source": "computed_from_correctness"
        if bool(reuse_correctness)
        else "computed_from_predictions",
        "workers": int(min(int(workers), max(1, total_tasks))),
        "parallel_mode": str(parallel_mode),
        "samples_per_task": int(samples_per_task),
        "skip_clash": bool(skip_clash),
        "num_targets": int(num_targets),
        "num_samples_per_target": int(num_samples),
        "k_values": k_values,
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "method": "target_resampling_percentile_95",
        },
        "correctness_path": str(correctness_path),
        "rows_path": str(rows_path) if bool(write_rows) else None,
        "matrix_summary": _matrix_summary(artifact),
        "records": pass_at_k_records(
            artifact=artifact,
            ks=k_values,
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_seed=int(bootstrap_seed),
        ),
    }
    for field_name in (
        "checkpoint",
        "ckpt_path",
        "epoch",
        "completed_epoch",
        "global_step",
    ):
        if field_name in manifest:
            payload[field_name] = manifest[field_name]
    if rows:
        payload["num_error_rows"] = int(sum(bool(row.get("errors")) for row in rows))
    return payload


def main() -> None:
    """Run OXtal pass@k evaluation from the command line."""
    args = parse_args()
    output_path = args.predictions_dir / "eval" / "pass_at_k.json"
    timing_path = args.predictions_dir / "eval" / "oxtal_pass_at_k_timing.json"
    started_at = datetime.now(timezone.utc)
    start_seconds = perf_counter()
    timing_base: dict[str, Any] = {
        "predictions_dir": str(args.predictions_dir),
        "started_at": started_at.isoformat(),
        "output_path": str(output_path),
        "workers": int(args.workers),
        "parallel_mode": str(args.parallel_mode),
        "samples_per_task": int(args.samples_per_task),
        "skip_clash": bool(args.skip_clash),
        "reuse_correctness": bool(args.reuse_correctness),
        "write_rows": not bool(args.no_rows),
        "bootstrap_samples": int(args.bootstrap_samples),
    }
    try:
        payload = evaluate_predictions_dir(
            predictions_dir=args.predictions_dir,
            workers=int(args.workers),
            ks=args.ks,
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=int(args.bootstrap_seed),
            reuse_correctness=bool(args.reuse_correctness),
            write_rows=not bool(args.no_rows),
            show_progress=not bool(args.no_progress),
            parallel_mode=str(args.parallel_mode),
            samples_per_task=int(args.samples_per_task),
            skip_clash=bool(args.skip_clash),
        )
        payload["timing_path"] = str(timing_path)
        _write_json(output_path, payload)
    except Exception as exc:
        finished_at = datetime.now(timezone.utc)
        _write_json(
            timing_path,
            {
                **timing_base,
                "finished_at": finished_at.isoformat(),
                "elapsed_seconds": float(perf_counter() - start_seconds),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finished_at = datetime.now(timezone.utc)
    elapsed_seconds = float(perf_counter() - start_seconds)
    completed_samples = int(payload["num_targets"]) * int(
        payload["num_samples_per_target"]
    )
    _write_json(
        timing_path,
        {
            **timing_base,
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "completed_samples": completed_samples,
            "average_samples_per_second": float(completed_samples / elapsed_seconds)
            if elapsed_seconds > 0
            else None,
            "status": "ok",
        },
    )
    print(json.dumps(_to_jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
