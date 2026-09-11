"""Aggregate evaluated crystal candidates with the CLARI resampling protocol."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SAMPLE_METRIC_FIELDS: dict[str, str] = {
    "Col_S": "clash",
    "Pac_S": "passed",
    "Rec_S": "recovered",
}
CRYSTAL_METRIC_FIELDS: dict[str, str] = {
    "Pac_C": "passed",
    "Rec_C": "recovered",
    "Sol_C": "matched",
}
METRIC_FIELDS: dict[str, str] = {
    **SAMPLE_METRIC_FIELDS,
    **CRYSTAL_METRIC_FIELDS,
}
DEFAULT_DRAW_SIZE = 30
DEFAULT_BOOTSTRAP_REPLICATES = 5_000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_REPLICATE_CHUNK_SIZE = 64


@dataclass(frozen=True)
class CorrectnessArtifact:
    """Store target-by-sample correctness matrices and aligned metadata."""

    target_ids: np.ndarray
    csd_refcodes: np.ndarray
    flexibilities: np.ndarray
    sample_indices: np.ndarray
    matrices: dict[str, np.ndarray]


@dataclass(frozen=True)
class ClariEvaluation:
    """Store the CLARI summary and replicate-level metric values."""

    summary: dict[str, Any]
    replicates: dict[str, np.ndarray]


def build_correctness_artifact(
    rows: Sequence[Mapping[str, Any]],
    expected_pool_size: int | None = None,
) -> CorrectnessArtifact:
    """Build complete target-by-sample correctness matrices from evaluated rows."""
    if not rows:
        raise ValueError("Cannot build CLARI correctness matrices without rows.")

    targets: OrderedDict[int, dict[str, str]] = OrderedDict()
    samples_by_target: dict[int, set[int]] = {}
    seen_rows: set[tuple[int, int]] = set()
    for row in rows:
        target_id = int(row["dataset_index"])
        sample_index = int(row["sample_index"])
        row_key = (target_id, sample_index)
        if row_key in seen_rows:
            raise ValueError(f"Duplicate CLARI row identity: {row_key}.")
        seen_rows.add(row_key)

        refcode = str(row["csd_refcode"])
        flexibility = str(row.get("flexibility", ""))
        if target_id not in targets:
            targets[target_id] = {
                "csd_refcode": refcode,
                "flexibility": flexibility,
            }
        elif targets[target_id] != {
            "csd_refcode": refcode,
            "flexibility": flexibility,
        }:
            raise ValueError(
                f"Inconsistent target metadata for dataset index {target_id}."
            )
        samples_by_target.setdefault(target_id, set()).add(sample_index)

    pool_size = (
        max(len(sample_indices) for sample_indices in samples_by_target.values())
        if expected_pool_size is None
        else int(expected_pool_size)
    )
    if pool_size < 1:
        raise ValueError("expected_pool_size must be >= 1.")

    expected_indices = list(range(pool_size))
    invalid_targets = {
        target_id: sorted(sample_indices)
        for target_id, sample_indices in samples_by_target.items()
        if sorted(sample_indices) != expected_indices
    }
    if invalid_targets:
        observed = {
            target_id: {
                "count": len(indices),
                "min": indices[0] if indices else None,
                "max": indices[-1] if indices else None,
            }
            for target_id, indices in invalid_targets.items()
        }
        raise ValueError(
            "CLARI requires sample indices 0.."
            f"{pool_size - 1} exactly once per target; invalid={observed}.",
        )

    target_ids = np.asarray(list(targets), dtype=np.int64)
    target_positions = {
        int(target_id): position
        for position, target_id in enumerate(target_ids.tolist())
    }
    matrices = {
        metric: np.zeros((len(target_ids), pool_size), dtype=np.bool_)
        for metric in METRIC_FIELDS
    }
    for row in rows:
        target_position = target_positions[int(row["dataset_index"])]
        sample_index = int(row["sample_index"])
        for metric, field_name in METRIC_FIELDS.items():
            matrices[metric][target_position, sample_index] = bool(row[field_name])

    return CorrectnessArtifact(
        target_ids=target_ids,
        csd_refcodes=np.asarray(
            [targets[int(target_id)]["csd_refcode"] for target_id in target_ids],
            dtype="U64",
        ),
        flexibilities=np.asarray(
            [targets[int(target_id)]["flexibility"] for target_id in target_ids],
            dtype="U16",
        ),
        sample_indices=np.arange(pool_size, dtype=np.int64),
        matrices=matrices,
    )


def evaluate_clari(
    artifact: CorrectnessArtifact,
    draw_size: int = DEFAULT_DRAW_SIZE,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    replicate_chunk_size: int = DEFAULT_REPLICATE_CHUNK_SIZE,
) -> ClariEvaluation:
    """Compute CLARI bootstrap and exact coverage metrics."""
    draws_per_target = int(draw_size)
    replicate_count = int(bootstrap_replicates)
    chunk_size = int(replicate_chunk_size)
    if draws_per_target < 1:
        raise ValueError("draw_size must be >= 1.")
    if replicate_count < 1:
        raise ValueError("bootstrap_replicates must be >= 1.")
    if chunk_size < 1:
        raise ValueError("replicate_chunk_size must be >= 1.")

    num_targets = int(artifact.target_ids.shape[0])
    pool_size = int(artifact.sample_indices.shape[0])
    if num_targets < 1 or pool_size < 1:
        raise ValueError("CLARI correctness matrices must be non-empty.")
    expected_shape = (num_targets, pool_size)
    if set(artifact.matrices) != set(METRIC_FIELDS):
        raise ValueError(
            f"CLARI correctness matrices must contain exactly {sorted(METRIC_FIELDS)}.",
        )
    for metric, matrix in artifact.matrices.items():
        if matrix.shape != expected_shape:
            raise ValueError(
                f"CLARI matrix {metric} has shape {matrix.shape}, expected {expected_shape}.",
            )

    rng = np.random.default_rng(int(bootstrap_seed))
    replicate_values = {
        metric: np.empty(replicate_count, dtype=np.float64) for metric in METRIC_FIELDS
    }
    target_positions = np.arange(num_targets, dtype=np.int64)[None, :, None]
    for start in range(0, replicate_count, chunk_size):
        stop = min(start + chunk_size, replicate_count)
        candidate_indices = rng.integers(
            0,
            pool_size,
            size=(stop - start, num_targets, draws_per_target),
            dtype=np.int64,
        )
        for metric, matrix in artifact.matrices.items():
            selected = matrix[target_positions, candidate_indices]
            replicate_values[metric][start:stop] = (
                selected.mean(axis=(1, 2), dtype=np.float64)
                if metric in SAMPLE_METRIC_FIELDS
                else selected.any(axis=2).mean(axis=1, dtype=np.float64)
            )

    bootstrap_budget = f"{draws_per_target}/{draws_per_target}"
    exact_budget = f"{pool_size}/{pool_size}"
    metrics: dict[str, Any] = {}
    for metric, matrix in artifact.matrices.items():
        values = replicate_values[metric]
        exact_value = (
            matrix.mean(dtype=np.float64)
            if metric in SAMPLE_METRIC_FIELDS
            else matrix.any(axis=1).mean(dtype=np.float64)
        )
        metrics[metric] = {
            bootstrap_budget: {
                "mean": float(values.mean()),
                "standard_error": float(values.std(ddof=0)),
            },
            exact_budget: {
                "value": float(exact_value),
            },
        }

    summary = {
        "protocol": "clari",
        "num_targets": num_targets,
        "pool_size": pool_size,
        "bootstrap": {
            "draw_size": draws_per_target,
            "replicates": replicate_count,
            "seed": int(bootstrap_seed),
            "replacement": True,
            "uncertainty": "population_standard_deviation_of_replicate_scores",
        },
        "metrics": metrics,
    }
    return ClariEvaluation(summary=summary, replicates=replicate_values)


def write_correctness_artifact(path: Path, artifact: CorrectnessArtifact) -> None:
    """Write correctness matrices and aligned target metadata atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            target_ids=artifact.target_ids,
            csd_refcodes=artifact.csd_refcodes,
            flexibilities=artifact.flexibilities,
            sample_indices=artifact.sample_indices,
            **artifact.matrices,
        )
    temporary.replace(path)


def write_replicate_artifact(
    path: Path,
    evaluation: ClariEvaluation,
) -> None:
    """Write replicate-level CLARI metric values atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **evaluation.replicates)
    temporary.replace(path)


def flat_clari_metrics(summary: Mapping[str, Any]) -> dict[str, float | int]:
    """Flatten public CLARI summary metrics for generic metrics artifacts."""
    output: dict[str, float | int] = {
        "num_targets": int(summary["num_targets"]),
        "pool_size": int(summary["pool_size"]),
    }
    draw_size = int(dict(summary["bootstrap"])["draw_size"])
    pool_size = int(summary["pool_size"])
    bootstrap_budget = f"{draw_size}/{draw_size}"
    exact_budget = f"{pool_size}/{pool_size}"
    for metric, budgets in dict(summary["metrics"]).items():
        output[f"{metric}_{draw_size}_mean"] = float(
            budgets[bootstrap_budget]["mean"],
        )
        output[f"{metric}_{draw_size}_standard_error"] = float(
            budgets[bootstrap_budget]["standard_error"],
        )
        output[f"{metric}_{pool_size}"] = float(budgets[exact_budget]["value"])
    return output
