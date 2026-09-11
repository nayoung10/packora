from __future__ import annotations

import json
import sqlite3
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluate import (
    CACHE_FILENAME,
    DEFAULT_CACHE_FLUSH_ROWS,
    DEFAULT_PARALLEL_MODE,
    DEFAULT_SAMPLES_PER_TASK,
    DEFAULT_SKIP_CLASH,
    DEFAULT_WORKERS,
    _aggregate_oxtal_per_target_rows,
    _evaluate_sample,
    _evaluate_targets,
    _group_bundle_rows,
    _sample_chunk_tasks,
    evaluate_predictions_dir,
    parse_args,
)
from src.prediction.io import PredictionBundle


def _bundle() -> PredictionBundle:
    """Build a small prediction bundle with unsorted targets and samples."""
    row_count = 4
    pred = {
        "cart_coords": torch.zeros((row_count, 2, 3)),
        "lattice": torch.zeros((row_count, 9)),
        "atomic_numbers": torch.ones((row_count, 2), dtype=torch.long),
        "atom_mask": torch.ones((row_count, 2), dtype=torch.bool),
    }
    ref = {**pred, "bond_adj": torch.zeros((row_count, 2, 2), dtype=torch.bool)}
    return PredictionBundle(
        pred=pred,
        ref=ref,
        dataset_indices=np.asarray([20, 10, 20, 10], dtype=np.int64),
        sample_indices=np.asarray([1, 1, 0, 0], dtype=np.int64),
        metadata={
            "csd_refcode": ["BBBBBB", "AAAAAA", "BBBBBB", "AAAAAA"],
            "material_id": ["BBBBBB", "AAAAAA", "BBBBBB", "AAAAAA"],
            "flexibility": ["rigid", "flexible", "rigid", "flexible"],
            "benchmark_truth_refcodes": [
                ["BBBBBB", "BBBBBB01"],
                ["AAAAAA"],
                ["BBBBBB", "BBBBBB01"],
                ["AAAAAA"],
            ],
        },
    )


def _metric_row(
    target: dict[str, object], sample_index: int, row_id: int
) -> dict[str, object]:
    """Build one deterministic raw OXtal result row."""
    return {
        "dataset_index": int(target["dataset_index"]),
        "csd_refcode": str(target["csd_refcode"]),
        "sample_index": int(sample_index),
        "row_id": int(row_id),
        "best_true_refcode": "",
        "clash": False,
        "passed": False,
        "errors": "",
        "nmatched_1": 0,
        "rmsd_1": None,
        "nmatched_15": 0,
        "rmsd_15": None,
        "recovered": False,
        "matched": False,
    }


def _cache_keys(path: Path) -> set[tuple[int, int]]:
    """Read committed sample keys from an evaluation cache."""
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT dataset_index, sample_index FROM results"
        ).fetchall()
    return {
        (int(dataset_index), int(sample_index)) for dataset_index, sample_index in rows
    }


def _interrupt_evaluation_after_two_rows(
    monkeypatch: pytest.MonkeyPatch,
    predictions_dir: Path,
    output_dir: Path,
) -> set[tuple[int, int]]:
    """Create a valid partial evaluation cache containing two rows."""
    bundle = _bundle()
    completed: list[tuple[int, int]] = []

    def interrupted_evaluate_sample(
        bundle: PredictionBundle,
        target: dict[str, object],
        sample_index: int,
        row_id: int,
        skip_clash: bool = True,
    ) -> dict[str, object]:
        """Fail after two completed sample rows."""
        del bundle, skip_clash
        if len(completed) == 2:
            raise RuntimeError("injected interruption")
        completed.append((int(target["dataset_index"]), int(sample_index)))
        return _metric_row(target, sample_index, row_id)

    monkeypatch.setattr("src.evaluate.load_prediction_bundle", lambda path: bundle)
    monkeypatch.setattr("src.evaluate._evaluate_sample", interrupted_evaluate_sample)
    with pytest.raises(RuntimeError, match="injected interruption"):
        evaluate_predictions_dir(
            predictions_dir=predictions_dir,
            output_dir=output_dir,
            workers=1,
            show_progress=False,
            cache_results=True,
            cache_flush_rows=2,
        )
    return set(completed)


def test_parse_args_uses_oxtal_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse default OXtal evaluator CLI values."""
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--predictions-dir", "/tmp/preds"])

    args = parse_args()

    assert args.predictions_dir == Path("/tmp/preds")
    assert args.protocol == "oxtal"
    assert args.workers == DEFAULT_WORKERS
    assert args.parallel_mode == DEFAULT_PARALLEL_MODE
    assert args.samples_per_task == DEFAULT_SAMPLES_PER_TASK
    assert args.skip_clash is DEFAULT_SKIP_CLASH
    assert args.overwrite is False
    assert args.cache_results is False
    assert args.cache_flush_rows == DEFAULT_CACHE_FLUSH_ROWS


def test_parse_args_can_disable_skip_clash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse the full-comparison skip-clash override."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--predictions-dir", "/tmp/preds", "--no-skip-clash"],
    )

    args = parse_args()

    assert args.skip_clash is False


def test_parse_args_accepts_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse the force-recompute overwrite flag."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--predictions-dir", "/tmp/preds", "--overwrite"],
    )

    args = parse_args()

    assert args.overwrite is True


def test_parse_args_accepts_clari_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse the opt-in CLARI aggregation protocol."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--predictions-dir", "/tmp/preds", "--protocol", "clari"],
    )

    args = parse_args()

    assert args.protocol == "clari"


def test_parse_args_accepts_result_cache_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse opt-in result caching and its flush size."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--predictions-dir",
            "/tmp/preds",
            "--cache-results",
            "--cache-flush-rows",
            "10",
        ],
    )

    args = parse_args()

    assert args.cache_results is True
    assert args.cache_flush_rows == 10


def test_group_bundle_rows_sorts_samples_per_target() -> None:
    """Group rows by dataset index while sorting samples inside each target."""
    grouped = _group_bundle_rows(_bundle())

    assert [target["dataset_index"] for target in grouped] == [20, 10]
    assert list(grouped[0]["samples"].items()) == [(0, 2), (1, 0)]
    assert list(grouped[1]["samples"].items()) == [(0, 3), (1, 1)]
    assert grouped[0]["truth_refcodes"] == ["BBBBBB", "BBBBBB01"]
    assert grouped[1]["truth_refcodes"] == ["AAAAAA"]


def test_evaluate_sample_uses_benchmark_truth_refcodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass benchmark truth groups into OXtal comparison."""
    captured: dict[str, object] = {}

    class FakeComparison:
        """Small stand-in for the OXtal comparison result."""

        csd_refcode = "BBBBBB"
        best_true_refcode = "BBBBBB01"
        clash = False
        passed = True
        errors: tuple[str, ...] = ()
        nmatched_1 = 1
        rmsd_1 = 0.1
        nmatched_15 = 15
        rmsd_15 = 0.2

    def fake_compare_packing(**kwargs: object) -> FakeComparison:
        """Capture the truth map passed to OXtal comparison."""
        captured.update(kwargs)
        return FakeComparison()

    monkeypatch.setattr("src.evaluate._payload_to_crystal", lambda payload: object())
    monkeypatch.setattr("src.evaluate.has_collision", lambda crystal: False)
    monkeypatch.setattr("src.evaluate.compare_packing", fake_compare_packing)

    bundle = _bundle()
    target = _group_bundle_rows(bundle)[0]
    row = _evaluate_sample(
        bundle=bundle,
        target=target,
        sample_index=0,
        row_id=2,
        skip_clash=True,
    )

    assert captured["csd_refcode"] == "BBBBBB"
    assert captured["truth_map"] == {
        "BBBBBB": ["BBBBBB", "BBBBBB01"],
        "BBBBBB01": ["BBBBBB", "BBBBBB01"],
    }
    assert row["best_true_refcode"] == "BBBBBB01"
    assert row["matched"] is True


def test_sample_chunk_tasks_split_samples_per_target() -> None:
    """Split sample tasks without crossing target boundaries."""
    targets = [
        {
            "dataset_index": 1,
            "samples": OrderedDict([(0, 10), (1, 11), (2, 12)]),
        },
        {
            "dataset_index": 2,
            "samples": OrderedDict([(0, 20), (1, 21)]),
        },
    ]

    tasks = _sample_chunk_tasks(targets, samples_per_task=2)

    assert [(task[0]["dataset_index"], task[1]) for task in tasks] == [
        (1, [(0, 10), (1, 11)]),
        (1, [(2, 12)]),
        (2, [(0, 20), (1, 21)]),
    ]


def test_evaluate_targets_returns_stable_sample_chunk_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return rows ordered by dataset index and sample index."""

    def fake_evaluate_sample(
        bundle: PredictionBundle,
        target: dict[str, object],
        sample_index: int,
        row_id: int,
        skip_clash: bool = True,
    ) -> dict[str, object]:
        """Return a minimal deterministic OXtal row."""
        return {
            "dataset_index": int(target["dataset_index"]),
            "csd_refcode": str(target["csd_refcode"]),
            "sample_index": int(sample_index),
            "row_id": int(row_id),
            "clash": False,
            "passed": False,
            "recovered": False,
            "matched": False,
            "errors": "",
        }

    monkeypatch.setattr("src.evaluate._evaluate_sample", fake_evaluate_sample)

    rows = _evaluate_targets(
        predictions_dir=tmp_path,
        bundle=_bundle(),
        workers=1,
        show_progress=False,
        parallel_mode="sample-chunk",
        samples_per_task=1,
        skip_clash=False,
    )

    assert [(row["dataset_index"], row["sample_index"]) for row in rows] == [
        (10, 0),
        (10, 1),
        (20, 0),
        (20, 1),
    ]


def test_evaluate_targets_target_mode_does_not_build_sample_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Avoid constructing unused per-sample futures in target mode."""

    def fail_sample_tasks(*args: object, **kwargs: object) -> None:
        """Fail if target mode tries to construct sample-chunk tasks."""
        raise AssertionError("sample tasks should not be built")

    def fake_evaluate_target(**kwargs: object) -> dict[str, object]:
        """Return identity rows for one grouped target."""
        target = kwargs["target"]
        assert isinstance(target, dict)
        dataset_index = int(target["dataset_index"])
        samples = target["samples"]
        assert isinstance(samples, dict)
        return {
            "dataset_index": dataset_index,
            "rows": [
                {
                    "dataset_index": dataset_index,
                    "sample_index": int(sample_index),
                }
                for sample_index in samples
            ],
        }

    monkeypatch.setattr("src.evaluate._sample_chunk_tasks", fail_sample_tasks)
    monkeypatch.setattr("src.evaluate._evaluate_target", fake_evaluate_target)

    rows = _evaluate_targets(
        predictions_dir=tmp_path,
        bundle=_bundle(),
        workers=1,
        show_progress=False,
        parallel_mode="target",
    )

    assert [(row["dataset_index"], row["sample_index"]) for row in rows] == [
        (10, 0),
        (10, 1),
        (20, 0),
        (20, 1),
    ]


def test_evaluate_predictions_dir_skips_when_done_exists(tmp_path: Path) -> None:
    """Skip evaluation when the done marker already exists."""
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / "eval.done").write_text("done\n", encoding="utf-8")

    payload = evaluate_predictions_dir(
        predictions_dir=tmp_path,
        output_dir=output_dir,
        show_progress=False,
    )

    assert payload == {
        "status": "skipped",
        "output_dir": str(output_dir),
    }


def test_evaluate_predictions_dir_resumes_cached_rows_and_removes_cache_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resume missing samples and remove the cache after successful evaluation."""
    predictions_path = tmp_path / "predictions.pt"
    predictions_path.write_bytes(b"prediction identity")
    output_dir = tmp_path / "eval"
    first_attempt = _interrupt_evaluation_after_two_rows(
        monkeypatch,
        predictions_dir=tmp_path,
        output_dir=output_dir,
    )

    cache_path = output_dir / CACHE_FILENAME
    cached_keys = _cache_keys(cache_path)
    assert cached_keys == first_attempt
    assert len(cached_keys) == 2

    resumed_attempt: list[tuple[int, int]] = []

    def resumed_evaluate_sample(
        bundle: PredictionBundle,
        target: dict[str, object],
        sample_index: int,
        row_id: int,
        skip_clash: bool = True,
    ) -> dict[str, object]:
        """Evaluate only samples missing from the cache."""
        del bundle, skip_clash
        resumed_attempt.append((int(target["dataset_index"]), int(sample_index)))
        return _metric_row(target, sample_index, row_id)

    monkeypatch.setattr("src.evaluate._evaluate_sample", resumed_evaluate_sample)

    payload = evaluate_predictions_dir(
        predictions_dir=tmp_path,
        output_dir=output_dir,
        workers=1,
        show_progress=False,
        cache_results=True,
        cache_flush_rows=2,
    )

    assert set(resumed_attempt).isdisjoint(cached_keys)
    assert len(resumed_attempt) == 2
    assert payload["timing"]["reused_samples"] == 2
    assert payload["timing"]["evaluated_samples_this_run"] == 2
    assert not cache_path.exists()
    assert not Path(f"{cache_path}-wal").exists()
    assert not Path(f"{cache_path}-shm").exists()


def test_evaluate_predictions_dir_rejects_incompatible_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject cached rows produced with different metric semantics."""
    (tmp_path / "predictions.pt").write_bytes(b"prediction identity")
    output_dir = tmp_path / "eval"
    _interrupt_evaluation_after_two_rows(
        monkeypatch,
        predictions_dir=tmp_path,
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="rerun with --overwrite"):
        evaluate_predictions_dir(
            predictions_dir=tmp_path,
            output_dir=output_dir,
            workers=1,
            skip_clash=False,
            show_progress=False,
            cache_results=True,
            cache_flush_rows=2,
        )


def test_evaluate_predictions_dir_overwrite_discards_partial_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Discard all cached rows when overwrite requests a fresh evaluation."""
    (tmp_path / "predictions.pt").write_bytes(b"prediction identity")
    output_dir = tmp_path / "eval"
    old_keys = _interrupt_evaluation_after_two_rows(
        monkeypatch,
        predictions_dir=tmp_path,
        output_dir=output_dir,
    )
    evaluated_keys: list[tuple[int, int]] = []

    def evaluate_sample(
        bundle: PredictionBundle,
        target: dict[str, object],
        sample_index: int,
        row_id: int,
        skip_clash: bool = True,
    ) -> dict[str, object]:
        """Record every sample recomputed after resetting the cache."""
        del bundle, skip_clash
        evaluated_keys.append((int(target["dataset_index"]), int(sample_index)))
        return _metric_row(target, sample_index, row_id)

    monkeypatch.setattr("src.evaluate._evaluate_sample", evaluate_sample)

    payload = evaluate_predictions_dir(
        predictions_dir=tmp_path,
        output_dir=output_dir,
        workers=1,
        show_progress=False,
        overwrite=True,
        cache_results=True,
        cache_flush_rows=2,
    )

    assert old_keys.issubset(set(evaluated_keys))
    assert len(evaluated_keys) == 4
    assert payload["timing"]["reused_samples"] == 0


def test_evaluate_predictions_dir_overwrite_ignores_done_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run evaluation when overwrite is set even if the done marker exists."""
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    (output_dir / "eval.done").write_text("done\n", encoding="utf-8")

    def fake_load_prediction_bundle(path: Path) -> PredictionBundle:
        """Return the test bundle without reading predictions.pt."""
        assert path == tmp_path / "predictions.pt"
        return _bundle()

    def fake_evaluate_targets(**kwargs: object) -> list[dict[str, object]]:
        """Return deterministic rows for all bundle entries."""
        bundle = kwargs["bundle"]
        assert isinstance(bundle, PredictionBundle)
        return [
            {
                "dataset_index": int(dataset_index),
                "csd_refcode": str(bundle.metadata["csd_refcode"][row_id]),
                "sample_index": int(bundle.sample_indices[row_id]),
                "row_id": int(row_id),
                "best_true_refcode": str(bundle.metadata["csd_refcode"][row_id]),
                "clash": False,
                "passed": False,
                "errors": "",
                "nmatched_1": 0,
                "rmsd_1": None,
                "nmatched_15": 0,
                "rmsd_15": None,
                "recovered": False,
                "matched": False,
            }
            for row_id, dataset_index in enumerate(bundle.dataset_indices.tolist())
        ]

    monkeypatch.setattr(
        "src.evaluate.load_prediction_bundle", fake_load_prediction_bundle
    )
    monkeypatch.setattr("src.evaluate._evaluate_targets", fake_evaluate_targets)

    payload = evaluate_predictions_dir(
        predictions_dir=tmp_path,
        output_dir=output_dir,
        show_progress=False,
        overwrite=True,
    )

    assert payload["summary"]["num_samples"] == 4
    assert payload["summary"]["num_targets"] == 2
    assert set(payload["timing"]) == {
        "predictions_dir",
        "output_dir",
        "worker_count",
        "parallel_mode",
        "samples_per_task",
        "skip_clash",
        "completed_samples",
        "elapsed_seconds",
        "total_time_seconds",
        "average_samples_per_second",
    }
    assert (output_dir / "oxtal_per_sample.jsonl").is_file()


@pytest.mark.parametrize("samples_per_target", [250, 1_000])
def test_clari_protocol_uses_isolated_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    samples_per_target: int,
) -> None:
    """Write CLARI artifacts without disturbing the legacy completion marker."""
    row_count = 2 * samples_per_target
    dataset_indices = np.repeat(np.asarray([10, 20]), samples_per_target)
    sample_indices = np.tile(np.arange(samples_per_target), 2)
    pred = {
        "cart_coords": torch.zeros((row_count, 1, 3)),
        "lattice": torch.zeros((row_count, 9)),
        "atomic_numbers": torch.ones((row_count, 1), dtype=torch.long),
        "atom_mask": torch.ones((row_count, 1), dtype=torch.bool),
    }
    bundle = PredictionBundle(
        pred=pred,
        ref=dict(pred),
        dataset_indices=dataset_indices,
        sample_indices=sample_indices,
        metadata={
            "csd_refcode": ["AAAAAA"] * samples_per_target
            + ["BBBBBB"] * samples_per_target,
            "material_id": ["AAAAAA"] * samples_per_target
            + ["BBBBBB"] * samples_per_target,
            "flexibility": ["rigid"] * samples_per_target
            + ["flexible"] * samples_per_target,
        },
    )

    def fake_evaluate_targets(**kwargs: object) -> list[dict[str, object]]:
        """Return complete deterministic per-sample match rows."""
        loaded_bundle = kwargs["bundle"]
        assert isinstance(loaded_bundle, PredictionBundle)
        rows: list[dict[str, object]] = []
        for row_id, dataset_index in enumerate(loaded_bundle.dataset_indices):
            sample_index = int(loaded_bundle.sample_indices[row_id])
            solved = bool(int(dataset_index) == 10 and sample_index == 0)
            rows.append(
                {
                    "dataset_index": int(dataset_index),
                    "csd_refcode": "AAAAAA" if int(dataset_index) == 10 else "BBBBBB",
                    "sample_index": sample_index,
                    "row_id": row_id,
                    "best_true_refcode": "",
                    "clash": sample_index == 1,
                    "passed": solved,
                    "errors": "",
                    "nmatched_1": 0,
                    "rmsd_1": None,
                    "nmatched_15": 0,
                    "rmsd_15": None,
                    "recovered": solved,
                    "matched": solved,
                },
            )
        return rows

    monkeypatch.setattr("src.evaluate.load_prediction_bundle", lambda path: bundle)
    monkeypatch.setattr("src.evaluate._evaluate_targets", fake_evaluate_targets)
    legacy_eval_dir = tmp_path / "eval"
    legacy_eval_dir.mkdir()
    (legacy_eval_dir / "eval.done").write_text("done\n", encoding="utf-8")

    payload = evaluate_predictions_dir(
        predictions_dir=tmp_path,
        workers=1,
        show_progress=False,
        protocol="clari",
    )

    clari_dir = legacy_eval_dir / "clari"
    assert payload["summary"]["protocol"] == "clari"
    assert payload["summary"]["pool_size"] == samples_per_target
    assert set(payload["summary"]["metrics"]) == {
        "Col_S",
        "Pac_S",
        "Pac_C",
        "Rec_S",
        "Rec_C",
        "Sol_C",
    }
    exact_budget = f"{samples_per_target}/{samples_per_target}"
    assert payload["summary"]["metrics"]["Col_S"][exact_budget][
        "value"
    ] == pytest.approx(1 / samples_per_target)
    assert payload["summary"]["metrics"]["Pac_S"][exact_budget][
        "value"
    ] == pytest.approx(1 / (2 * samples_per_target))
    assert payload["summary"]["metrics"]["Sol_C"][exact_budget]["value"] == 0.5
    replicates = np.load(clari_dir / "bootstrap_replicates.npz", allow_pickle=False)
    assert set(replicates.files) == set(payload["summary"]["metrics"])
    metrics_payload = json.loads((clari_dir / "metrics.json").read_text())
    assert set(metrics_payload["metrics"]) >= {
        "Col_S_30_mean",
        "Col_S_30_standard_error",
        f"Col_S_{samples_per_target}",
        "Pac_S_30_mean",
        "Rec_S_30_mean",
    }
    assert (clari_dir / "correctness.npz").is_file()
    assert (clari_dir / "bootstrap_replicates.npz").is_file()
    assert (clari_dir / "eval.done").is_file()
    assert not (legacy_eval_dir / "summary.json").exists()


def test_aggregate_oxtal_per_target_rows_or_reduces_samples() -> None:
    """Aggregate per-sample OXtal rows into target-level booleans."""
    rows = [
        {
            "dataset_index": 10,
            "sample_index": 0,
            "passed": False,
            "recovered": True,
            "matched": False,
        },
        {
            "dataset_index": 10,
            "sample_index": 1,
            "passed": True,
            "recovered": False,
            "matched": False,
        },
    ]
    context = {
        10: {
            "dataset_index": 10,
            "target_identifier": "AAAAAA",
            "csd_refcode": "AAAAAA",
            "flexibility": "flexible",
        },
    }

    targets = _aggregate_oxtal_per_target_rows(
        rows=rows,
        target_contexts=context,
        target_indices={10: 0},
    )

    assert targets == [
        {
            "target_index": 0,
            "target_identifier": "AAAAAA",
            "csd_refcode": "AAAAAA",
            "dataset_index": 10,
            "num_samples": 2,
            "passed": True,
            "recovered": True,
            "matched": False,
            "flexibility": "flexible",
        },
    ]
