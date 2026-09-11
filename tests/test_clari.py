from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval.clari import (
    build_correctness_artifact,
    evaluate_clari,
    write_correctness_artifact,
    write_replicate_artifact,
)


def _rows(
    clashed: list[list[bool]],
    passed: list[list[bool]],
    recovered: list[list[bool]],
    matched: list[list[bool]],
) -> list[dict[str, object]]:
    """Build aligned evaluated rows for small CLARI matrices."""
    rows: list[dict[str, object]] = []
    for target_position in range(len(passed)):
        for sample_index in range(len(passed[target_position])):
            rows.append(
                {
                    "dataset_index": 10 + target_position,
                    "csd_refcode": f"TARGET{target_position}",
                    "flexibility": "rigid" if target_position == 0 else "flexible",
                    "sample_index": sample_index,
                    "clash": clashed[target_position][sample_index],
                    "passed": passed[target_position][sample_index],
                    "recovered": recovered[target_position][sample_index],
                    "matched": matched[target_position][sample_index],
                },
            )
    return rows


def test_clari_resampling_matches_seeded_reference() -> None:
    """Match a direct seeded implementation for every coverage metric."""
    rows = _rows(
        clashed=[[True, False, False, False], [False, True, False, False]],
        passed=[[True, False, False, False], [False, True, False, False]],
        recovered=[[False, True, True, False], [False, False, False, False]],
        matched=[[False, False, True, False], [True, False, False, True]],
    )
    artifact = build_correctness_artifact(rows)

    result = evaluate_clari(
        artifact,
        draw_size=2,
        bootstrap_replicates=7,
        bootstrap_seed=42,
        replicate_chunk_size=3,
    )

    rng = np.random.default_rng(42)
    draws = rng.integers(0, 4, size=(7, 2, 2), dtype=np.int64)
    target_positions = np.arange(2, dtype=np.int64)[None, :, None]
    sample_metrics = {"Col_S", "Pac_S", "Rec_S"}
    assert set(result.replicates) == {
        "Col_S",
        "Pac_S",
        "Pac_C",
        "Rec_S",
        "Rec_C",
        "Sol_C",
    }
    for metric, matrix in artifact.matrices.items():
        selected = matrix[target_positions, draws]
        expected = (
            selected.mean(axis=(1, 2), dtype=np.float64)
            if metric in sample_metrics
            else selected.any(axis=2).mean(axis=1)
        )
        assert result.replicates[metric] == pytest.approx(expected)
        assert result.summary["metrics"][metric]["2/2"]["mean"] == pytest.approx(
            expected.mean(),
        )
        assert result.summary["metrics"][metric]["2/2"][
            "standard_error"
        ] == pytest.approx(expected.std(ddof=0))

    assert result.summary["metrics"]["Col_S"]["4/4"]["value"] == 0.25
    assert result.summary["metrics"]["Pac_S"]["4/4"]["value"] == 0.25
    assert result.summary["metrics"]["Pac_C"]["4/4"]["value"] == 1.0
    assert result.summary["metrics"]["Rec_S"]["4/4"]["value"] == 0.25
    assert result.summary["metrics"]["Rec_C"]["4/4"]["value"] == 0.5
    assert result.summary["metrics"]["Sol_C"]["4/4"]["value"] == 1.0
    assert "Sol_S" not in result.summary["metrics"]


def test_clari_matrix_rejects_incomplete_and_duplicate_pools() -> None:
    """Reject missing and duplicate target/sample identities."""
    complete_rows = _rows(
        clashed=[[False, False], [False, False]],
        passed=[[True, False], [False, True]],
        recovered=[[True, False], [False, True]],
        matched=[[True, False], [False, True]],
    )

    with pytest.raises(ValueError, match="sample indices"):
        build_correctness_artifact(complete_rows[:-1])

    with pytest.raises(ValueError, match="Duplicate CLARI row identity"):
        build_correctness_artifact(
            [*complete_rows, dict(complete_rows[0])],
        )


def test_clari_artifacts_are_independently_serializable(tmp_path: Path) -> None:
    """Write standalone correctness and replicate artifacts."""
    rows = _rows(
        clashed=[[False, True]],
        passed=[[True, False]],
        recovered=[[False, True]],
        matched=[[True, True]],
    )
    artifact = build_correctness_artifact(rows, expected_pool_size=2)
    result = evaluate_clari(
        artifact,
        draw_size=1,
        bootstrap_replicates=3,
        bootstrap_seed=42,
    )

    correctness_path = tmp_path / "correctness.npz"
    replicates_path = tmp_path / "bootstrap_replicates.npz"
    write_correctness_artifact(correctness_path, artifact)
    write_replicate_artifact(replicates_path, result)

    correctness = np.load(correctness_path, allow_pickle=False)
    replicates = np.load(replicates_path, allow_pickle=False)
    assert {"Col_S", "Pac_S", "Pac_C", "Rec_S", "Rec_C", "Sol_C"} <= set(
        correctness.files,
    )
    assert set(replicates.files) == {
        "Col_S",
        "Pac_S",
        "Pac_C",
        "Rec_S",
        "Rec_C",
        "Sol_C",
    }
    assert correctness["Sol_C"].shape == (1, 2)
    assert replicates["Sol_C"].shape == (3,)
