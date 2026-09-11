import pytest

from eval.oxtal_pass_at_k import (
    build_target_stats,
    pass_at_k_records,
    stable_pass_at_k,
)


def _rows_for_target(
    target_id: int,
    refcode: str,
    flexibility: str,
    passed_values: list[bool],
    recovered_values: list[bool] | None = None,
    matched_values: list[bool] | None = None,
) -> list[dict[str, object]]:
    """Build synthetic evaluated OXtal rows for one target."""
    recovered = passed_values if recovered_values is None else recovered_values
    matched = passed_values if matched_values is None else matched_values
    return [
        {
            "dataset_index": target_id,
            "csd_refcode": refcode,
            "flexibility": flexibility,
            "sample_index": sample_index,
            "passed": bool(passed),
            "recovered": bool(recovered[sample_index]),
            "matched": bool(matched[sample_index]),
        }
        for sample_index, passed in enumerate(passed_values)
    ]


def _record_value(records: list[dict[str, object]], group: str, metric: str, k: int) -> float:
    """Return one value from pass@k records."""
    matches = [
        float(row["value"])
        for row in records
        if row["group"] == group and row["metric"] == metric and row["k"] == k
    ]
    assert len(matches) == 1
    return matches[0]


def test_stable_pass_at_k_handles_edge_correct_counts() -> None:
    """Compute exact pass@k values for impossible and guaranteed targets."""
    assert stable_pass_at_k(num_samples=4, correct_count=0, k=1) == pytest.approx(0.0)
    assert stable_pass_at_k(num_samples=4, correct_count=0, k=4) == pytest.approx(0.0)
    assert stable_pass_at_k(num_samples=4, correct_count=4, k=1) == pytest.approx(1.0)
    assert stable_pass_at_k(num_samples=4, correct_count=4, k=4) == pytest.approx(1.0)


def test_pass_at_k_records_reuse_target_counts_by_group() -> None:
    """Aggregate all and split pass@k records from cached target counts."""
    rows = [
        *_rows_for_target(1, "AAAAAA", "rigid", [True, False, False, True]),
        *_rows_for_target(2, "BBBBBB", "rigid", [False, False, False, False]),
        *_rows_for_target(3, "CCCCCC", "flexible", [True, True, True, False]),
        *_rows_for_target(4, "DDDDDD", "flexible", [False, True, False, False]),
    ]

    records = pass_at_k_records(build_target_stats(rows))

    assert _record_value(records, "all", "Pac_C", 1) == pytest.approx(0.375)
    assert _record_value(records, "all", "Pac_C", 2) == pytest.approx(0.5833333333)
    assert _record_value(records, "all", "Pac_C", 4) == pytest.approx(0.75)
    assert _record_value(records, "rigid", "Pac_C", 2) == pytest.approx(0.4166666667)
    assert _record_value(records, "flexible", "Pac_C", 2) == pytest.approx(0.75)


def test_pass_at_k_records_keeps_metric_correctness_separate() -> None:
    """Use separate per-sample boolean fields for Pac, Rec, and Sol."""
    rows = _rows_for_target(
        1,
        "AAAAAA",
        "rigid",
        passed_values=[True, True, False, False],
        recovered_values=[True, False, False, False],
        matched_values=[False, False, False, False],
    )

    records = pass_at_k_records(build_target_stats(rows))

    assert _record_value(records, "all", "Pac_C", 4) == pytest.approx(1.0)
    assert _record_value(records, "all", "Rec_C", 4) == pytest.approx(1.0)
    assert _record_value(records, "all", "Sol_C", 4) == pytest.approx(0.0)
    assert _record_value(records, "all", "Pac_C", 1) == pytest.approx(0.5)
    assert _record_value(records, "all", "Rec_C", 1) == pytest.approx(0.25)


def test_build_target_stats_requires_uniform_sample_count() -> None:
    """Reject non-uniform sample counts before computing pass@k."""
    rows = [
        *_rows_for_target(1, "AAAAAA", "rigid", [True, False]),
        *_rows_for_target(2, "BBBBBB", "flexible", [True]),
    ]

    with pytest.raises(ValueError, match="uniform sample count"):
        build_target_stats(rows)
