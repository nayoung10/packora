from pathlib import Path

import numpy as np
import pytest
import torch
from einops import rearrange

from eval.spacegroup import evaluate_bundle, write_results
from src.prediction.io import PredictionBundle


def _spacegroup_bundle() -> PredictionBundle:
    """Return two generated samples for one cubic target crystal."""
    cells = torch.stack(
        [
            torch.eye(3),
            torch.diag(torch.tensor([1.0, 1.0, 2.0])),
        ]
    )
    pred = {
        "cart_coords": torch.zeros(2, 1, 3),
        "lattice": rearrange(cells, "b i j -> b (i j)"),
        "atomic_numbers": torch.full((2, 1), 14, dtype=torch.long),
        "atom_mask": torch.ones(2, 1, dtype=torch.bool),
    }
    return PredictionBundle(
        pred=pred,
        ref=dict(pred),
        dataset_indices=np.array([5, 5], dtype=np.int64),
        sample_indices=np.array([0, 1], dtype=np.int64),
        metadata={"spacegroup_number": [221, 221]},
    )


def test_evaluate_bundle_reports_per_sample_and_per_target_matches() -> None:
    """Count all generated rows and any successful sample for each target."""
    rows, summary = evaluate_bundle(_spacegroup_bundle(), show_progress=False)

    assert [row["detected_spacegroup"] for row in rows] == [221, 123]
    assert [row["match"] for row in rows] == [True, False]
    assert summary["per_sample"]["detection_failures"] == 0
    assert summary["per_sample"]["match_percent"] == pytest.approx(50.0)
    assert summary["per_target"]["total"] == 1
    assert summary["per_target"]["any_match_percent"] == pytest.approx(100.0)


def test_evaluate_bundle_rejects_missing_target_spacegroup() -> None:
    """Require a valid metadata target for every generated row."""
    bundle = _spacegroup_bundle()
    bundle.metadata["spacegroup_number"] = [None, None]

    with pytest.raises(ValueError, match="no valid target space group"):
        evaluate_bundle(bundle, show_progress=False)


def test_write_results_creates_json_artifacts(tmp_path: Path) -> None:
    """Write stable per-sample and aggregate output filenames."""
    rows, summary = evaluate_bundle(_spacegroup_bundle(), show_progress=False)

    write_results(tmp_path, rows, summary)

    assert (tmp_path / "spacegroup_per_sample.jsonl").is_file()
    assert (tmp_path / "spacegroup_summary.json").is_file()
