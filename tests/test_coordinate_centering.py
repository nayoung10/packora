"""Tests for configurable Cartesian coordinate centering."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.extract_dataset_stats import coordinate_statistics
from src.data import datamodule as datamodule_module
from src.data.components.prior.cart_coords.centered_gaussian import centered_gaussian
from src.models.components.heads.coord.linear import LinearCoordsHead
from src.models.components.scalers.standardizing_scaler import StandardizingScaler
from src.models.flow_model import MaterialFlowMatching


def _write_stats(data_dir: Path, include_uncentered: bool = True) -> None:
    """Write minimal coordinate and lattice statistics for scaler tests."""
    dataset_dir = data_dir / "toy"
    dataset_dir.mkdir()
    scaler_stats = {
        "coord_mean": [0.0, 0.0, 0.0],
        "coord_std": 2.0,
        "length_mean": [1.0, 1.0, 1.0],
        "length_std": [1.0, 1.0, 1.0],
        "angle_mean": [90.0, 90.0, 90.0],
        "angle_std": [1.0, 1.0, 1.0],
    }
    if include_uncentered:
        scaler_stats.update(
            {
                "coord_mean_uncentered": [1.0, 2.0, 3.0],
                "coord_std_uncentered": 4.0,
            }
        )
    (dataset_dir / "dataset_stats.json").write_text(
        json.dumps({"scaler_stats": scaler_stats}), encoding="utf-8"
    )


def test_coordinate_statistics_compute_both_profiles() -> None:
    """Compute centered and uncentered statistics over every atom component."""
    coords = [
        np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
        np.array([[5.0, 8.0, 11.0]]),
    ]

    stats = coordinate_statistics(coords)
    all_coords = np.concatenate(coords, axis=0)
    mean = all_coords.mean(axis=0)
    centered = np.concatenate([value - value.mean(axis=0) for value in coords])

    assert stats["coord_mean"] == [0.0, 0.0, 0.0]
    assert np.allclose(stats["coord_mean_uncentered"], mean)
    assert stats["coord_std"] == pytest.approx(centered.std())
    assert stats["coord_std_uncentered"] == pytest.approx((all_coords - mean).std())


def test_datamodule_test_and_predict_stages_propagate_centering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the shared centering mode to standalone test and predict datasets."""
    dataset_kwargs = []

    def fake_dataset(**kwargs: object) -> object:
        """Record one dataset construction without opening LMDB."""
        dataset_kwargs.append(kwargs)
        return object()

    monkeypatch.setattr(datamodule_module, "MaterialDataset", fake_dataset)
    data = datamodule_module.MaterialDataModule(
        data_dir="/tmp/data",
        dataset_name="toy",
        batch_size=1,
        center_cart_coords=False,
    )

    data.setup("test")
    data.setup("predict")

    assert [kwargs["center_cart_coords"] for kwargs in dataset_kwargs] == [False, False]


def test_uncentered_scaler_affine_round_trip_keeps_padding_zero(
    tmp_path: Path,
) -> None:
    """Apply one generic affine transform while excluding padded atoms."""
    _write_stats(tmp_path)
    scaler = StandardizingScaler(
        datasets=[{"data_dir": str(tmp_path), "dataset_name": "toy"}],
        center_cart_coords=False,
    )
    coords = torch.tensor([[[5.0, 6.0, 7.0], [9.0, 10.0, 11.0], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])

    scaled = scaler.scale_coords(coords, mask)
    restored = scaler.unscale_coords(scaled, mask)

    assert torch.allclose(scaled[0, 0], torch.ones(3))
    assert torch.equal(scaled[0, 2], torch.zeros(3))
    assert torch.allclose(restored[:, :2], coords[:, :2])
    assert torch.equal(restored[0, 2], torch.zeros(3))
    assert "coord_mean" not in scaler.state_dict()


def test_centered_scaler_defaults_legacy_mean_to_zero(tmp_path: Path) -> None:
    """Load legacy centered statistics without a stored coordinate mean."""
    _write_stats(tmp_path, include_uncentered=False)
    stats_path = tmp_path / "toy" / "dataset_stats.json"
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    payload["scaler_stats"].pop("coord_mean")
    stats_path.write_text(json.dumps(payload), encoding="utf-8")

    scaler = StandardizingScaler(
        datasets=[{"data_dir": str(tmp_path), "dataset_name": "toy"}],
        center_cart_coords=True,
    )

    assert torch.equal(scaler.coord_mean, torch.zeros(3))
    assert torch.allclose(
        scaler.scale_coords(torch.tensor([[[2.0, 4.0, 6.0]]])),
        torch.tensor([[[1.0, 2.0, 3.0]]]),
    )


def test_uncentered_scaler_requires_regenerated_statistics(tmp_path: Path) -> None:
    """Reject old statistics when uncentered normalization is requested."""
    _write_stats(tmp_path, include_uncentered=False)

    with pytest.raises(ValueError, match="extract_dataset_stats.py"):
        StandardizingScaler(
            datasets=[{"data_dir": str(tmp_path), "dataset_name": "toy"}],
            center_cart_coords=False,
        )


def test_gaussian_prior_respects_centering_mode() -> None:
    """Center the Gaussian prior only when the coordinate gauge requires it."""
    coords = torch.zeros(2, 4, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])

    torch.manual_seed(7)
    centered = centered_gaussian(coords, mask, center=True)
    torch.manual_seed(7)
    uncentered = centered_gaussian(coords, mask, center=False)

    assert torch.allclose(centered[0, :2].mean(dim=0), torch.zeros(3), atol=1e-6)
    assert not torch.allclose(uncentered[0, :2].mean(dim=0), torch.zeros(3))
    assert torch.equal(centered[~mask.bool()], torch.zeros_like(centered[~mask.bool()]))
    assert torch.equal(
        uncentered[~mask.bool()], torch.zeros_like(uncentered[~mask.bool()])
    )


def test_coordinate_head_respects_centering_mode() -> None:
    """Project head outputs to zero centroid only in centered mode."""
    features = torch.randn(1, 3, 4)
    mask = torch.tensor([[True, True, False]])
    centered = LinearCoordsHead(4, center=True, zero_init=False)
    uncentered = LinearCoordsHead(4, center=False, zero_init=False)
    uncentered.load_state_dict(centered.state_dict())

    centered_output = centered(features, mask)
    uncentered_output = uncentered(features, mask)

    assert torch.allclose(centered_output[0, :2].mean(dim=0), torch.zeros(3), atol=1e-6)
    assert not torch.allclose(
        uncentered_output[0, :2].mean(dim=0), torch.zeros(3), atol=1e-6
    )


@pytest.mark.parametrize("center", [True, False])
def test_sample_constraint_always_masks_and_optionally_centers(center: bool) -> None:
    """Keep padding zero without imposing a centroid in uncentered mode."""
    model = MaterialFlowMatching.__new__(MaterialFlowMatching)
    nn.Module.__init__(model)
    model.center_cart_coords = center
    coords = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [9.0, 9.0, 9.0]]])
    mask = torch.tensor([[True, True, False]])

    constrained = model._constrain_sample_coords(coords, mask)

    assert torch.equal(constrained[0, 2], torch.zeros(3))
    expected_mean = torch.zeros(3) if center else torch.tensor([2.0, 3.0, 4.0])
    assert torch.allclose(constrained[0, :2].mean(dim=0), expected_mean)
