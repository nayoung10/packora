import json
from pathlib import Path

import pytest
import torch
from einops import rearrange

from src.models.components.lattice_repr import cell_to_ltri_latent, ltri_latent_to_cell
from src.models.components.scalers.standardizing_scaler import StandardizingScaler


def _canonical_cell() -> torch.Tensor:
    """Build one lower-triangular cell matrix with positive diagonal."""
    return torch.tensor(
        [
            [6.0, 0.0, 0.0],
            [1.2, 7.5, 0.0],
            [-0.4, 2.1, 9.0],
        ],
        dtype=torch.float32,
    )


def _write_stats(data_dir: Path, dataset_name: str, cells: torch.Tensor) -> None:
    """Write a minimal dataset stats sidecar containing ltri statistics."""
    dataset_dir = data_dir / dataset_name
    dataset_dir.mkdir(parents=True)
    ltri = cell_to_ltri_latent(cells)
    stats = {
        "scaler_stats": {
            "coord_std": 1.0,
            "length_mean": [1.0, 1.0, 1.0],
            "length_std": [1.0, 1.0, 1.0],
            "angle_mean": [90.0, 90.0, 90.0],
            "angle_std": [1.0, 1.0, 1.0],
            "cell_mean": [0.0] * 9,
            "cell_std": [1.0] * 9,
            "ltri_mean": ltri.mean(dim=0).tolist(),
            "ltri_std": ltri.std(dim=0, correction=0).clamp(min=0.1).tolist(),
        }
    }
    with open(dataset_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f)


def test_ltri_round_trip_restores_canonical_cell() -> None:
    """Round-trip a canonical lower-triangular cell through ltri latent space."""
    cell = _canonical_cell()[None]
    latent = cell_to_ltri_latent(cell)
    decoded = ltri_latent_to_cell(latent)

    expected = rearrange(cell, "b i j -> b (i j)")
    assert torch.allclose(decoded, expected, atol=1e-5)


def test_ltri_round_trip_preserves_rotated_cell_gram() -> None:
    """Decode a rotated cell to the same lattice metric in canonical orientation."""
    cell = _canonical_cell()[None]
    theta = torch.tensor(0.4)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    zero = torch.zeros((), dtype=torch.float32)
    one = torch.ones((), dtype=torch.float32)
    rotation = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta, zero]),
            torch.stack([sin_theta, cos_theta, zero]),
            torch.stack([zero, zero, one]),
        ],
    )
    rotated = cell @ rearrange(rotation, "i j -> j i")
    decoded = ltri_latent_to_cell(cell_to_ltri_latent(rotated))
    decoded_cell = rearrange(decoded, "b (i j) -> b i j", i=3, j=3)

    rotated_gram = rotated @ rearrange(rotated, "b i j -> b j i")
    decoded_gram = decoded_cell @ rearrange(decoded_cell, "b i j -> b j i")
    assert torch.allclose(decoded_gram, rotated_gram, atol=1e-5)


def test_standardizing_ltri_pipeline_restores_original_cell(tmp_path: Path) -> None:
    """Run cell to ltri to z-score to inverse z-score to decoded cell end to end."""
    base = _canonical_cell()
    cells = torch.stack([base, base * 1.1 + torch.diag(torch.tensor([0.2, 0.3, 0.4]))])
    _write_stats(tmp_path, "toy", cells)

    scaler = StandardizingScaler(
        datasets=[{"data_dir": str(tmp_path), "dataset_name": "toy"}],
        lattice_repr="ltri",
    )
    model_units = scaler.scale_lattice(cells[:1])
    restored = scaler.unscale_lattice(model_units)
    expected = rearrange(cells[:1], "b i j -> b (i j)")

    assert torch.allclose(restored, expected, atol=1e-5)


def test_standardizing_ltri_requires_stats(tmp_path: Path) -> None:
    """Raise a clear error when ltri stats are missing."""
    dataset_dir = tmp_path / "toy"
    dataset_dir.mkdir()
    stats = {
        "scaler_stats": {
            "coord_std": 1.0,
            "length_mean": [1.0, 1.0, 1.0],
            "length_std": [1.0, 1.0, 1.0],
            "angle_mean": [90.0, 90.0, 90.0],
            "angle_std": [1.0, 1.0, 1.0],
        }
    }
    with open(dataset_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f)

    with pytest.raises(ValueError, match="ltri_mean"):
        StandardizingScaler(
            datasets=[{"data_dir": str(tmp_path), "dataset_name": "toy"}],
            lattice_repr="ltri",
        )
