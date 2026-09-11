"""Tests for the offline Packora prediction API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.prediction.api import PackoraPredictor, PredictionError
from src.prediction.api.backend import PredictionArrays
from src.prediction.api.chemistry import FeaturizedInput
from src.prediction.api.settings import PredictionSettings


class FakeBackend:
    """Return deterministic arrays without loading a real checkpoint."""

    def predict(
        self,
        model_id: str,
        featurized: FeaturizedInput,
        seed: int,
    ) -> PredictionArrays:
        """Build one finite cubic prediction in conditioning atom order."""
        count = featurized.total_atom_count
        coords = np.zeros((count, 3), dtype=np.float64)
        coords[:, 0] = np.arange(count, dtype=np.float64) * 0.1
        return PredictionArrays(
            cart_coords=coords,
            cell=np.eye(3, dtype=np.float64) * 12.0,
            atomic_numbers=featurized.conditioning.atomic_numbers.astype(np.int64),
            elapsed_seconds=0.01,
            checkpoint_id=f"fake-{model_id}",
            model_label="Packora-M" if model_id == "packora-m" else "Packora-L",
        )


def _manifest(tmp_path: Path) -> Path:
    """Write one minimal two-model manifest."""
    path = tmp_path / "inputs.json"
    path.write_text(
        json.dumps(
            {
                "methods": {
                    "medium": {
                        "checkpoint_id": "medium-id",
                        "checkpoint_path": "medium.ckpt",
                    },
                    "large": {
                        "checkpoint_id": "large-id",
                        "checkpoint_path": "large.ckpt",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _prior(tmp_path: Path) -> Path:
    """Write one deterministic empirical Z prior."""
    path = tmp_path / "z_distribution.json"
    path.write_text(
        json.dumps(
            {
                "categorical": {
                    "values": [1, 4],
                    "probabilities": [0.0, 1.0],
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_explicit_z_skips_prior_and_writes_prediction(tmp_path: Path) -> None:
    """Explicit Z should not require a prior and should produce paired outputs."""
    predictor = PackoraPredictor(
        checkpoint_paths={"packora-m": tmp_path / "packora-m.ckpt"},
        backend=FakeBackend(),
    )
    result = predictor.predict(
        {
            "model": "packora-m",
            "components": [{"smiles": "N#Cc1ccc(cc1)C#N", "ratio": 1}],
            "z": 1,
        }
    )

    assert result.seed == 42
    assert result.summary["z"] == 1
    assert result.summary["z_source"] == "explicit"
    assert result.atomic_numbers.shape == (14,)
    assert result.raw_cart_coords.shape == (14, 3)
    assert result.cart_coords.shape == (14, 3)
    assert result.frac_coords.shape == (14, 3)
    assert np.allclose(
        result.frac_coords, result.cart_coords @ np.linalg.inv(result.cell)
    )
    assert result.cell.shape == (3, 3)
    assert "_cell_length_a" in result.cif
    assert "@<TRIPOS>MOLECULE" in result.mol2

    paths = result.write(tmp_path / "outputs", stem="tepnit_seed_42")
    assert paths.cif.is_file()
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert payload["request"]["z"] == 1
    assert payload["summary"]["checkpoint_id"] == "fake-packora-m"
    assert len(payload["structure"]["frac_coords"]) == 14
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        result.write(tmp_path / "outputs", stem="tepnit_seed_42")


def test_omitted_z_draws_conditioned_prior_from_seed(tmp_path: Path) -> None:
    """Omitted Z should draw reproducibly from the configured empirical prior."""
    predictor = PackoraPredictor(
        model_manifest_path=_manifest(tmp_path),
        z_prior_path=_prior(tmp_path),
        backend=FakeBackend(),
    )
    request = {
        "model": "packora-m",
        "components": [{"smiles": "N#Cc1ccc(cc1)C#N", "ratio": 1}],
    }

    first = predictor.predict(request, seed=42)
    second = predictor.predict(request, seed=42)

    assert first.summary["z"] == 4
    assert first.summary["z_source"] == "empirical_prior"
    assert first.summary["total_atom_count"] == 56
    assert first.summary["z_prior"] == second.summary["z_prior"]


def test_checkpoint_precedence_is_explicit_env_then_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint arguments should override environment and manifest paths."""
    manifest = _manifest(tmp_path)
    environment = tmp_path / "environment.ckpt"
    explicit = tmp_path / "explicit.ckpt"
    monkeypatch.setenv("PACKORA_M_CHECKPOINT", str(environment))

    settings = PredictionSettings.resolve(
        model_manifest_path=manifest,
        checkpoint_paths={"packora-m": explicit},
    )

    assert settings.models["packora-m"].checkpoint_path == explicit.resolve()
    assert (
        settings.models["packora-l"].checkpoint_path
        == (tmp_path / "large.ckpt").resolve()
    )


def test_direct_checkpoint_paths_do_not_require_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit checkpoints should work without a model manifest."""
    monkeypatch.delenv("PACKORA_MODEL_MANIFEST", raising=False)
    monkeypatch.delenv("PACKORA_M_CHECKPOINT", raising=False)
    monkeypatch.delenv("PACKORA_L_CHECKPOINT", raising=False)
    checkpoint = tmp_path / "packora-m.ckpt"

    settings = PredictionSettings.resolve(checkpoint_paths={"packora-m": checkpoint})

    assert settings.model_manifest_path is None
    assert settings.models["packora-m"].checkpoint_path == checkpoint.resolve()
    assert settings.models["packora-m"].checkpoint_id == "packora-m"
    assert "packora-l" not in settings.models


def test_missing_manifest_and_checkpoints_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing direct and manifest configuration should fail clearly."""
    monkeypatch.delenv("PACKORA_MODEL_MANIFEST", raising=False)
    monkeypatch.delenv("PACKORA_M_CHECKPOINT", raising=False)
    monkeypatch.delenv("PACKORA_L_CHECKPOINT", raising=False)

    with pytest.raises(ValueError, match="No checkpoints are configured"):
        PredictionSettings.resolve()


def test_artifact_paths_resolve_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifest and prior environment variables should replace removed defaults."""
    manifest = _manifest(tmp_path)
    prior = _prior(tmp_path)
    monkeypatch.setenv("PACKORA_MODEL_MANIFEST", str(manifest))
    monkeypatch.setenv("PACKORA_Z_PRIOR", str(prior))

    settings = PredictionSettings.resolve()

    assert settings.model_manifest_path == manifest.resolve()
    assert settings.z_prior_path == prior.resolve()


def test_omitted_z_requires_configured_prior(tmp_path: Path) -> None:
    """Omitted Z should explain how to configure the empirical prior."""
    predictor = PackoraPredictor(
        model_manifest_path=_manifest(tmp_path),
        backend=FakeBackend(),
    )

    with pytest.raises(PredictionError, match="Pass z_prior_path"):
        predictor.predict(
            {
                "model": "packora-m",
                "components": [{"smiles": "N#N", "ratio": 1}],
            }
        )
