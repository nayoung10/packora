"""Public Python API for offline Packora structure prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.prediction.api.backend import ModelRegistry, PredictionBackend
from src.prediction.api.chemistry import featurize_request
from src.prediction.api.prior import EmpiricalZPrior
from src.prediction.api.schemas import PredictionRequest
from src.prediction.api.serialization import safe_identifier, serialize_prediction
from src.prediction.api.settings import PredictionSettings


__all__ = [
    "PackoraPredictor",
    "PredictionError",
    "PredictionPaths",
    "PredictionRequest",
    "PredictionResult",
]


class PredictionError(RuntimeError):
    """Raised when local prediction configuration is invalid."""


@dataclass(frozen=True)
class PredictionPaths:
    """Record the CIF and JSON files written for one prediction."""

    cif: Path
    json: Path


@dataclass(frozen=True)
class PredictionResult:
    """Expose one prediction in memory and through optional file writers."""

    request: PredictionRequest
    seed: int
    structure: dict[str, object]
    cif: str
    summary: dict[str, object]

    @property
    def raw_cart_coords(self) -> np.ndarray:
        """Return raw model Cartesian coordinates."""
        return np.asarray(self.structure["raw_cart_coords"], dtype=np.float64)

    @property
    def cart_coords(self) -> np.ndarray:
        """Return centered, molecule-wrapped Cartesian export coordinates."""
        return np.asarray(self.structure["cart_coords"], dtype=np.float64)

    @property
    def frac_coords(self) -> np.ndarray:
        """Return fractional coordinates matching the export coordinates."""
        return np.asarray(self.structure["frac_coords"], dtype=np.float64)

    @property
    def cell(self) -> np.ndarray:
        """Return the generated 3-by-3 unit-cell matrix."""
        return np.asarray(self.structure["cell"], dtype=np.float64)

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Return atomic numbers in prediction order."""
        return np.asarray(self.structure["atomic_numbers"], dtype=np.int64)

    @property
    def membership(self) -> np.ndarray:
        """Return the molecule index for each predicted atom."""
        return np.asarray(self.structure["membership"], dtype=np.int64)

    @property
    def mol2(self) -> str:
        """Return a MOL2 representation for in-memory viewers."""
        return str(self.structure["mol2"])

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation of the prediction."""
        return {
            "request": self.request.model_dump(mode="json"),
            "seed": self.seed,
            "summary": self.summary,
            "structure": self.structure,
        }

    def write(
        self,
        output_dir: str | Path,
        *,
        stem: str = "packora_prediction",
        overwrite: bool = False,
    ) -> PredictionPaths:
        """Write matching CIF and provenance JSON files."""
        directory = Path(output_dir).resolve()
        safe_stem = safe_identifier(stem)
        paths = PredictionPaths(
            cif=directory / f"{safe_stem}.cif",
            json=directory / f"{safe_stem}.json",
        )
        existing = [path for path in (paths.cif, paths.json) if path.exists()]
        if existing and not overwrite:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Refusing to overwrite prediction output: {joined}")
        directory.mkdir(parents=True, exist_ok=True)
        paths.cif.write_text(self.cif, encoding="utf-8")
        paths.json.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return paths


class PackoraPredictor:
    """Load Packora checkpoints lazily and predict one structure per call."""

    def __init__(
        self,
        *,
        device: str | None = None,
        max_atoms: int | None = None,
        num_steps: int | None = None,
        model_manifest_path: str | Path | None = None,
        checkpoint_paths: Mapping[str, str | Path] | None = None,
        z_prior_path: str | Path | None = None,
        backend: PredictionBackend | None = None,
    ) -> None:
        """Resolve local artifacts and initialize a lazy prediction backend."""
        normalized_checkpoints = (
            None
            if checkpoint_paths is None
            else {key: Path(value) for key, value in checkpoint_paths.items()}
        )
        self.settings = PredictionSettings.resolve(
            device=device,
            max_atoms=max_atoms,
            num_steps=num_steps,
            model_manifest_path=(
                None if model_manifest_path is None else Path(model_manifest_path)
            ),
            checkpoint_paths=normalized_checkpoints,
            z_prior_path=None if z_prior_path is None else Path(z_prior_path),
        )
        self._uses_default_backend = backend is None
        self.backend = ModelRegistry(self.settings) if backend is None else backend
        self._prior: EmpiricalZPrior | None = None

    def predict(
        self,
        request: PredictionRequest | Mapping[str, Any],
        *,
        seed: int = 42,
    ) -> PredictionResult:
        """Predict one crystal structure from one validated request."""
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if seed < 0 or seed > 2**32 - 1:
            raise ValueError("seed must be between 0 and 2**32 - 1.")
        validated = PredictionRequest.model_validate(request)
        requires_prior = validated.z is None
        errors = self.settings.validate_for(
            validated.model,
            require_z_prior=requires_prior,
            require_checkpoint=self._uses_default_backend,
            require_cuda=self._uses_default_backend,
        )
        if errors:
            raise PredictionError(" ".join(errors))
        if requires_prior and self._prior is None:
            prior_path = self.settings.z_prior_path
            if prior_path is None:
                raise PredictionError("Empirical Z prior is not configured.")
            self._prior = EmpiricalZPrior(prior_path)
        featurized = featurize_request(
            request=validated,
            prior=self._prior,
            max_atoms=self.settings.max_atoms,
            seed=seed,
        )
        prediction = self.backend.predict(
            model_id=validated.model,
            featurized=featurized,
            seed=seed,
        )
        identifier = f"packora_{validated.model.replace('-', '_')}_seed_{seed}"
        serialized = serialize_prediction(
            identifier=identifier,
            featurized=featurized,
            prediction=prediction,
            model_id=validated.model,
        )
        return PredictionResult(
            request=validated,
            seed=seed,
            structure=serialized.structure,
            cif=serialized.cif,
            summary=serialized.summary,
        )
