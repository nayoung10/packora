"""Empirical unit-cell multiplicity sampling for molecular prediction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ZDraw:
    """Record one atom-cap-conditioned empirical multiplicity draw."""

    value: int
    probability: float
    unconstrained_probability: float
    excluded_probability_mass: float
    prior_sha256: str


class EmpiricalZPrior:
    """Load and sample the persisted empirical CSD multiplicity prior."""

    def __init__(self, path: Path) -> None:
        """Load and validate one empirical prior JSON artifact."""
        self.path = path.resolve()
        payload = _read_json(self.path)
        categorical = payload.get("categorical")
        if not isinstance(categorical, dict):
            raise ValueError("Empirical Z prior has no categorical mapping.")
        self.values = np.asarray(categorical.get("values"), dtype=np.int64)
        self.probabilities = np.asarray(
            categorical.get("probabilities"),
            dtype=np.float64,
        )
        if (
            self.values.ndim != 1
            or not self.values.size
            or self.probabilities.shape != self.values.shape
        ):
            raise ValueError("Empirical Z values and probabilities are misaligned.")
        if np.any(self.values < 1) or np.any(self.probabilities < 0):
            raise ValueError("Empirical Z prior contains invalid values.")
        if not np.isclose(self.probabilities.sum(), 1.0):
            raise ValueError("Empirical Z probabilities do not sum to one.")
        self.sha256 = _sha256_path(self.path)

    def draw(self, seed: int, atoms_per_formula: int, max_atoms: int) -> ZDraw:
        """Draw a deterministic Z after conditioning on the atom cap."""
        if atoms_per_formula < 1 or max_atoms < 1:
            raise ValueError("Atom counts must be positive.")
        valid = self.values <= max_atoms // atoms_per_formula
        valid_mass = self.probabilities[valid].sum(dtype=np.float64)
        if not np.isfinite(valid_mass) or valid_mass <= 0:
            raise ValueError(
                f"No empirical Z value fits {atoms_per_formula} formula-unit "
                f"atoms under the {max_atoms}-atom cap."
            )
        conditioned = np.where(valid, self.probabilities / valid_mass, 0.0)
        generator = np.random.default_rng(int(seed))
        value = int(generator.choice(self.values, p=conditioned))
        index = int(np.flatnonzero(self.values == value)[0])
        return ZDraw(
            value=value,
            probability=float(conditioned[index]),
            unconstrained_probability=float(self.probabilities[index]),
            excluded_probability_mass=float(self.probabilities[~valid].sum()),
            prior_sha256=self.sha256,
        )


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}.")
    return payload


def _sha256_path(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
