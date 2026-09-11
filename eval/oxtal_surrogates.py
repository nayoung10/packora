"""StructureMatcher and clash-rate surrogate CSP metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from ase.data import covalent_radii
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure


def _mean_bool(values: Sequence[bool]) -> float:
    """Return the arithmetic mean of boolean indicator values."""
    if not values:
        raise ValueError("Cannot average an empty indicator sequence.")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _mean_float(values: Sequence[float]) -> float:
    """Return the arithmetic mean of scalar values."""
    if not values:
        raise ValueError("Cannot average an empty scalar sequence.")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _covalent_radii_lookup(atomic_numbers: np.ndarray) -> np.ndarray:
    """Lookup covalent radii for atomic numbers with a safe fallback value."""
    max_z = len(covalent_radii) - 1
    fallback_radius = 1.5
    radii = np.full(atomic_numbers.shape, fallback_radius, dtype=np.float64)
    for index, atomic_number in enumerate(atomic_numbers.tolist()):
        if 0 < int(atomic_number) <= max_z:
            radius = float(covalent_radii[int(atomic_number)])
            if np.isfinite(radius) and radius > 0.0:
                radii[index] = radius
    return radii


def clash_rate(sample: Structure, alpha: float = 0.75) -> float:
    """Return the fraction of atoms participating in at least one short contact."""
    if len(sample) <= 1:
        return 0.0

    atomic_numbers = np.asarray(
        [int(site.specie.Z) for site in sample.sites],
        dtype=np.int64,
    )
    radii = _covalent_radii_lookup(atomic_numbers)

    dist_matrix = np.array(sample.distance_matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(dist_matrix, np.inf)

    # clash_rate(x_{c,s}): fraction of atoms with any alpha-scaled covalent clash
    threshold_matrix = alpha * (radii[:, None] + radii[None, :])
    clash_pairs = dist_matrix < threshold_matrix
    clashing_atoms = np.any(clash_pairs, axis=1)
    return float(np.mean(clashing_atoms))


def _target_identifier(target: Structure, target_index: int) -> str:
    """Return a stable identifier for one individual target crystal entry."""
    properties = getattr(target, "properties", {})
    for key in ("target_identifier", "identifier", "material_id", "csd_identifier"):
        value = properties.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(target_index)


def _validate_inputs(
    target_structures: Sequence[Structure],
    samples_by_target: Sequence[Sequence[Structure]],
) -> None:
    """Validate the grouped target/sample evaluation shape."""
    if not target_structures:
        raise ValueError("target_structures must contain at least one individual target entry.")
    if len(target_structures) != len(samples_by_target):
        raise ValueError("samples_by_target must have one sample sequence per target structure.")
    for target_index, samples in enumerate(samples_by_target):
        if not samples:
            raise ValueError(f"samples_by_target[{target_index}] must contain at least one sample.")


def make_structure_matcher(
    stol: float = 0.5,
    ltol: float = 0.3,
    angle_tol: float = 10.0,
) -> StructureMatcher:
    """Build a StructureMatcher configured for surrogate CSP matching."""
    return StructureMatcher(
        stol=stol,
        ltol=ltol,
        angle_tol=angle_tol,
        primitive_cell=False,
    )


def structure_match(
    sample: Structure,
    target: Structure,
    matcher: StructureMatcher,
) -> bool:
    """Return whether one generated sample matches its target structure."""
    try:
        return bool(matcher.fit(sample, target))
    except Exception:
        return False


def evaluate(
    target_structures: Sequence[Structure],
    samples_by_target: Sequence[Sequence[Structure]],
    alpha: float = 0.75,
    matcher: StructureMatcher | None = None,
) -> dict[str, object]:
    """Evaluate StructureMatcher surrogate metrics grouped by target crystal."""
    _validate_inputs(target_structures, samples_by_target)
    structure_matcher = matcher if matcher is not None else make_structure_matcher()

    per_sample: list[dict[str, object]] = []
    per_target: list[dict[str, object]] = []

    for target_index, target in enumerate(target_structures):
        target_identifier = _target_identifier(target, target_index)
        samples = samples_by_target[target_index]

        match_values: list[bool] = []
        sol_values: list[bool] = []

        for sample_index, sample in enumerate(samples):
            # match_{c,s}: StructureMatcher(x_{c,s}, x_c*) fit is True
            matched = structure_match(sample, target, matcher=structure_matcher)
            clash = clash_rate(sample, alpha=alpha)
            # col_{c,s}: cheap clash surrogate is positive
            collision = clash > 0.0
            # sol_{c,s}: collision-free and StructureMatcher-matched
            solved = not collision and matched

            match_values.append(matched)
            sol_values.append(solved)

            per_sample.append(
                {
                    "target_index": target_index,
                    "target_identifier": target_identifier,
                    "sample_index": sample_index,
                    "structure_match": bool(matched),
                    "clash_rate": float(clash),
                    "collision": bool(collision),
                    "surrogate_solved": bool(solved),
                },
            )

        # Crystal-level metrics are OR/max over samples s for this individual target c
        per_target.append(
            {
                "target_index": target_index,
                "target_identifier": target_identifier,
                "num_samples": int(len(samples)),
                "structure_match": bool(any(match_values)),
                "surrogate_solved": bool(any(sol_values)),
            },
        )

    # Sample-level metrics average over every generated candidate x_{c,s}
    summary = {
        "SMatch_S": _mean_bool([bool(row["structure_match"]) for row in per_sample]),
        "SMatch_C": _mean_bool([bool(row["structure_match"]) for row in per_target]),
        "SClash_S": _mean_float([float(row["clash_rate"]) for row in per_sample]),
        "SSol_C": _mean_bool([bool(row["surrogate_solved"]) for row in per_target]),
    }

    return {
        "summary": summary,
        "per_sample": per_sample,
        "per_target": per_target,
    }
