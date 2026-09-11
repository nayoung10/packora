"""Evaluate generated crystal space groups with spglib."""

# ruff: noqa: F722,F821

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import spglib
from tqdm import tqdm

from src.prediction.io import PredictionBundle, load_prediction_bundle
from src.utils.structure_io import tensors_to_atoms
from src.utils.tensor_typing import Bool, Float, Int


def _metadata_value(
    bundle: PredictionBundle,
    field: str,
    row_index: int,
) -> Any | None:
    """Return one optional metadata value from a bundle row."""
    values = bundle.metadata.get(field)
    return None if values is None else values[row_index]


def target_spacegroup(bundle: PredictionBundle, row_index: int) -> int:
    """Return the valid target space group stored for one prediction row."""
    value = _metadata_value(bundle, "spacegroup_number", row_index)
    if value is None:
        value = _metadata_value(bundle, "spacegroup", row_index)
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Prediction row {row_index} has no valid target space group."
        ) from error
    if number < 1 or number > 230:
        raise ValueError(
            f"Prediction row {row_index} target space group must be in [1, 230], "
            f"got {number}."
        )
    return number


def detect_spacegroup(
    coords: Float["n 3"],
    lattice: Float["d"],
    atomic_numbers: Int["n"],
    atom_mask: Bool["n"],
    symprec: float,
    angle_tolerance: float,
) -> tuple[int, str]:
    """Detect one generated structure's space group with spglib."""
    atoms = tensors_to_atoms(coords, lattice, atomic_numbers, atom_mask)
    cell = (
        atoms.cell.array,
        atoms.get_scaled_positions(wrap=True),
        atoms.numbers,
    )
    dataset = spglib.get_symmetry_dataset(
        cell,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if dataset is None:
        raise ValueError("spglib could not determine a space group.")
    return int(dataset.number), str(dataset.international)


def evaluate_bundle(
    bundle: PredictionBundle,
    symprec: float = 0.1,
    angle_tolerance: float = -1.0,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate every generated row and summarize per-sample and per-target matches."""
    if symprec <= 0.0:
        raise ValueError("symprec must be > 0.")
    required = {"cart_coords", "lattice", "atomic_numbers", "atom_mask"}
    missing = required.difference(bundle.pred)
    if missing:
        raise KeyError(
            f"Prediction tensors are missing required keys: {sorted(missing)}"
        )

    total_samples = int(bundle.dataset_indices.shape[0])
    for key in required:
        if int(bundle.pred[key].shape[0]) != total_samples:
            raise ValueError(f"Prediction tensor '{key}' has the wrong row count.")

    rows: list[dict[str, Any]] = []
    target_by_dataset_index: dict[int, int] = {}
    any_match_by_dataset_index: dict[int, bool] = {}
    detected_samples = 0
    matched_samples = 0

    iterator = tqdm(
        range(total_samples),
        desc="Detecting space groups",
        disable=not show_progress,
    )
    for row_index in iterator:
        dataset_index = int(bundle.dataset_indices[row_index])
        sample_index = int(bundle.sample_indices[row_index])
        target = target_spacegroup(bundle, row_index)
        previous_target = target_by_dataset_index.setdefault(dataset_index, target)
        if previous_target != target:
            raise ValueError(
                f"Dataset index {dataset_index} has inconsistent target space groups."
            )

        detected_number: int | None = None
        detected_symbol: str | None = None
        error_message: str | None = None
        try:
            detected_number, detected_symbol = detect_spacegroup(
                coords=bundle.pred["cart_coords"][row_index],
                lattice=bundle.pred["lattice"][row_index],
                atomic_numbers=bundle.pred["atomic_numbers"][row_index],
                atom_mask=bundle.pred["atom_mask"][row_index],
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )
        except Exception as error:  # spglib and ASE expose several failure types
            error_message = f"{type(error).__name__}: {error}"

        detected = detected_number is not None
        matched = detected_number == target
        detected_samples += int(detected)
        matched_samples += int(matched)
        any_match_by_dataset_index[dataset_index] = (
            any_match_by_dataset_index.get(dataset_index, False) or matched
        )
        rows.append(
            {
                "row_index": row_index,
                "dataset_index": dataset_index,
                "sample_index": sample_index,
                "target_spacegroup": target,
                "detected_spacegroup": detected_number,
                "detected_symbol": detected_symbol,
                "detected": detected,
                "match": matched,
                "error": error_message,
            }
        )

    detection_failures = total_samples - detected_samples
    total_targets = len(target_by_dataset_index)
    targets_with_any_match = sum(any_match_by_dataset_index.values())
    match_rate = matched_samples / total_samples if total_samples else 0.0
    detection_rate = detected_samples / total_samples if total_samples else 0.0
    detected_match_rate = (
        matched_samples / detected_samples if detected_samples else 0.0
    )
    target_any_match_rate = (
        targets_with_any_match / total_targets if total_targets else 0.0
    )
    summary = {
        "symprec": symprec,
        "angle_tolerance": angle_tolerance,
        "per_sample": {
            "total": total_samples,
            "detected": detected_samples,
            "detection_failures": detection_failures,
            "detection_rate": detection_rate,
            "detection_percent": 100.0 * detection_rate,
            "matches": matched_samples,
            "match_rate": match_rate,
            "match_percent": 100.0 * match_rate,
            "detected_only_match_rate": detected_match_rate,
            "detected_only_match_percent": 100.0 * detected_match_rate,
        },
        "per_target": {
            "total": total_targets,
            "with_any_match": targets_with_any_match,
            "any_match_rate": target_any_match_rate,
            "any_match_percent": 100.0 * target_any_match_rate,
        },
    }
    return rows, summary


def write_results(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Write per-sample JSONL and aggregate JSON evaluation artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "spacegroup_per_sample.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with (output_dir / "spacegroup_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Detect generated crystal space groups with spglib."
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing predictions.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <predictions-dir>/eval).",
    )
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tolerance", type=float, default=-1.0)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run space-group evaluation from the command line."""
    args = build_parser().parse_args(argv)
    predictions_dir = args.predictions_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else predictions_dir / "eval"
    )
    bundle = load_prediction_bundle(predictions_dir / "predictions.pt")
    rows, summary = evaluate_bundle(
        bundle,
        symprec=float(args.symprec),
        angle_tolerance=float(args.angle_tolerance),
        show_progress=not args.no_progress,
    )
    write_results(output_dir, rows, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
