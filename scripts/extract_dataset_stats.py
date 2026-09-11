"""Extract dataset statistics (scaler params, num_atoms histogram) from a preprocessed LMDB dataset.

Usage:
    python scripts/extract_dataset_stats.py --dataset_name omc25
    python scripts/extract_dataset_stats.py --dataset_name csd --limit 10000
"""

import argparse
import logging
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from einops import rearrange

from src.data.components.prior.stats_io import save_dataset_stats
from src.models.components.lattice_repr import cell_to_ltri_latent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def coordinate_statistics(coords_list: list[np.ndarray]) -> dict[str, object]:
    """Compute centered and uncentered coordinate normalization statistics."""
    all_coords = np.concatenate(coords_list, axis=0)
    coord_mean_uncentered = all_coords.mean(axis=0)
    residuals = all_coords - coord_mean_uncentered

    centered = [coords - coords.mean(axis=0, keepdims=True) for coords in coords_list]
    all_centered = np.concatenate(centered, axis=0)

    # TODO: Ablate pooled scalar coordinate std against per-axis standardization.
    return {
        "coord_mean": [0.0, 0.0, 0.0],
        "coord_std": float(all_centered.std()),
        "coord_mean_uncentered": coord_mean_uncentered.tolist(),
        "coord_std_uncentered": float(residuals.std()),
        "num_coord_atoms": int(all_coords.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract dataset stats from a preprocessed LMDB dataset."
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max entries to read. Set to 0 to use all.",
    )
    args = parser.parse_args()

    lmdb_path = Path(args.data_dir) / args.dataset_name / f"{args.split}.lmdb"
    if not lmdb_path.is_dir():
        raise FileNotFoundError(
            f"LMDB not found at {lmdb_path}. Run preprocessing first."
        )

    # Open LMDB and read entry count
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    with env.begin() as txn:
        total = int(txn.get(b"__len__").decode())

    # Determine how many entries to read
    n_read = min(total, args.limit) if args.limit > 0 else total
    logger.info("Reading %d / %d entries from %s", n_read, total, lmdb_path)

    # Collect lattice parameters, cell matrices, coordinates, and atom counts from LMDB
    lattice_list = []
    cell_list = []
    coords_list = []
    num_atoms_list = []
    with env.begin() as txn:
        for i in range(n_read):
            key = f"{i:08d}".encode()
            material = pickle.loads(txn.get(key))
            lattice_list.append(material.lattice_parameters)
            cell_list.append(material.cell)
            coords_list.append(material.cart_coords)
            num_atoms_list.append(len(material.conditioning.atomic_numbers))
    env.close()

    coord_stats = coordinate_statistics(coords_list)
    logger.info(
        "Coordinate stats: centered_std=%.4f, uncentered_mean=%s, "
        "uncentered_std=%.4f (from %d structures, %d atoms)",
        coord_stats["coord_std"],
        coord_stats["coord_mean_uncentered"],
        coord_stats["coord_std_uncentered"],
        len(coords_list),
        coord_stats["num_coord_atoms"],
    )

    # Compute lattice statistics for standardization
    lattice_params = torch.tensor(np.stack(lattice_list), dtype=torch.float32)
    lengths = lattice_params[:, :3]
    angles = lattice_params[:, 3:]

    scaler_stats = {
        key: value for key, value in coord_stats.items() if key != "num_coord_atoms"
    }
    scaler_stats.update(
        {
            "length_mean": lengths.mean(dim=0).tolist(),
            "length_std": lengths.std(dim=0, correction=0).tolist(),
            "angle_mean": angles.mean(dim=0).tolist(),
            "angle_std": angles.std(dim=0, correction=0).tolist(),
        }
    )
    logger.info(
        "Scaler stats — length_mean: %s, length_std: %s",
        scaler_stats["length_mean"],
        scaler_stats["length_std"],
    )
    logger.info(
        "Scaler stats — angle_mean: %s, angle_std: %s",
        scaler_stats["angle_mean"],
        scaler_stats["angle_std"],
    )

    # Compute cell matrix statistics for standardization (per-element)
    cells = torch.tensor(np.stack(cell_list), dtype=torch.float32)  # (M, 3, 3)
    cells_flat = rearrange(cells, "m i j -> m (i j)")  # (M, 9)
    scaler_stats["cell_mean"] = cells_flat.mean(dim=0).tolist()
    scaler_stats["cell_std"] = cells_flat.std(dim=0, correction=0).tolist()
    logger.info(
        "Scaler stats — cell_mean: %s",
        [f"{v:.4f}" for v in scaler_stats["cell_mean"]],
    )
    logger.info(
        "Scaler stats — cell_std: %s",
        [f"{v:.4f}" for v in scaler_stats["cell_std"]],
    )

    # Compute Crystalite-style log-ltri statistics for standardization
    ltri = cell_to_ltri_latent(cells)
    scaler_stats["ltri_mean"] = ltri.mean(dim=0).tolist()
    scaler_stats["ltri_std"] = ltri.std(dim=0, correction=0).tolist()
    logger.info(
        "Scaler stats — ltri_mean: %s",
        [f"{v:.4f}" for v in scaler_stats["ltri_mean"]],
    )
    logger.info(
        "Scaler stats — ltri_std: %s",
        [f"{v:.4f}" for v in scaler_stats["ltri_std"]],
    )

    # Build num_atoms histogram
    num_atoms_arr = np.array(num_atoms_list)
    max_atoms = int(num_atoms_arr.max())
    values = list(range(1, max_atoms + 1))
    counts = [int(np.sum(num_atoms_arr == v)) for v in values]

    logger.info(
        "num_atoms range: [%d, %d], median: %d",
        num_atoms_arr.min(),
        max_atoms,
        int(np.median(num_atoms_arr)),
    )

    # Save to JSON sidecar
    stats = {
        "num_atoms_histogram": {
            "values": values,
            "counts": counts,
        },
        "scaler_stats": scaler_stats,
    }
    path = save_dataset_stats(args.data_dir, args.dataset_name, stats)
    logger.info("Saved to %s", path)


if __name__ == "__main__":
    main()
