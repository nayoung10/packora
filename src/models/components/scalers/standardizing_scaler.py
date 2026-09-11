# ruff: noqa: F722,F821

import logging

import torch
import torch.nn as nn

from src.data.components.prior.stats_io import load_dataset_stats
from src.models.components.lattice_repr import cell_to_ltri_latent, ltri_latent_to_cell
from src.models.components.scalers import Scaler
from src.utils.tensor_typing import Bool, Float

logger = logging.getLogger(__name__)

# Safety floors to avoid division by near-zero std
MIN_COORD_STD: float = 0.1  # angstroms
MIN_LENGTH_STD: float = 0.1  # angstroms
MIN_ANGLE_STD: float = 1.0  # degrees
MIN_LTRI_STD: float = 0.1


class StandardizingScaler(Scaler, nn.Module):
    """Data-dependent z-score scaler with per-dataset statistics."""

    def __init__(
        self,
        datasets: list[dict],
        lattice_repr: str = "params",
        center_cart_coords: bool = True,
    ) -> None:
        """Load one dataset's statistics and build scaler buffers.

        Args:
            datasets: list containing one dict with keys data_dir and dataset_name.
            lattice_repr: "params", "cell", or "ltri".
        """
        nn.Module.__init__(self)
        self.lattice_repr = lattice_repr

        if len(datasets) != 1:
            raise ValueError(
                "StandardizingScaler expects exactly one dataset stats entry. "
                "Use an explicit dataset-id scaler for mixed-dataset training.",
            )

        ds = datasets[0]
        stats = load_dataset_stats(ds["data_dir"], ds["dataset_name"])
        ss = stats["scaler_stats"]

        if center_cart_coords:
            coord_mean_value = ss.get("coord_mean", [0.0, 0.0, 0.0])
            coord_std_value = ss["coord_std"]
        else:
            missing = [
                key
                for key in ("coord_mean_uncentered", "coord_std_uncentered")
                if key not in ss
            ]
            if missing:
                raise ValueError(
                    "Uncentered Cartesian coordinates require dataset statistics "
                    f"{missing} for '{ds['dataset_name']}'. Re-run: python "
                    "scripts/extract_dataset_stats.py "
                    f"--data_dir {ds['data_dir']} --dataset_name {ds['dataset_name']}"
                )
            coord_mean_value = ss["coord_mean_uncentered"]
            coord_std_value = ss["coord_std_uncentered"]

        coord_mean = torch.tensor(coord_mean_value, dtype=torch.float32)
        if coord_mean.shape != (3,):
            raise ValueError(
                f"Coordinate mean must have shape (3,), got {tuple(coord_mean.shape)}."
            )
        coord_std = torch.tensor(
            float(max(coord_std_value, MIN_COORD_STD)), dtype=torch.float32
        )
        length_mean = torch.tensor(ss["length_mean"])
        length_std = torch.tensor(ss["length_std"]).clamp(min=MIN_LENGTH_STD)
        angle_mean = torch.tensor(ss["angle_mean"])
        angle_std = torch.tensor(ss["angle_std"]).clamp(min=MIN_ANGLE_STD)
        cell_mean = torch.zeros(9)
        cell_std = torch.ones(9)
        ltri_mean = torch.zeros(6)
        ltri_std = torch.ones(6)

        # Load cell stats if available
        if "cell_mean" in ss:
            cell_mean = torch.tensor(ss["cell_mean"])
            cell_std = torch.tensor(ss["cell_std"]).clamp(min=MIN_LENGTH_STD)
        elif lattice_repr == "cell":
            raise ValueError(
                f"lattice_repr='cell' but 'cell_mean' not found in stats for "
                f"'{ds['dataset_name']}'. Re-run: python scripts/extract_dataset_stats.py "
                f"--dataset_name {ds['dataset_name']}"
            )

        # Load log-ltri stats if available
        if "ltri_mean" in ss:
            ltri_mean = torch.tensor(ss["ltri_mean"])
            ltri_std = torch.tensor(ss["ltri_std"]).clamp(min=MIN_LTRI_STD)
        elif lattice_repr == "ltri":
            raise ValueError(
                f"lattice_repr='ltri' but 'ltri_mean' not found in stats for "
                f"'{ds['dataset_name']}'. Re-run: python scripts/extract_dataset_stats.py "
                f"--dataset_name {ds['dataset_name']}"
            )

        logger.info(
            "Loaded scaler stats for %s: coord_mean=%s, coord_std=%.3f, "
            "lattice_repr=%s",
            ds["dataset_name"],
            coord_mean.tolist(),
            coord_std.item(),
            lattice_repr,
        )

        self.register_buffer("coord_mean", coord_mean, persistent=False)
        self.register_buffer("coord_std", coord_std)
        self.register_buffer("length_mean", length_mean)
        self.register_buffer("length_std", length_std)
        self.register_buffer("angle_mean", angle_mean)
        self.register_buffer("angle_std", angle_std)
        self.register_buffer("cell_mean", cell_mean)
        self.register_buffer("cell_std", cell_std)
        self.register_buffer("ltri_mean", ltri_mean)
        self.register_buffer("ltri_std", ltri_std)

    def scale_coords(
        self,
        cart_coords: Float["b n 3"],
        atom_mask: Bool["b n"] | None = None,
    ) -> Float["b n 3"]:
        """Standardize coordinates by the configured dataset mean and std."""
        scaled = (cart_coords - self.coord_mean) / self.coord_std
        if atom_mask is not None:
            scaled = scaled * atom_mask[..., None].to(dtype=scaled.dtype)
        return scaled

    def scale_lattice(
        self,
        lattice: Float["b d"],
    ) -> Float["b d"]:
        """Standardize lattice by per-dataset mean and std."""
        if self.lattice_repr == "cell":
            return (lattice - self.cell_mean) / self.cell_std
        if self.lattice_repr == "ltri":
            latent = cell_to_ltri_latent(lattice)
            return (latent - self.ltri_mean) / self.ltri_std

        # 6D params: separate length/angle standardization
        lengths = lattice[:, :3]
        angles = lattice[:, 3:]

        lengths_scaled = (lengths - self.length_mean) / self.length_std
        angles_scaled = (angles - self.angle_mean) / self.angle_std
        return torch.cat([lengths_scaled, angles_scaled], dim=-1)

    def unscale_coords(
        self,
        cart_coords: Float["b n 3"],
        atom_mask: Bool["b n"] | None = None,
    ) -> Float["b n 3"]:
        """Reverse coordinate standardization."""
        unscaled = cart_coords * self.coord_std + self.coord_mean
        if atom_mask is not None:
            unscaled = unscaled * atom_mask[..., None].to(dtype=unscaled.dtype)
        return unscaled

    def unscale_lattice(
        self,
        lattice: Float["b d"],
    ) -> Float["b d"]:
        """Reverse lattice standardization."""
        if self.lattice_repr == "cell":
            return lattice * self.cell_std + self.cell_mean
        if self.lattice_repr == "ltri":
            latent = lattice * self.ltri_std + self.ltri_mean
            return ltri_latent_to_cell(latent)

        # 6D params: separate length/angle unstandardization
        lengths_scaled = lattice[:, :3]
        angles_scaled = lattice[:, 3:]

        lengths = lengths_scaled * self.length_std + self.length_mean
        angles = angles_scaled * self.angle_std + self.angle_mean
        return torch.cat([lengths, angles], dim=-1)
