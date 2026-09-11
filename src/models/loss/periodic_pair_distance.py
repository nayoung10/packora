# ruff: noqa: F722,F821

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from src.utils.tensor_typing import Bool, Float


class PeriodicPairDistanceLoss(nn.Module):
    """Compute PBC pair-distance auxiliary losses for crystal endpoints."""

    def __init__(
        self,
        variant: Literal["l1", "smooth_lddt"],
        cutoff: float = 15.0,
        thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        disable_autocast: bool = False,
    ) -> None:
        """Initialize the periodic pair-distance loss."""
        super().__init__()
        if variant not in {"l1", "smooth_lddt"}:
            raise ValueError(f"Unsupported periodic pair-distance variant: {variant}")
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive.")
        threshold_values = tuple(float(threshold) for threshold in thresholds)
        if len(threshold_values) == 0:
            raise ValueError("thresholds must contain at least one value.")
        if any(threshold <= 0.0 for threshold in threshold_values):
            raise ValueError("thresholds must be positive.")

        self.variant = variant
        self.cutoff = float(cutoff)
        self.disable_autocast = bool(disable_autocast)
        ticks = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        offsets = torch.cartesian_prod(ticks, ticks, ticks)
        self.register_buffer(
            "pbc_offsets",
            rearrange(offsets, "k c -> 1 1 1 k c"),
            persistent=False,
        )
        threshold_tensor = torch.tensor(threshold_values, dtype=torch.float32)
        self.register_buffer("smooth_thresholds", threshold_tensor, persistent=False)

    def _cell_matrix(self, cell: Float["b ..."]) -> Float["b 3 3"]:
        """Normalize flattened or matrix cell tensors to matrix form."""
        if cell.ndim == 3 and cell.shape[-2:] == (3, 3):
            return cell
        if cell.ndim == 2 and cell.shape[-1] == 9:
            return rearrange(cell, "b (i j) -> b i j", i=3, j=3)
        raise ValueError(f"Expected cell shape (B, 3, 3) or (B, 9), got {cell.shape}.")

    def _minimum_image_distances(
        self,
        coords: Float["b n 3"],
        cell: Float["b ..."],
    ) -> Float["b n n"]:
        """Compute wrapped 27-image PBC minimum distances in angstroms."""
        cell_matrix = self._cell_matrix(cell)
        autocast_context = (
            torch.autocast(device_type=coords.device.type, enabled=False)
            if self.disable_autocast
            else nullcontext()
        )
        with autocast_context:
            coords_calc = coords.to(dtype=torch.float32)
            cell_calc = cell_matrix.to(dtype=torch.float32)
            frac = torch.einsum(
                "b n c, b c f -> b n f",
                coords_calc,
                torch.linalg.inv(cell_calc),
            )
            frac = frac - frac.floor()
            delta = rearrange(frac, "b i c -> b i 1 c") - rearrange(
                frac,
                "b j c -> b 1 j c",
            )
            offsets = self.pbc_offsets.to(device=coords.device, dtype=coords_calc.dtype)
            delta_images = rearrange(delta, "b i j c -> b i j 1 c") + offsets
            gram = cell_calc @ rearrange(cell_calc, "b i j -> b j i")
            dist2_images = torch.einsum(
                "b i j k c, b c d, b i j k d -> b i j k",
                delta_images,
                gram,
                delta_images,
            )
            min_dist2 = dist2_images.amin(dim=-1)
            return min_dist2.clamp_min(1e-12).sqrt()

    def _pair_mask(
        self,
        true_dist: Float["b n n"],
        atom_mask: Bool["b n"],
    ) -> Bool["b n n"]:
        """Build the non-self true-distance inclusion mask."""
        real_pair = rearrange(atom_mask, "b i -> b i 1") & rearrange(
            atom_mask,
            "b j -> b 1 j",
        )
        n = atom_mask.shape[1]
        diagonal = torch.eye(n, dtype=torch.bool, device=atom_mask.device)
        return (
            real_pair & ~rearrange(diagonal, "i j -> 1 i j") & (true_dist < self.cutoff)
        )

    def _masked_pair_mean(
        self,
        values: Float["b n n"],
        pair_mask: Bool["b n n"],
    ) -> Float["b"]:
        """Average pair values per structure with zero for empty masks."""
        weights = pair_mask.to(dtype=values.dtype)
        counts = weights.sum(dim=(-1, -2))
        summed = (values * weights).sum(dim=(-1, -2))
        return torch.where(
            counts > 0,
            summed / counts.clamp_min(1.0),
            torch.zeros_like(summed),
        )

    def forward(
        self,
        pred_coords: Float["b n 3"],
        pred_cell: Float["b ..."],
        true_coords: Float["b n 3"],
        true_cell: Float["b ..."],
        atom_mask: Bool["b n"],
        coord_scale: Float[""],
    ) -> tuple[Float[""], Float["b"]]:
        """Return scalar and per-sample periodic pair-distance losses."""
        pred_dist = self._minimum_image_distances(pred_coords, pred_cell)
        true_dist = self._minimum_image_distances(true_coords, true_cell)
        pair_mask = self._pair_mask(true_dist, atom_mask)
        distance_error = (pred_dist - true_dist).abs()

        if self.variant == "l1":
            coord_scale = coord_scale.to(
                device=distance_error.device,
                dtype=distance_error.dtype,
            )
            pair_loss = distance_error / coord_scale.clamp_min(1e-12)
        else:
            thresholds = self.smooth_thresholds.to(
                device=distance_error.device,
                dtype=distance_error.dtype,
            )
            score = torch.sigmoid(
                rearrange(thresholds, "k -> 1 1 1 k")
                - rearrange(distance_error, "b i j -> b i j 1")
            ).mean(dim=-1)
            pair_loss = 1.0 - score

        per_sample = self._masked_pair_mean(pair_loss, pair_mask)
        return per_sample.mean(), per_sample
