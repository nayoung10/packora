"""Test periodic pair-distance auxiliary loss behavior."""

# ruff: noqa: F722,F821

import torch
import torch.nn as nn
from einops import rearrange

from src.models.loss import PeriodicPairDistanceLoss
from src.models.flow_model import MaterialFlowMatching
from src.utils.tensor_typing import Float


class _CoordStdCellScaler:
    """Test scaler with cell lattice representation and coordinate std."""

    lattice_repr = "cell"

    def __init__(self, coord_std: float = 1.0) -> None:
        """Store the coordinate standardization value."""
        self.coord_std = torch.tensor(float(coord_std))

    def scale_coords(
        self,
        cart_coords: Float["b n 3"],
        atom_mask: torch.Tensor | None = None,
    ) -> Float["b n 3"]:
        """Scale coordinates by coordinate std."""
        return cart_coords / self.coord_std

    def scale_lattice(self, lattice: Float["b d"]) -> Float["b d"]:
        """Return cell lattice unchanged."""
        return lattice

    def unscale_coords(
        self,
        cart_coords: Float["b n 3"],
        atom_mask: torch.Tensor | None = None,
    ) -> Float["b n 3"]:
        """Unscale coordinates by coordinate std."""
        return cart_coords * self.coord_std

    def unscale_lattice(self, lattice: Float["b d"]) -> Float["b d"]:
        """Return cell lattice unchanged."""
        return lattice


def _cell(size: float) -> Float["1 3 3"]:
    """Build one cubic cell."""
    return torch.eye(3).unsqueeze(0) * size


def _flat_cell(cell: Float["b 3 3"]) -> Float["b 9"]:
    """Flatten a cell matrix."""
    return rearrange(cell, "b i j -> b (i j)")


def test_l1_scores_true_minimum_image_pairs_inside_cutoff() -> None:
    """L1 scores pairs included by true minimum-image distance."""
    loss_fn = PeriodicPairDistanceLoss(variant="l1")
    true_cell = _cell(20.0)
    pred_cell = _flat_cell(true_cell)
    true_coords = torch.tensor([[[0.0, 0.0, 0.0], [19.0, 0.0, 0.0]]])
    pred_coords = torch.tensor([[[0.0, 0.0, 0.0], [18.0, 0.0, 0.0]]])
    atom_mask = torch.tensor([[True, True]])
    coord_scale = torch.tensor(2.0)

    loss, per_sample = loss_fn(
        pred_coords=pred_coords,
        pred_cell=pred_cell,
        true_coords=true_coords,
        true_cell=true_cell,
        atom_mask=atom_mask,
        coord_scale=coord_scale,
    )

    assert torch.allclose(per_sample, torch.tensor([0.5]))
    assert torch.allclose(loss, torch.tensor(0.5))


def test_distances_use_predicted_cell_for_prediction_side() -> None:
    """Predicted distances use predicted coords and predicted cell."""
    loss_fn = PeriodicPairDistanceLoss(variant="l1")
    true_cell = _cell(10.0)
    pred_cell = _flat_cell(_cell(20.0))
    coords = torch.tensor([[[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]])
    atom_mask = torch.tensor([[True, True]])

    loss, per_sample = loss_fn(
        pred_coords=coords,
        pred_cell=pred_cell,
        true_coords=coords,
        true_cell=true_cell,
        atom_mask=atom_mask,
        coord_scale=torch.tensor(1.0),
    )

    assert torch.allclose(per_sample, torch.tensor([8.0]))
    assert torch.allclose(loss, torch.tensor(8.0))


def test_smooth_lddt_has_boltz_unit_sigmoid_floor() -> None:
    """Smooth-LDDT perfect matches keep the Boltz unit-sigmoid floor."""
    loss_fn = PeriodicPairDistanceLoss(variant="smooth_lddt")
    cell = _cell(10.0)
    coords = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    atom_mask = torch.tensor([[True, True]])
    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0])
    expected = 1.0 - torch.sigmoid(thresholds).mean()

    loss, per_sample = loss_fn(
        pred_coords=coords,
        pred_cell=_flat_cell(cell),
        true_coords=coords,
        true_cell=cell,
        atom_mask=atom_mask,
        coord_scale=torch.tensor(1.0),
    )

    assert torch.allclose(per_sample, expected.unsqueeze(0))
    assert torch.allclose(loss, expected)


def test_empty_scored_pair_mask_returns_zero() -> None:
    """Empty 15 angstrom pair masks return zero for both variants."""
    true_cell = _cell(100.0)
    pred_cell = _flat_cell(true_cell)
    coords = torch.tensor([[[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]]])
    atom_mask = torch.tensor([[True, True]])

    for variant in ("l1", "smooth_lddt"):
        loss_fn = PeriodicPairDistanceLoss(variant=variant)
        loss, per_sample = loss_fn(
            pred_coords=coords,
            pred_cell=pred_cell,
            true_coords=coords,
            true_cell=true_cell,
            atom_mask=atom_mask,
            coord_scale=torch.tensor(1.0),
        )

        assert torch.allclose(per_sample, torch.tensor([0.0]))
        assert torch.allclose(loss, torch.tensor(0.0))


def test_compute_loss_adds_periodic_pair_distance_when_enabled() -> None:
    """MaterialFlowMatching adds the auxiliary loss only when configured."""
    flow_matching = MaterialFlowMatching.__new__(MaterialFlowMatching)
    nn.Module.__init__(flow_matching)
    flow_matching.net = nn.Identity()
    flow_matching.scaler = _CoordStdCellScaler(coord_std=2.0)
    flow_matching.lattice_repr = "cell"
    flow_matching.periodic_pair_distance = PeriodicPairDistanceLoss(variant="l1")

    cell = _cell(20.0)
    batch = {
        "cart_coords": torch.tensor([[[0.0, 0.0, 0.0], [19.0, 0.0, 0.0]]]),
        "cell": cell,
        "atom_mask": torch.tensor([[True, True]]),
    }
    preds = {
        "coords": torch.tensor([[[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]]),
        "lattice": _flat_cell(cell),
    }

    loss_dict = flow_matching.compute_loss(batch, preds, t=torch.tensor([0.5]))

    assert "loss_periodic_pair_distance" in loss_dict
    assert "per_sample_periodic_pair_distance" in loss_dict
    assert torch.allclose(loss_dict["loss_periodic_pair_distance"], torch.tensor(0.5))
