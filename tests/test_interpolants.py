"""Verify interpolant equations: coefficients, derivatives, and velocity formula."""

# ruff: noqa: F722,F821

import torch
import pytest
import torch.nn as nn

from src.models.components.interpolants.linear import LinearInterpolant
from src.models.components.interpolants.trigonometric import TrigonometricInterpolant
from src.models.components.interpolants.variance_preserving import (
    VariancePreservingInterpolant,
    TauConstantSchedule,
    TauCosineSchedule,
    TauLinearSchedule,
)
from src.models.flow_model import MaterialFlowModel, MaterialFlowMatching
from src.utils.tensor_typing import Bool, Float


INTERPS = [
    LinearInterpolant(),
    TrigonometricInterpolant(),
    VariancePreservingInterpolant(TauConstantSchedule()),
    VariancePreservingInterpolant(TauCosineSchedule(offset=0.008)),
    VariancePreservingInterpolant(TauLinearSchedule(beta_min=0.1, beta_max=20.0)),
]

# Avoid exact 0/1 where some interpolants have singularities
T_VALS = torch.linspace(0.01, 0.99, 50)


@pytest.mark.parametrize("interp", INTERPS, ids=lambda i: type(i).__name__)
class TestInterpolantEquations:
    """Test suite for all interpolant implementations."""

    def test_boundary_a0_b0(self, interp: object) -> None:
        """a(~0) ≈ 1 and b(~0) ≈ 0 (prior-dominated at small t)."""
        t = torch.tensor([1e-3])
        assert torch.allclose(interp.a(t), torch.ones(1), atol=5e-2)
        assert torch.allclose(interp.b(t), torch.zeros(1), atol=5e-2)

    def test_boundary_a1_b1(self, interp: object) -> None:
        """a(~1) ≈ 0 and b(~1) ≈ 1 (data-dominated at large t)."""
        t = torch.tensor([1.0 - 1e-3])
        assert torch.allclose(interp.a(t), torch.zeros(1), atol=5e-2)
        assert torch.allclose(interp.b(t), torch.ones(1), atol=5e-2)

    def test_derivative_consistency(self, interp: object) -> None:
        """adot and bdot match finite-difference approximations."""
        eps = 1e-5
        # Use double precision for tighter finite-difference accuracy
        t = T_VALS.clone().double()
        a_fd = (interp.a(t + eps) - interp.a(t - eps)) / (2 * eps)
        b_fd = (interp.b(t + eps) - interp.b(t - eps)) / (2 * eps)
        assert torch.allclose(interp.adot(t), a_fd, atol=1e-3), (
            f"adot mismatch: max err {(interp.adot(t) - a_fd).abs().max():.6f}"
        )
        assert torch.allclose(interp.bdot(t), b_fd, atol=1e-3), (
            f"bdot mismatch: max err {(interp.bdot(t) - b_fd).abs().max():.6f}"
        )

    def test_interpolation_formula(self, interp: object) -> None:
        """It(t, x0, x1) = a(t)*x0 + b(t)*x1."""
        t = T_VALS[:5]
        x0 = torch.randn(5, 3)
        x1 = torch.randn(5, 3)
        x_t = interp.It(t, x0, x1)
        a_t = interp.a(t).unsqueeze(-1)
        b_t = interp.b(t).unsqueeze(-1)
        expected = a_t * x0 + b_t * x1
        assert torch.allclose(x_t, expected, atol=1e-6)

    def test_velocity_from_x1_pred(self, interp: object) -> None:
        """velocity_from_x1_pred matches dtIt when x0 is recovered from x_t and x1."""
        t = T_VALS[:5]
        x0 = torch.randn(5, 3)
        x1 = torch.randn(5, 3)

        x_t = interp.It(t, x0, x1)
        # Ground truth velocity
        v_true = interp.dtIt(t, x0, x1)
        # Velocity from x1 prediction (pretend network perfectly predicts x1)
        v_pred = interp.velocity_from_x1_pred(t, x_t, x1)

        assert torch.allclose(v_pred, v_true, atol=1e-4), (
            f"velocity mismatch: max err {(v_pred - v_true).abs().max():.6f}"
        )


class TestLinearVelocityBackwardsCompat:
    """Verify the general velocity formula matches the old linear-specific formula."""

    def test_matches_old_formula(self) -> None:
        """velocity_from_x1_pred == (x1 - x_t) / (1 - t) for LinearInterpolant."""
        interp = LinearInterpolant()
        t = T_VALS[:10]
        x0 = torch.randn(10, 4, 3)
        x1 = torch.randn(10, 4, 3)

        x_t = interp.It(t, x0, x1)
        # Old formula
        t_exp = t.unsqueeze(-1).unsqueeze(-1)
        v_old = (x1 - x_t) / (1.0 - t_exp)
        # New general formula
        v_new = interp.velocity_from_x1_pred(t, x_t, x1)

        assert torch.allclose(v_new, v_old, atol=1e-5), (
            f"backwards compat failure: max err {(v_new - v_old).abs().max():.6f}"
        )


@pytest.mark.parametrize("interp", INTERPS, ids=lambda i: type(i).__name__)
class TestScoreAndDiffusion:
    """Test suite for score_from_x1_pred and diffusion_coeff."""

    def test_score_consistency(self, interp: object) -> None:
        """Given x_t = a(t)*eps + b(t)*x1, score should equal -eps / a(t)."""
        t = T_VALS[:10]
        eps = torch.randn(10, 3)
        x1 = torch.randn(10, 3)
        x_t = interp.It(t, eps, x1)

        score = interp.score_from_x1_pred(t, x_t, x1)
        a_t = interp.a(t).unsqueeze(-1)
        expected = -eps / a_t

        assert torch.allclose(score, expected, atol=1e-4), (
            f"score mismatch: max err {(score - expected).abs().max():.6f}"
        )

    def test_diffusion_coeff_nonnegative(self, interp: object) -> None:
        """beta(t) = -2 * adot(t) * a(t) should be >= 0 for t in (0, 1)."""
        t = T_VALS.clone()
        beta = interp.diffusion_coeff(t)
        assert (beta >= -1e-6).all(), (
            f"negative diffusion coeff: min = {beta.min():.6f}"
        )

    def test_diffusion_coeff_zero_at_t0(self, interp: object) -> None:
        """beta(t) should be small near t=0 (prior end) for interpolants with adot(0)~-1, a(0)~1."""
        t_near_0 = torch.tensor([0.001])
        beta = interp.diffusion_coeff(t_near_0)
        # beta(0) = -2*adot(0)*a(0); for linear: 2*1*1=2, for trig: pi*sin(~0)*cos(~0)~0
        # Just verify it's finite and non-negative
        assert torch.isfinite(beta).all(), f"non-finite beta near t=0: {beta}"
        assert (beta >= -1e-6).all(), f"negative beta near t=0: {beta}"


class TestVariancePreserving:
    """Verify a(t)^2 + b(t)^2 = 1 for variance-preserving interpolants."""

    @pytest.mark.parametrize(
        "interp",
        [
            TrigonometricInterpolant(),
            VariancePreservingInterpolant(TauConstantSchedule()),
            VariancePreservingInterpolant(TauCosineSchedule(offset=0.008)),
        ],
        ids=["trigonometric", "vp_constant", "vp_cosine"],
    )
    def test_unit_norm(self, interp: object) -> None:
        """a(t)^2 + b(t)^2 should equal 1 for all t."""
        t = T_VALS.clone()
        norm_sq = interp.a(t) ** 2 + interp.b(t) ** 2
        assert torch.allclose(norm_sq, torch.ones_like(norm_sq), atol=1e-5), (
            f"norm deviation: max err {(norm_sq - 1.0).abs().max():.6f}"
        )


class IdentityScaler:
    """Test scaler that leaves coordinates and lattice unchanged."""

    lattice_repr = "params"

    def scale_coords(
        self,
        cart_coords: Float["b n 3"],
        atom_mask: Bool["b n"] | None = None,
    ) -> Float["b n 3"]:
        """Return coordinates unchanged."""
        return cart_coords

    def scale_lattice(self, lattice: Float["b d"]) -> Float["b d"]:
        """Return lattice unchanged."""
        return lattice


class TestEDMLossWeighting:
    """Verify EDM loss weights match the preconditioning equations."""

    def _model(self, eps: float = 1e-5) -> MaterialFlowModel:
        """Build a minimal model for coefficient helper tests."""
        return MaterialFlowModel(
            input_embedder=nn.Identity(),
            timestep_embedder=nn.Identity(),
            transformer=nn.Identity(),
            heads=nn.Identity(),
            edm_preconditioning={"enabled": True, "eps": eps},
        )

    def _flow_matching(self, net: nn.Module) -> MaterialFlowMatching:
        """Build a minimal flow-matching object for loss tests."""
        flow_matching = MaterialFlowMatching.__new__(MaterialFlowMatching)
        nn.Module.__init__(flow_matching)
        flow_matching.net = net
        flow_matching.scaler = IdentityScaler()
        flow_matching.lattice_repr = "params"
        return flow_matching

    def test_edm_l1_loss_weight_formula(self) -> None:
        """EDM L1 loss weight equals reciprocal c_out."""
        t = torch.tensor([0.1, 0.5, 0.9])
        one_minus_t = 1.0 - t
        denom = one_minus_t.square() + t.square()
        expected = denom.sqrt() / one_minus_t

        weight = self._model().edm_loss_weight(t, "l1")

        assert torch.allclose(weight, expected)

    def test_edm_l2_loss_weight_formula(self) -> None:
        """EDM L2 loss weight equals reciprocal c_out squared."""
        t = torch.tensor([0.1, 0.5, 0.9])
        one_minus_t = 1.0 - t
        denom = one_minus_t.square() + t.square()
        expected = denom / one_minus_t.square()

        weight = self._model().edm_loss_weight(t, "l2")

        assert torch.allclose(weight, expected)

    def test_edm_loss_weight_clips_endpoint_times(self) -> None:
        """EDM loss weight uses the same clipped time range as preconditioning."""
        eps = 1e-3
        t = torch.tensor([0.0, 1.0])
        clipped_t = t.clamp(min=eps, max=1.0 - eps)
        one_minus_t = 1.0 - clipped_t
        denom = one_minus_t.square() + clipped_t.square()
        expected = denom / one_minus_t.square()

        weight = self._model(eps=eps).edm_loss_weight(t, "l2")

        assert torch.isfinite(weight).all()
        assert torch.allclose(weight, expected)

    def test_compute_loss_applies_edm_weight_only_when_enabled(self) -> None:
        """EDM-enabled clean endpoint losses are weighted per sample."""
        t = torch.tensor([0.25, 0.75])
        batch = {
            "cart_coords": torch.ones(2, 2, 3),
            "lattice": torch.ones(2, 6),
            "atom_mask": torch.ones(2, 2, dtype=torch.bool),
        }
        preds = {
            "coords": torch.zeros(2, 2, 3),
            "lattice": torch.zeros(2, 6),
        }
        edm_net = self._model()
        edm_flow = self._flow_matching(edm_net)
        plain_flow = self._flow_matching(nn.Identity())
        plain_flow.net.edm_preconditioning_enabled = False

        edm_loss = edm_flow.compute_loss(batch, preds, t=t, loss_type="l1")
        plain_loss = plain_flow.compute_loss(batch, preds, t=t, loss_type="l1")
        expected_weighted = edm_net.edm_loss_weight(t, "l1").mean()
        edm_l2_loss = edm_flow.compute_loss(batch, preds, t=t, loss_type="l2")
        plain_l2_loss = plain_flow.compute_loss(batch, preds, t=t, loss_type="l2")
        expected_l2_weighted = edm_net.edm_loss_weight(t, "l2").mean()

        assert torch.allclose(plain_loss["loss_coords"], torch.tensor(1.0))
        assert torch.allclose(edm_loss["loss_coords"], expected_weighted)
        assert torch.allclose(edm_loss["loss_lattice_lengths"], expected_weighted)
        assert torch.allclose(edm_loss["loss_lattice_angles"], expected_weighted)
        assert torch.allclose(plain_l2_loss["loss_coords"], torch.tensor(1.0))
        assert torch.allclose(edm_l2_loss["loss_coords"], expected_l2_weighted)

    def test_compute_loss_can_disable_edm_loss_weight(self) -> None:
        """EDM preconditioning can keep loss weights at one."""
        t = torch.tensor([0.25, 0.75])
        batch = {
            "cart_coords": torch.ones(2, 2, 3),
            "lattice": torch.ones(2, 6),
            "atom_mask": torch.ones(2, 2, dtype=torch.bool),
        }
        preds = {
            "coords": torch.zeros(2, 2, 3),
            "lattice": torch.zeros(2, 6),
        }
        edm_net = self._model()
        edm_net.edm_loss_weight_enabled = False
        edm_flow = self._flow_matching(edm_net)

        loss = edm_flow.compute_loss(batch, preds, t=t, loss_type="l1")

        assert edm_net.edm_preconditioning_enabled is True
        assert torch.allclose(loss["loss_coords"], torch.tensor(1.0))
        assert torch.allclose(loss["loss_lattice_lengths"], torch.tensor(1.0))
        assert torch.allclose(loss["loss_lattice_angles"], torch.tensor(1.0))
