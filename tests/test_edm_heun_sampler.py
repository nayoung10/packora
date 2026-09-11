import torch
import torch.nn as nn
import pytest

from src.models.components.interpolants.linear import LinearInterpolant
from src.models.components.sampling.edm_heun import (
    EDMHeunSamplerConfig,
    edm_derivative,
    edm_heun_config_from_dict,
    karras_sigma_schedule,
    sigma_to_flow_time,
)
from src.models.flow_model import MaterialFlowMatching


class _IdentityScaler:
    """Minimal scaler that leaves generated tensors unchanged."""

    lattice_repr = "params"

    def scale_coords(self, cart_coords, atom_mask=None):
        """Return coordinates unchanged."""
        return cart_coords

    def scale_lattice(self, lattice):
        """Return lattice unchanged."""
        return lattice

    def unscale_coords(self, cart_coords, atom_mask=None):
        """Return coordinates unchanged."""
        return cart_coords

    def unscale_lattice(self, lattice):
        """Return lattice unchanged."""
        return lattice


class _UnusedTimeSampler:
    """Placeholder time sampler for constructing MaterialFlowMatching."""

    def sample(self, batch_size, device):
        """Return unused zero times."""
        return torch.zeros(batch_size, device=device)


class _CleanEndpointNet(nn.Module):
    """Predict a deterministic centered clean endpoint."""

    def forward(self, noisy_batch):
        """Return clean coordinate and lattice endpoints."""
        x_t = noisy_batch["x_t"]
        l_t = noisy_batch["l_t"]
        coords = torch.zeros_like(x_t)
        coords[:, 0, 0] = 1.0
        coords[:, 1, 0] = -1.0
        lattice = torch.ones_like(l_t)
        return {"coords": coords, "lattice": lattice}


class _OffsetEndpointNet(nn.Module):
    """Predict a clean endpoint with a nonzero Cartesian centroid."""

    def forward(self, noisy_batch):
        """Return offset coordinate and zero lattice endpoints."""
        return {
            "coords": torch.full_like(noisy_batch["x_t"], 2.0),
            "lattice": torch.zeros_like(noisy_batch["l_t"]),
        }


def test_edm_heun_config_defaults_and_overrides() -> None:
    """Parse default and overridden sampler args."""
    default = edm_heun_config_from_dict(None)
    overridden = edm_heun_config_from_dict({"rho": 3.0, "s_churn": 0.0})

    assert default.sigma_min == 0.002
    assert default.sigma_max == 80.0
    assert default.s_churn == 60.0
    assert overridden.rho == 3.0
    assert overridden.s_churn == 0.0


def test_karras_sigma_schedule_is_descending_with_terminal_zero() -> None:
    """Build a Karras schedule with the expected endpoints."""
    config = EDMHeunSamplerConfig(sigma_min=0.1, sigma_max=10.0, rho=2.0)
    sigmas = karras_sigma_schedule(
        num_steps=4,
        config=config,
        device=torch.device("cpu"),
    )

    assert sigmas.shape == (5,)
    assert torch.allclose(sigmas[0], torch.tensor(10.0))
    assert torch.allclose(sigmas[-2], torch.tensor(0.1), atol=1e-6)
    assert torch.allclose(sigmas[-1], torch.tensor(0.0))
    assert torch.all(sigmas[:-2] > sigmas[1:-1])


def test_sigma_to_flow_time_matches_linear_mapping() -> None:
    """Map EDM sigma to linear flow time."""
    sigma = torch.tensor([80.0, 1.0, 0.0])
    expected = torch.tensor([1.0 / 81.0, 0.5, 1.0])

    assert torch.allclose(sigma_to_flow_time(sigma), expected)


def test_edm_derivative_uses_clean_endpoint_prediction() -> None:
    """Compute the EDM derivative from y_sigma and clean x1."""
    y_sigma = torch.tensor([3.0, 5.0])
    x1_pred = torch.tensor([1.0, 1.0])
    sigma = torch.tensor(2.0)

    assert torch.allclose(
        edm_derivative(y_sigma, x1_pred, sigma), torch.tensor([1.0, 2.0])
    )


def test_material_flow_matching_heun_one_step_reaches_clean_prediction() -> None:
    """One deterministic Heun step reaches the clean endpoint at sigma zero."""
    model = MaterialFlowMatching(
        net=_CleanEndpointNet(),
        interpolant=LinearInterpolant(),
        scaler=_IdentityScaler(),
        prior={
            "coord_sampler": "centered_gaussian",
            "lattice_sampler": "standard_gaussian",
            "lattice_dim": 6,
        },
        time_sampler=_UnusedTimeSampler(),
    )
    model.coord_sampler = lambda cart_coords, atom_mask: torch.zeros_like(cart_coords)
    model.lattice_sampler = lambda batch_size: torch.zeros(batch_size, 6)

    result = model.sample(
        num_atoms=torch.tensor([2], dtype=torch.long),
        multiplicity=1,
        num_steps=1,
        method="heun",
        conditioning={"atomic_numbers": torch.tensor([[6, 1]], dtype=torch.long)},
        sampler_args={
            "sigma_min": 0.1,
            "sigma_max": 1.0,
            "s_churn": 0.0,
        },
    )

    expected_coords = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    assert torch.allclose(result["cart_coords"], expected_coords)
    assert torch.allclose(result["lattice"], torch.ones(1, 6))


def test_material_flow_matching_sample_preserves_padded_atom_count() -> None:
    """Generate with the padded conditioning width, not num_atoms.max()."""
    model = MaterialFlowMatching(
        net=_CleanEndpointNet(),
        interpolant=LinearInterpolant(),
        scaler=_IdentityScaler(),
        prior={
            "coord_sampler": "centered_gaussian",
            "lattice_sampler": "standard_gaussian",
            "lattice_dim": 6,
        },
        time_sampler=_UnusedTimeSampler(),
    )
    model.coord_sampler = lambda cart_coords, atom_mask: torch.zeros_like(cart_coords)
    model.lattice_sampler = lambda batch_size: torch.zeros(batch_size, 6)

    result = model.sample(
        num_atoms=torch.tensor([2], dtype=torch.long),
        multiplicity=1,
        num_steps=1,
        method="ode",
        conditioning={"atomic_numbers": torch.tensor([[6, 1, 8, 9]])},
    )

    assert result["cart_coords"].shape == (1, 4, 3)
    assert result["atom_mask"].tolist() == [[True, True, False, False]]
    assert result["atomic_numbers"].tolist() == [[6, 1, 0, 0]]


@pytest.mark.parametrize("method", ["ode", "sde", "heun"])
def test_sampling_methods_preserve_uncentered_endpoint(method: str) -> None:
    """Retain nonzero centroids through every supported sampling path."""
    model = MaterialFlowMatching(
        net=_OffsetEndpointNet(),
        interpolant=LinearInterpolant(),
        scaler=_IdentityScaler(),
        prior={
            "coord_sampler": "centered_gaussian",
            "lattice_sampler": "standard_gaussian",
            "lattice_dim": 6,
        },
        time_sampler=_UnusedTimeSampler(),
        center_cart_coords=False,
    )
    model.coord_sampler = lambda cart_coords, atom_mask: torch.zeros_like(cart_coords)
    model.lattice_sampler = lambda batch_size: torch.zeros(batch_size, 6)
    sampler_args = (
        {"sigma_min": 0.1, "sigma_max": 1.0, "s_churn": 0.0}
        if method == "heun"
        else None
    )

    result = model.sample(
        num_atoms=torch.tensor([2], dtype=torch.long),
        num_steps=1,
        method=method,
        sde_noise_scale=0.0,
        conditioning={"atomic_numbers": torch.tensor([[6, 1]], dtype=torch.long)},
        sampler_args=sampler_args,
    )

    assert torch.all(result["cart_coords"].mean(dim=1) > 1.0)
