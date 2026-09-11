import torch
import torch.nn as nn
import pytest

import src.models.flow_model as flow_model_module


class _IdentityScaler:
    """Minimal scaler for sampling tests."""

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


class _LinearInterpolant:
    """Minimal linear interpolant for sampling tests."""

    def velocity_from_x1_pred(self, t, x_t, x1_pred):
        """Return the linear endpoint velocity."""
        expanded_t = t
        for _ in range(x_t.ndim - 1):
            expanded_t = expanded_t.unsqueeze(-1)
        return (x1_pred - x_t) / (1.0 - expanded_t + 1e-6)


class _UnusedTimeSampler:
    """Placeholder time sampler for constructing MaterialFlowMatching."""

    def sample(self, batch_size, device):
        """Return unused zero times."""
        return torch.zeros(batch_size, device=device)


class _ParticleValueNet(nn.Module):
    """Predict particle-dependent coordinates with lower energy in particle 1."""

    def forward(self, noisy_batch):
        """Return deterministic clean endpoint predictions."""
        x_t = noisy_batch["x_t"]
        l_t = noisy_batch["l_t"]
        row_ids = torch.arange(x_t.shape[0], device=x_t.device)
        group_ids = torch.div(row_ids, 2, rounding_mode="floor")
        particle_ids = row_ids - group_ids * 2
        values = (group_ids.to(dtype=x_t.dtype) + 1.0) * 10.0
        values = values - particle_ids.to(dtype=x_t.dtype) * 8.0
        coords = torch.zeros_like(x_t)
        coords[:, 0, 0] = values
        coords[:, 1, 0] = -values
        return {
            "coords": coords,
            "lattice": torch.zeros_like(l_t),
        }


class _CoordinateEnergy(nn.Module):
    """Fake MLIP energy equal to the first x-coordinate."""

    def __init__(self, *args, **kwargs) -> None:
        """Ignore model-loading arguments."""
        super().__init__()

    def forward(self, cart_coords, lattice, atomic_numbers, atom_mask):
        """Return a deterministic per-row energy."""
        return cart_coords[:, 0, 0]


def test_fk_steering_returns_lowest_final_energy_particle(monkeypatch) -> None:
    """Reduce FK particles to the lowest final-energy particle per source group."""
    monkeypatch.setattr(flow_model_module, "MLIPEnergy", _CoordinateEnergy)
    model = flow_model_module.MaterialFlowMatching(
        net=_ParticleValueNet(),
        interpolant=_LinearInterpolant(),
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
        num_atoms=torch.tensor([2, 2], dtype=torch.long),
        multiplicity=1,
        num_steps=1,
        method="ode",
        conditioning={
            "atomic_numbers": torch.tensor([[6, 1], [8, 1]], dtype=torch.long),
        },
        steering_args={
            "fk_steering": True,
            "energy_fn": "mlip",
            "num_particles": 2,
            "fk_lambda": 2.0,
            "fk_resampling_interval": 5,
            "fk_start_time": 0.80,
            "potential_mode": "immediate",
            "mlip_model": "uma-s-1p2",
            "mlip_task_name": "omc",
        },
    )

    scale = torch.tensor(0.998 / (0.999 + 1e-6))
    expected_energy = torch.tensor([2.0, 12.0]) * scale
    assert result["cart_coords"].shape == (2, 2, 3)
    assert result["final_energy"].shape == (2,)
    assert torch.allclose(result["final_energy"], expected_energy, atol=1e-5)
    assert torch.allclose(result["cart_coords"][:, 0, 0], expected_energy, atol=1e-5)


def test_inference_cache_rejects_fk_steering() -> None:
    """Inference cache fails clearly before FK resampling can reorder tensors."""
    model = flow_model_module.MaterialFlowMatching(
        net=_ParticleValueNet(),
        interpolant=_LinearInterpolant(),
        scaler=_IdentityScaler(),
        prior={
            "coord_sampler": "centered_gaussian",
            "lattice_sampler": "standard_gaussian",
            "lattice_dim": 6,
        },
        time_sampler=_UnusedTimeSampler(),
    )

    with pytest.raises(ValueError, match="Inference cache is not supported"):
        model.sample(
            num_atoms=torch.tensor([2], dtype=torch.long),
            multiplicity=1,
            num_steps=1,
            method="sde",
            conditioning={
                "atomic_numbers": torch.tensor([[6, 1]], dtype=torch.long),
            },
            steering_args={"fk_steering": True},
            use_inference_cache=True,
        )
