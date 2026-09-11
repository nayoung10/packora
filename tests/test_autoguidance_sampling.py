import pytest
import torch
import torch.nn as nn

from src.models.components.interpolants.linear import LinearInterpolant
from src.models.flow_model import MaterialFlowMatching


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


class _UnusedTimeSampler:
    """Placeholder time sampler for constructing MaterialFlowMatching."""

    def sample(self, batch_size, device):
        """Return unused zero times."""
        return torch.zeros(batch_size, device=device)


class _ConstantNet(nn.Module):
    """Return deterministic endpoint predictions."""

    def __init__(
        self,
        coord_value: float,
        lattice_value: float,
        mismatch_coords: bool = False,
    ) -> None:
        """Store constant prediction values."""
        super().__init__()
        self.coord_value = float(coord_value)
        self.lattice_value = float(lattice_value)
        self.mismatch_coords = bool(mismatch_coords)
        self.calls = 0

    def forward(self, noisy_batch):
        """Return constant clean endpoint predictions."""
        self.calls += 1
        coords = torch.full_like(noisy_batch["x_t"], self.coord_value)
        lattice = torch.full_like(noisy_batch["l_t"], self.lattice_value)
        if self.mismatch_coords:
            coords = coords[:, :1]
        return {
            "coords": coords,
            "lattice": lattice,
        }


class _CacheNet(nn.Module):
    """Return predictions that expose which cache was used."""

    def __init__(self, base_value: float, cache_bias: float) -> None:
        """Store prediction and cache values."""
        super().__init__()
        self.base_value = float(base_value)
        self.cache_bias = float(cache_bias)
        self.cache_builds = 0
        self.forward_cache_biases: list[float] = []

    def build_inference_cache(self, noisy_batch):
        """Build a scalar cache unique to this net."""
        self.cache_builds += 1
        return {
            "bias": torch.tensor(
                self.cache_bias,
                device=noisy_batch["x_t"].device,
                dtype=noisy_batch["x_t"].dtype,
            )
        }

    def forward(self, noisy_batch):
        """Return endpoint predictions shifted by this net's cache."""
        cache = noisy_batch.get("inference_cache")
        if cache is None:
            raise AssertionError("Expected inference cache.")
        bias = cache["bias"]
        self.forward_cache_biases.append(float(bias.item()))
        value = self.base_value + bias
        return {
            "coords": torch.ones_like(noisy_batch["x_t"]) * value,
            "lattice": torch.ones_like(noisy_batch["l_t"]) * value,
        }


def _build_model(net: nn.Module) -> MaterialFlowMatching:
    """Build a deterministic flow-matching sampler for tests."""
    model = MaterialFlowMatching(
        net=net,
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
    return model


def _conditioning() -> dict[str, torch.Tensor]:
    """Return minimal conditioning for one two-atom source sample."""
    return {
        "atomic_numbers": torch.tensor([[6, 1]], dtype=torch.long),
    }


@pytest.mark.parametrize("method", ["ode", "sde", "heun"])
def test_autoguidance_weight_one_matches_off_for_sampling_methods(
    method: str,
) -> None:
    """Autoguidance weight one is tensor-exactly identical to guidance off."""
    model = _build_model(_ConstantNet(coord_value=1.0, lattice_value=2.0))

    torch.manual_seed(123)
    off = model.sample(
        num_atoms=torch.tensor([2], dtype=torch.long),
        multiplicity=1,
        num_steps=2,
        method=method,
        sde_noise_scale=0.1,
        conditioning=_conditioning(),
    )

    bad_net = _ConstantNet(coord_value=10.0, lattice_value=20.0)
    model.set_autoguidance(bad_net=bad_net, weight=1.0)
    torch.manual_seed(123)
    on = model.sample(
        num_atoms=torch.tensor([2], dtype=torch.long),
        multiplicity=1,
        num_steps=2,
        method=method,
        sde_noise_scale=0.1,
        conditioning=_conditioning(),
    )

    assert bad_net.calls == 0
    for key in ("cart_coords", "lattice", "atomic_numbers", "atom_mask"):
        assert torch.equal(on[key], off[key])


def test_autoguidance_combines_main_and_bad_endpoint_predictions() -> None:
    """Guided predictions follow w * main + (1 - w) * bad."""
    model = _build_model(_ConstantNet(coord_value=3.0, lattice_value=5.0))
    model.set_autoguidance(
        bad_net=_ConstantNet(coord_value=1.0, lattice_value=2.0),
        weight=2.0,
    )
    noisy_batch = {
        "x_t": torch.zeros(2, 3, 3),
        "l_t": torch.zeros(2, 6),
    }

    preds = model._sample_model_forward(noisy_batch)

    assert torch.allclose(preds["coords"], torch.full((2, 3, 3), 5.0))
    assert torch.allclose(preds["lattice"], torch.full((2, 6), 8.0))


def test_autoguidance_rejects_bad_prediction_shape_mismatch() -> None:
    """Bad model output shapes must match main model endpoint shapes."""
    model = _build_model(_ConstantNet(coord_value=3.0, lattice_value=5.0))
    model.set_autoguidance(
        bad_net=_ConstantNet(coord_value=1.0, lattice_value=2.0, mismatch_coords=True),
        weight=2.0,
    )
    noisy_batch = {
        "x_t": torch.zeros(2, 3, 3),
        "l_t": torch.zeros(2, 6),
    }

    with pytest.raises(ValueError, match="shape mismatch"):
        model._sample_model_forward(noisy_batch)


def test_autoguidance_uses_separate_inference_caches() -> None:
    """Main and bad models build and receive separate inference caches."""
    main_net = _CacheNet(base_value=3.0, cache_bias=0.25)
    bad_net = _CacheNet(base_value=1.0, cache_bias=-0.5)
    model = _build_model(main_net)
    model.set_autoguidance(bad_net=bad_net, weight=2.0)

    model.sample(
        num_atoms=torch.tensor([2], dtype=torch.long),
        multiplicity=1,
        num_steps=1,
        method="ode",
        conditioning=_conditioning(),
        use_inference_cache=True,
    )

    assert main_net.cache_builds == 1
    assert bad_net.cache_builds == 1
    assert main_net.forward_cache_biases == [0.25]
    assert bad_net.forward_cache_biases == [-0.5]
