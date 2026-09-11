from typing import Any

import pytest
import torch
import torch.nn as nn
from einops import rearrange

from src.data.components.conditioning import SPACEGROUP_MODEL_NULL_ID
from src.models.components.interpolants.linear import LinearInterpolant
from src.models.flow_model import MaterialFlowMatching


class _IdentityScaler:
    """Provide identity transforms for sampler tests."""

    lattice_repr = "params"

    def scale_coords(self, cart_coords: Any, atom_mask: Any = None) -> Any:
        """Return coordinates unchanged."""
        return cart_coords

    def scale_lattice(self, lattice: Any) -> Any:
        """Return lattice unchanged."""
        return lattice

    def unscale_coords(self, cart_coords: Any, atom_mask: Any = None) -> Any:
        """Return coordinates unchanged."""
        return cart_coords

    def unscale_lattice(self, lattice: Any) -> Any:
        """Return lattice unchanged."""
        return lattice


class _UnusedTimeSampler:
    """Provide an unused time sampler for construction."""

    def sample(self, batch_size: int, device: torch.device) -> Any:
        """Return zero times."""
        return torch.zeros(batch_size, device=device)


class _SpacegroupNet(nn.Module):
    """Return nontrivial endpoints determined by the space-group mask."""

    def __init__(self) -> None:
        """Initialize call and cache traces."""
        super().__init__()
        self.forward_masks: list[float] = []
        self.cache_masks: list[float] = []

    def build_inference_cache(self, noisy_batch: dict[str, Any]) -> dict[str, Any]:
        """Cache the branch-specific space-group mask."""
        mask = noisy_batch["conditioning"]["spacegroup_mask"].clone()
        self.cache_masks.extend(float(value) for value in mask.tolist())
        return {"spacegroup_mask": mask}

    def forward(self, noisy_batch: dict[str, Any]) -> dict[str, Any]:
        """Predict distinct conditional and unconditional endpoints."""
        cache = noisy_batch.get("inference_cache")
        if cache is None:
            mask = noisy_batch["conditioning"]["spacegroup_mask"]
        else:
            mask = cache["spacegroup_mask"]
        self.forward_masks.extend(float(value) for value in mask.tolist())
        coord_value = 1.0 + 2.0 * rearrange(mask, "b -> b 1 1")
        lattice_value = 2.0 + 3.0 * rearrange(mask, "b -> b 1")
        return {
            "coords": torch.ones_like(noisy_batch["x_t"]) * coord_value,
            "lattice": torch.ones_like(noisy_batch["l_t"]) * lattice_value,
        }


def _build_model(net: nn.Module | None = None) -> MaterialFlowMatching:
    """Build a deterministic sampler with nonzero conditional behavior."""
    model = MaterialFlowMatching(
        net=_SpacegroupNet() if net is None else net,
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


def _conditioning(spacegroup_on: bool = True) -> dict[str, Any]:
    """Return one source row with an explicit or NULL space-group branch."""
    return {
        "atomic_numbers": torch.tensor([[6, 1]], dtype=torch.long),
        "formal_charges": torch.tensor([[0, 0]], dtype=torch.long),
        "spacegroup_number": torch.tensor(
            [14 if spacegroup_on else SPACEGROUP_MODEL_NULL_ID],
            dtype=torch.long,
        ),
        "spacegroup_mask": torch.tensor([1.0 if spacegroup_on else 0.0]),
    }


@pytest.mark.parametrize(
    ("spacegroup_on", "weight", "expected_mask"),
    [(True, 1.0, 1.0), (False, 1.0, 0.0), (False, 0.0, 0.0)],
)
def test_spacegroup_conditioning_uses_one_prepared_forward(
    spacegroup_on: bool,
    weight: float,
    expected_mask: float,
) -> None:
    """Use one forward with the space-group mask prepared by policy and presence."""
    model = _build_model()

    model.sample(
        num_atoms=torch.tensor([2]),
        num_steps=1,
        method="ode",
        conditioning=_conditioning(spacegroup_on=spacegroup_on),
        spacegroup_guidance_weight=weight,
    )

    assert model.net.forward_masks == [expected_mask]


def test_spacegroup_conditioning_builds_one_inference_cache() -> None:
    """Build one inference cache for ordinary space-group conditioning."""
    model = _build_model()

    model.sample(
        num_atoms=torch.tensor([2]),
        num_steps=1,
        method="ode",
        conditioning=_conditioning(spacegroup_on=True),
        spacegroup_guidance_weight=1.0,
        use_inference_cache=True,
    )

    assert model.net.cache_masks == [1.0]
    assert model.net.forward_masks == [1.0]


def test_non_unit_spacegroup_cfg_is_unsupported() -> None:
    """Reject non-unit space-group CFG until its implementation is restored."""
    model = _build_model()

    with pytest.raises(NotImplementedError, match="CFG is currently unsupported"):
        model.sample(
            num_atoms=torch.tensor([2]),
            num_steps=1,
            conditioning=_conditioning(spacegroup_on=True),
            spacegroup_guidance_weight=2.0,
        )


def test_ordinary_spacegroup_conditioning_supports_autoguidance() -> None:
    """Allow normal space-group conditioning with the separate autoguidance model."""
    model = _build_model()
    bad_net = _SpacegroupNet()
    model.set_autoguidance(bad_net=bad_net, weight=2.0)

    model.sample(
        num_atoms=torch.tensor([2]),
        num_steps=1,
        conditioning=_conditioning(spacegroup_on=True),
        spacegroup_guidance_weight=1.0,
    )

    assert model.net.forward_masks == [1.0]
    assert bad_net.forward_masks == [1.0]


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_spacegroup_cfg_rejects_invalid_weights(weight: float) -> None:
    """Reject negative and nonfinite guidance weights."""
    model = _build_model()

    with pytest.raises(ValueError, match="finite and >= 0"):
        model.sample(
            num_atoms=torch.tensor([2]),
            num_steps=1,
            conditioning=_conditioning(spacegroup_on=True),
            spacegroup_guidance_weight=weight,
        )
