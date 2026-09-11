"""Check self-conditioning masking, training, caching, and sampler histories."""

# ruff: noqa: F722,F821
from typing import Any
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch import nn

from src.models.components.embedders.coord.fourier import FourierCartCoordEmbedder
from src.models.components.embedders.lattice.mlp import MLPLatticeEmbedder
from src.models.components.embedders.self_conditioning import SelfConditioningEmbedder
from src.models.components.transformers import DiT
from src.utils.tensor_typing import Float
from tests.test_autoguidance_sampling import _build_model, _conditioning
from tests.test_input_geometry_pairmixer import (
    _batch,
    _input_embedder,
    _model,
    _PairmixerRecorder,
)


def _embedder() -> SelfConditioningEmbedder:
    """Build a small real Fourier and lattice branch."""
    return SelfConditioningEmbedder(
        FourierCartCoordEmbedder(dim=3, num_channels=4),
        MLPLatticeEmbedder(input_dim=3, dim=3),
    )


def test_gate_after_embedding_and_padding() -> None:
    """Zero coordinates remain present features, while absence and padding add zero."""
    branch = _embedder()
    batch = _batch(torch.zeros(2, 3, 3))
    batch["atom_mask"][0, -1] = False
    absent = branch(batch)
    assert torch.count_nonzero(absent) == 0
    absent.sum().backward()
    assert all(p.grad is not None for p in branch.parameters())
    batch["self_conditioning"] = {
        "coords": torch.zeros(2, 3, 3),
        "lattice": torch.zeros(2, 3),
        "available": torch.tensor([True, False]),
    }
    present = branch(batch)
    assert torch.count_nonzero(present[0, :2]) > 0
    assert torch.count_nonzero(present[0, -1]) == 0
    assert torch.count_nonzero(present[1]) == 0


class _RecordingNet(nn.Module):
    """Record guesses and return distinct differentiable endpoint predictions."""

    def __init__(self, offset: float = 1.0) -> None:
        """Initialize an enabled branch marker and a trainable scalar."""
        super().__init__()
        self.self_conditioning_embedder = nn.Identity()
        self.weight = nn.Parameter(torch.tensor(offset))
        self.batches: list[dict[str, Any]] = []
        self.grad_enabled: list[bool] = []
        self.outputs: list[dict[str, Float["..."]]] = []

    def forward(self, batch: dict[str, Any]) -> dict[str, Float["..."]]:
        """Return per-row endpoints exposing detached reuse and particle ordering."""
        self.batches.append(batch)
        self.grad_enabled.append(torch.is_grad_enabled())
        rows = torch.arange(batch["x_t"].shape[0], device=batch["x_t"].device)
        value = self.weight + len(self.batches) + rows
        result = {
            "coords": torch.ones_like(batch["x_t"]) * value[:, None, None],
            "lattice": torch.ones_like(batch["l_t"]) * value[:, None],
        }
        result = {key: value.to(self.weight.dtype) for key, value in result.items()}
        self.outputs.append(result)
        return result


@pytest.mark.parametrize("selection", [[0.1, 0.9], [0.9, 0.9], [0.1], [0.9]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_training_subset_and_detach(selection: list[float], dtype: torch.dtype) -> None:
    """Only selected examples receive a same-state no-grad preliminary pass."""
    net = _RecordingNet().to(dtype=dtype)
    flow = _build_model(net)
    b = len(selection)
    batch = {
        "cart_coords": torch.randn(b, 2, 3),
        "lattice": torch.ones(b, 6),
        "atom_mask": torch.ones(b, 2, dtype=torch.bool),
        "indices": torch.zeros(b, 2, dtype=torch.long),
        "conditioning": {"atomic_numbers": torch.ones(b, 2, dtype=torch.long)},
    }
    with patch(
        "src.models.flow_model.torch.rand", return_value=torch.tensor(selection)
    ):
        result = flow(batch)
    selected = torch.tensor(selection) < 0.5
    assert len(net.batches) == 1 + int(selected.any())
    if selected.any():
        preliminary, main = net.batches
        assert net.grad_enabled == [False, True]
        for key in ("x_t", "l_t", "times"):
            torch.testing.assert_close(preliminary[key], main[key][selected])
        assert preliminary["conditioning_context"] == main["conditioning_context"]
        guess = main["self_conditioning"]
        assert not guess["coords"].requires_grad
        torch.testing.assert_close(
            guess["coords"][selected], net.outputs[0]["coords"].float()
        )
        torch.testing.assert_close(guess["available"], selected)
        assert torch.count_nonzero(guess["coords"][~selected]) == 0
    result["coords"].sum().backward()
    assert net.weight.grad is not None
    net.batches.clear()
    flow(batch, conditioning_context="val_loss")
    assert len(net.batches) == 1
    assert net.batches[0].get("self_conditioning") is None
    net.batches.clear()
    flow.eval()(batch)
    assert len(net.batches) == 1


@pytest.mark.parametrize("method", ["ode", "sde", "heun"])
@pytest.mark.parametrize("guidance", [False, True])
def test_sampler_histories_and_call_counts(method: str, guidance: bool) -> None:
    """Reuse each branch's last endpoint, including Heun correctors, and reset per sample."""
    main, bad = _RecordingNet(), _RecordingNet(10.0)
    flow = _build_model(main).eval()
    if guidance:
        flow.set_autoguidance(bad, 1.5)
    for _ in range(2):
        for net in (main, bad):
            net.batches.clear()
            net.outputs.clear()
        result = flow.sample(
            torch.tensor([2]), num_steps=3, method=method, conditioning=_conditioning()
        )
        assert torch.isfinite(result["cart_coords"]).all()
        for net in [main, bad] if guidance else [main]:
            assert len(net.batches) == (5 if method == "heun" else 3)
            assert net.batches[0].get("self_conditioning") is None
            for batch, previous in zip(net.batches[1:], net.outputs[:-1], strict=True):
                for key in ("coords", "lattice"):
                    torch.testing.assert_close(
                        batch["self_conditioning"][key], previous[key]
                    )


@pytest.mark.parametrize("edm", [False, True])
def test_cache_and_pair_track_independence(edm: bool) -> None:
    """Changing guesses preserve cached predictions and fixed pair representations."""
    transformer = DiT(dim=3, depth=2, heads=1, dim_pair=2, share_pair_bias_norm=True)
    for parameter in transformer.parameters():
        nn.init.normal_(parameter, std=0.2)
    pairmixer = _PairmixerRecorder()
    model = _model(
        _input_embedder(use_pairwise=True),
        transformer,
        pairmixer=pairmixer,
        noisy_input_entry="after_pairmixer",
    ).eval()
    model.self_conditioning_embedder = _embedder()
    model.edm_preconditioning_enabled = edm
    batch = _batch(torch.randn(2, 3, 3), torch.randn(2, 3, 3, 2))
    batch["flow_times"] = batch["times"]
    with torch.no_grad():
        cache = model.build_inference_cache(batch)
        for _ in range(2):
            batch["self_conditioning"] = {
                "coords": torch.randn(2, 3, 3),
                "lattice": torch.randn(2, 3),
                "available": torch.tensor([True, False]),
            }
            reference = model(batch)
            calls = pairmixer.calls
            with patch.object(
                transformer,
                "_project_pair_biases",
                wraps=transformer._project_pair_biases,
            ) as project:
                cached = model({**batch, "inference_cache": cache})
                assert project.call_count == 0
            assert pairmixer.calls == calls
            for key in reference:
                torch.testing.assert_close(reference[key], cached[key], rtol=0, atol=0)


def test_config_default_off_and_independent_weights() -> None:
    """Default adds no weights and enabling instantiates independent embedders."""
    with initialize(version_base="1.3", config_path="../configs"):
        off = compose(config_name="train.yaml")
        on = compose(
            config_name="train.yaml", overrides=["model/self_conditioning=default"]
        )
    assert off.model.net.self_conditioning_embedder is None
    assert on.model.self_conditioning_probability == 0.5
    branch = instantiate(on.model.net.self_conditioning_embedder)
    noisy = instantiate(on.model.net.input_embedder.single_embedder.coord_embedder)
    assert type(branch.coord_embedder) is type(noisy)
    assert all(
        a is not b
        for a, b in zip(
            branch.coord_embedder.parameters(), noisy.parameters(), strict=True
        )
    )


def test_fk_reindexes_both_histories() -> None:
    """Resampled particles inherit the selected parent estimates on both branches."""
    main, bad = _RecordingNet(), _RecordingNet(10.0)
    flow = _build_model(main).eval()
    flow.set_autoguidance(bad, 1.5)
    selected = torch.tensor([1, 1])
    with (
        patch.object(
            flow, "_get_mlip_energy", return_value=lambda x, lattice, a, m: x[:, 0, 0]
        ),
        patch.object(flow, "_fk_resample_indices", return_value=selected),
    ):
        flow.sample(
            torch.tensor([2]),
            num_steps=2,
            method="sde",
            conditioning=_conditioning(),
            steering_args={
                "fk_steering": True,
                "num_particles": 2,
                "fk_start_time": 0.0,
                "fk_resampling_interval": 1,
            },
        )
    for net in (main, bad):
        for key in ("coords", "lattice"):
            torch.testing.assert_close(
                net.batches[1]["self_conditioning"][key], net.outputs[0][key][selected]
            )


class _AutocastNet(nn.Module):
    """Expose cached linear weights in preliminary and main denoiser passes."""

    def __init__(self) -> None:
        """Create projections eligible for autocast's weight cache."""
        super().__init__()
        self.self_conditioning_embedder = nn.Identity()
        self.coord = nn.Linear(3, 3)
        self.lattice = nn.Linear(6, 6)
        self.cache_states: list[bool] = []
        self.output_dtypes: list[torch.dtype] = []

    def forward(self, batch: dict[str, Any]) -> dict[str, Float["..."]]:
        """Record autocast state and predict through the same weights twice."""
        self.cache_states.append(torch.is_autocast_cache_enabled())
        result = {
            "coords": self.coord(batch["x_t"]),
            "lattice": self.lattice(batch["l_t"]),
        }
        self.output_dtypes.append(result["coords"].dtype)
        return result


def test_preliminary_autocast_preserves_training_gradients() -> None:
    """Prevent no-grad cached casts from detaching the main pass's parameters."""
    net = _AutocastNet()
    flow = _build_model(net)
    flow.self_conditioning_probability = 1.0
    batch = {
        "cart_coords": torch.randn(2, 2, 3),
        "lattice": torch.ones(2, 6),
        "atom_mask": torch.ones(2, 2, dtype=torch.bool),
        "indices": torch.zeros(2, 2, dtype=torch.long),
        "conditioning": {"atomic_numbers": torch.ones(2, 2, dtype=torch.long)},
    }
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=True):
        result = flow(batch)
        loss = (
            result["coords"].float().square().mean()
            + result["lattice"].float().square().mean()
        )
        assert torch.is_autocast_cache_enabled()
    assert loss.requires_grad
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()
    )
    assert net.cache_states == [False, True]
    assert net.output_dtypes == [torch.bfloat16, torch.bfloat16]
