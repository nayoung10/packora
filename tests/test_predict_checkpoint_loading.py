from pathlib import Path

import torch

import src.prediction.checkpoint as checkpoint_module
from src.utils.checkpoint_state import remap_state_dict_keys


def test_remap_state_dict_keys_strips_orig_mod_for_eager_target() -> None:
    """Compiled checkpoint keys load into an eager model namespace."""
    state_dict = {
        "flow_matching.net._orig_mod.layer.weight": "weight",
        "flow_matching.scaler.coord_std": "scale",
    }
    target_keys = {
        "flow_matching.net.layer.weight",
        "flow_matching.scaler.coord_std",
    }

    adapted = remap_state_dict_keys(state_dict, target_keys)

    assert adapted == {
        "flow_matching.net.layer.weight": "weight",
        "flow_matching.scaler.coord_std": "scale",
    }


def test_remap_state_dict_keys_inserts_orig_mod_for_compiled_target() -> None:
    """Eager checkpoint keys load into a compiled model namespace."""
    state_dict = {
        "flow_matching.net.layer.weight": "weight",
        "flow_matching.scaler.coord_std": "scale",
    }
    target_keys = {
        "flow_matching.net._orig_mod.layer.weight",
        "flow_matching.scaler.coord_std",
    }

    adapted = remap_state_dict_keys(state_dict, target_keys)

    assert adapted == {
        "flow_matching.net._orig_mod.layer.weight": "weight",
        "flow_matching.scaler.coord_std": "scale",
    }


def test_load_model_preserving_rng_restores_torch_rng(monkeypatch) -> None:
    """Bad checkpoint loading must not advance the caller's RNG stream."""

    def fake_load_model(*args, **kwargs):
        """Consume RNG like model initialization would."""
        torch.rand(4)
        return "model", "cfg"

    monkeypatch.setattr(checkpoint_module, "load_model", fake_load_model)
    torch.manual_seed(123)
    expected_state = torch.random.get_rng_state()

    result = checkpoint_module.load_model_preserving_rng(Path("/tmp/bad.ckpt"))

    assert result == ("model", "cfg")
    assert torch.equal(torch.random.get_rng_state(), expected_state)
