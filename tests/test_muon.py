"""Unit tests for the mixed Muon and AdamW optimizer."""

from __future__ import annotations

import pytest
import torch

from src.models.optim.muon import MuonWithAdamW
from src.models.optim.scheduler import LinearWarmupLRScheduler


class _TinyModule(torch.nn.Module):
    """Minimal module with parameters that exercise Muon splitting."""

    def __init__(self) -> None:
        """Initialize small named submodules."""
        super().__init__()
        self.hidden = torch.nn.Linear(4, 4)
        self.input_embedder = torch.nn.Linear(4, 4)
        self.heads = torch.nn.Linear(4, 2)


def _assign_unit_grads(parameters: list[torch.nn.Parameter]) -> None:
    """Attach deterministic gradients to parameters."""
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)


def test_muon_splits_named_parameters() -> None:
    """Tests named parameters are split into Muon and AdamW groups."""
    module = _TinyModule()
    optimizer = MuonWithAdamW(module.named_parameters(), lr=1e-3)

    muon_group = next(group for group in optimizer.param_groups if group["use_muon"])
    adamw_group = next(
        group for group in optimizer.param_groups if not group["use_muon"]
    )

    assert muon_group["param_names"] == ["hidden.weight"]
    assert "hidden.bias" in adamw_group["param_names"]
    assert "input_embedder.weight" in adamw_group["param_names"]
    assert "heads.weight" in adamw_group["param_names"]


def test_scheduler_updates_real_muon_and_adamw_groups() -> None:
    """Tests LR schedulers update the optimizer groups that are stepped."""
    module = _TinyModule()
    optimizer = MuonWithAdamW(module.named_parameters(), lr=1.0)
    scheduler = LinearWarmupLRScheduler(
        optimizer=optimizer,
        start_lr=0.0,
        target_lr=0.2,
        warmup_no_steps=2,
    )

    _assign_unit_grads(list(module.parameters()))
    optimizer.step()
    scheduler.step()

    observed_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    assert observed_lrs == pytest.approx([0.1, 0.1])


def test_muon_state_dict_round_trip() -> None:
    """Tests mixed optimizer state can be saved and loaded."""
    module = _TinyModule()
    optimizer = MuonWithAdamW(module.named_parameters(), lr=1e-3)
    _assign_unit_grads(list(module.parameters()))
    optimizer.step()

    reloaded_module = _TinyModule()
    reloaded_optimizer = MuonWithAdamW(reloaded_module.named_parameters(), lr=1e-3)
    reloaded_optimizer.load_state_dict(optimizer.state_dict())

    assert len(reloaded_optimizer.state) == len(optimizer.state)
    assert reloaded_optimizer.param_groups[0]["use_muon"] is True
    assert reloaded_optimizer.param_groups[1]["use_muon"] is False


@pytest.mark.skipif(
    not hasattr(torch.optim, "Muon"),
    reason="torch.optim.Muon is unavailable in this environment.",
)
def test_muon_step_matches_torch_muon_for_2d_params() -> None:
    """Tests the batched Muon update matches PyTorch Muon for 2D params."""
    generator = torch.Generator().manual_seed(123)
    ours = [
        torch.nn.Parameter(torch.randn(4, 3, generator=generator)),
        torch.nn.Parameter(torch.randn(4, 3, generator=generator)),
    ]
    refs = [torch.nn.Parameter(parameter.detach().clone()) for parameter in ours]
    for parameter, reference in zip(ours, refs):
        grad = torch.randn(parameter.shape, generator=generator)
        parameter.grad = grad.clone()
        reference.grad = grad.clone()

    ours_optimizer = MuonWithAdamW(
        ours,
        lr=1e-3,
        weight_decay=0.0,
        adjust_lr_fn="original",
    )
    ref_optimizer = torch.optim.Muon(
        refs,
        lr=1e-3,
        weight_decay=0.0,
        adjust_lr_fn="original",
    )

    ours_optimizer.step()
    ref_optimizer.step()

    for parameter, reference in zip(ours, refs):
        torch.testing.assert_close(
            parameter.detach(),
            reference.detach(),
            atol=1e-4,
            rtol=1e-4,
        )
