"""Unit tests for optimizer-step-based learning rate schedulers."""

import pytest
import torch

from src.models.optim.scheduler import AlphaFoldLRScheduler, LinearWarmupLRScheduler


def _build_optimizer(initial_lr: float = 1.0) -> torch.optim.Optimizer:
    """Build a minimal optimizer for scheduler tests."""
    # Use one scalar parameter to keep the optimizer setup simple.
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    return torch.optim.SGD([parameter], lr=initial_lr)


def _collect_lrs(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_steps: int,
) -> list[float]:
    """Advance optimizer and scheduler and collect LR values."""
    lrs: list[float] = []
    for _ in range(num_steps):
        # Follow the standard optimizer-then-scheduler update order.
        optimizer.step()
        scheduler.step()
        lrs.append(float(optimizer.param_groups[0]["lr"]))
    return lrs


def test_linear_warmup_schedule_profile() -> None:
    """Tests linear warmup reaches target and then stays constant."""
    optimizer = _build_optimizer()
    scheduler = LinearWarmupLRScheduler(
        optimizer=optimizer,
        start_lr=0.0,
        target_lr=1.0,
        warmup_no_steps=4,
    )

    observed_lrs = _collect_lrs(optimizer=optimizer, scheduler=scheduler, num_steps=7)
    expected_lrs = [0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
    assert observed_lrs == pytest.approx(expected_lrs)


def test_linear_warmup_zero_steps_is_constant() -> None:
    """Tests zero warmup jumps directly to target LR."""
    optimizer = _build_optimizer(initial_lr=3.0)
    scheduler = LinearWarmupLRScheduler(
        optimizer=optimizer,
        start_lr=0.0,
        target_lr=0.2,
        warmup_no_steps=0,
    )

    observed_lrs = _collect_lrs(optimizer=optimizer, scheduler=scheduler, num_steps=3)
    assert observed_lrs == pytest.approx([0.2, 0.2, 0.2])


def test_linear_warmup_negative_steps_raises() -> None:
    """Tests linear warmup rejects negative warmup steps."""
    optimizer = _build_optimizer()
    with pytest.raises(ValueError, match="warmup_no_steps"):
        LinearWarmupLRScheduler(
            optimizer=optimizer,
            warmup_no_steps=-1,
        )


def test_alphafold_schedule_profile() -> None:
    """Tests AlphaFold schedule warmup, plateau, and stepwise decay."""
    optimizer = _build_optimizer()
    scheduler = AlphaFoldLRScheduler(
        optimizer=optimizer,
        base_lr=0.0,
        max_lr=1.0,
        warmup_no_steps=2,
        start_decay_after_n_steps=5,
        decay_every_n_steps=3,
        decay_factor=0.5,
    )

    observed_lrs = _collect_lrs(optimizer=optimizer, scheduler=scheduler, num_steps=12)
    expected_lrs = [
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
        0.5,
        0.5,
        0.25,
        0.25,
        0.25,
        0.125,
        0.125,
    ]
    assert observed_lrs == pytest.approx(expected_lrs)


def test_alphafold_zero_warmup_starts_at_max_lr() -> None:
    """Tests AlphaFold schedule starts at max LR when warmup is disabled."""
    optimizer = _build_optimizer()
    scheduler = AlphaFoldLRScheduler(
        optimizer=optimizer,
        base_lr=0.0,
        max_lr=0.5,
        warmup_no_steps=0,
        start_decay_after_n_steps=3,
        decay_every_n_steps=4,
        decay_factor=0.5,
    )

    observed_lrs = _collect_lrs(optimizer=optimizer, scheduler=scheduler, num_steps=5)
    assert observed_lrs == pytest.approx([0.5, 0.5, 0.5, 0.25, 0.25])


def test_alphafold_invalid_step_args_raise() -> None:
    """Tests AlphaFold schedule validates warmup and milestone constraints."""
    optimizer = _build_optimizer()
    with pytest.raises(ValueError, match="warmup_no_steps must not exceed"):
        AlphaFoldLRScheduler(
            optimizer=optimizer,
            warmup_no_steps=10,
            start_decay_after_n_steps=5,
        )
    with pytest.raises(ValueError, match="start_decay_after_n_steps"):
        AlphaFoldLRScheduler(
            optimizer=optimizer,
            start_decay_after_n_steps=-1,
        )
    with pytest.raises(ValueError, match="decay_every_n_steps"):
        AlphaFoldLRScheduler(
            optimizer=optimizer,
            decay_every_n_steps=0,
        )
