"""Learning rate schedulers for optimizer-step-based training."""

import torch


class LinearWarmupLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linearly warm up learning rate, then keep it constant."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        start_lr: float = 0.0,
        target_lr: float = 1e-3,
        warmup_no_steps: int = 1000,
    ) -> None:
        """Initialize the linear warmup scheduler."""
        # Validate warmup settings
        if warmup_no_steps < 0:
            msg = "warmup_no_steps must be nonnegative"
            raise ValueError(msg)

        self.optimizer = optimizer
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_no_steps = warmup_no_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rates for the current optimizer step."""
        if not self._get_lr_called_within_step:
            msg = (
                "To get the last learning rate computed by the scheduler, use "
                "get_last_lr()"
            )
            raise RuntimeError(msg)

        step_no = self.last_epoch

        # Jump directly to target LR when warmup is disabled
        if self.warmup_no_steps == 0:
            lr = self.target_lr
        # Linearly interpolate from start_lr to target_lr during warmup
        elif step_no <= self.warmup_no_steps:
            warmup_fraction = step_no / self.warmup_no_steps
            lr = self.start_lr + warmup_fraction * (self.target_lr - self.start_lr)
        # Hold the peak LR after warmup
        else:
            lr = self.target_lr

        return [lr for _ in self.optimizer.param_groups]


class AlphaFoldLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Apply linear warmup, plateau, then stepwise multiplicative decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        base_lr: float = 0.0,
        max_lr: float = 1.8e-3,
        warmup_no_steps: int = 1000,
        start_decay_after_n_steps: int = 50000,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
    ) -> None:
        """Initialize the AlphaFold-style learning rate scheduler."""
        # Validate warmup and decay milestones
        if warmup_no_steps < 0:
            msg = "warmup_no_steps must be nonnegative"
            raise ValueError(msg)
        if start_decay_after_n_steps < 0:
            msg = "start_decay_after_n_steps must be nonnegative"
            raise ValueError(msg)
        if warmup_no_steps > start_decay_after_n_steps:
            msg = "warmup_no_steps must not exceed start_decay_after_n_steps"
            raise ValueError(msg)
        if decay_every_n_steps <= 0:
            msg = "decay_every_n_steps must be positive"
            raise ValueError(msg)

        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_no_steps = warmup_no_steps
        self.start_decay_after_n_steps = start_decay_after_n_steps
        self.decay_every_n_steps = decay_every_n_steps
        self.decay_factor = decay_factor
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rates for the current optimizer step."""
        if not self._get_lr_called_within_step:
            msg = (
                "To get the last learning rate computed by the scheduler, use "
                "get_last_lr()"
            )
            raise RuntimeError(msg)

        step_no = self.last_epoch

        # Run linear warmup when configured
        if self.warmup_no_steps > 0 and step_no <= self.warmup_no_steps:
            warmup_fraction = step_no / self.warmup_no_steps
            lr = self.base_lr + warmup_fraction * (self.max_lr - self.base_lr)
        # Decay after the configured step threshold
        elif step_no > self.start_decay_after_n_steps:
            steps_since_decay = step_no - self.start_decay_after_n_steps
            exponent = (steps_since_decay // self.decay_every_n_steps) + 1
            lr = self.max_lr * (self.decay_factor**exponent)
        # Hold maximum LR on the plateau
        else:
            lr = self.max_lr

        return [lr for _ in self.optimizer.param_groups]
