"""Lightweight training throughput logging callback."""

from __future__ import annotations

import time
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer


class SimpleThroughputLoggingCallback(Callback):
    """Log simple step throughput metrics during training."""

    def __init__(self, every_n_steps: int = 100, warmup_steps: int = 50) -> None:
        """Initialize throughput logging cadence."""
        super().__init__()
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0.")
        self.every_n_steps = int(every_n_steps)
        self.warmup_steps = int(warmup_steps)
        self._batch_start_time: float | None = None

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Start timing the current training batch."""
        self._batch_start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log throughput for selected training batches."""
        global_step = int(trainer.global_step)
        if not self._should_log(global_step):
            return

        now = time.perf_counter()
        step_seconds = max(now - (self._batch_start_time or now), 1e-12)
        samples = self._global_num_samples(pl_module, batch)
        if not trainer.is_global_zero:
            return

        pl_module.log(
            "train/samples_per_sec",
            samples / step_seconds,
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
            rank_zero_only=True,
        )
        pl_module.log(
            "train/steps_per_sec",
            1.0 / step_seconds,
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
            rank_zero_only=True,
        )

    def _should_log(self, global_step: int) -> bool:
        """Return whether throughput should be logged for a global step."""
        if global_step == 0:
            return False
        if global_step <= self.warmup_steps:
            return False
        return global_step % self.every_n_steps == 0

    def _global_num_samples(
        self,
        pl_module: LightningModule,
        batch: Any,
    ) -> float:
        """Return the global number of samples in the current batch."""
        local_samples = float(batch["num_atoms"].shape[0])
        samples = torch.tensor(
            local_samples,
            device=pl_module.device,
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(samples, op=torch.distributed.ReduceOp.SUM)
        return float(samples.item())
