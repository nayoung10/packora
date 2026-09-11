"""Callbacks for throughput trials with aggregate and per-step logging."""

# ruff: noqa: F722

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer

from src.utils.tensor_typing import Float


class ThroughputStepLoggingCallback(Callback):
    """Measure global training throughput and write per-step diagnostics."""

    def __init__(
        self,
        output_path: str,
        step_output_path: str,
        warmup_steps: int = 50,
        measured_steps: int = 200,
        measure_epoch: int | None = None,
        trial_id: str | int | None = None,
        summary: str = "",
        write_step_rows: bool = True,
    ) -> None:
        """Initialize throughput measurement settings."""
        super().__init__()
        self.output_path = Path(output_path)
        self.step_output_path = Path(step_output_path)
        self.warmup_steps = warmup_steps
        self.target_measured_steps = measured_steps
        self.measure_epoch = measure_epoch
        self.trial_id = trial_id
        self.summary = summary
        self.write_step_rows = bool(write_step_rows)
        self._seen_steps = 0
        self._measured_steps = 0
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._batch_start_time: float | None = None
        self._epoch_start_times: dict[int, float] = {}
        self._epoch_durations_seconds: dict[int, float] = {}
        self._global_samples = 0
        self._global_real_atoms = 0
        self._global_padded_atoms = 0
        self._global_real_pair_slots = 0
        self._global_padded_pair_slots = 0
        self._pre_measure_peak_allocated = 0
        self._pre_measure_peak_reserved = 0
        self._written = False

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Prepare output files on rank zero before training starts."""
        self._reset_cuda_peak_stats(pl_module)
        if not trainer.is_global_zero:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.write_step_rows:
            self.step_output_path.parent.mkdir(parents=True, exist_ok=True)
            self.step_output_path.write_text("")

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Start epoch timers and reset measured memory peaks when configured."""
        self._synchronize(pl_module)
        epoch = int(trainer.current_epoch)
        self._trainer_current_epoch = epoch
        self._epoch_start_times[epoch] = time.perf_counter()
        if self.measure_epoch is None:
            return
        if epoch != int(self.measure_epoch):
            return
        self._pre_measure_peak_allocated = self._local_peak_allocated(pl_module)
        self._pre_measure_peak_reserved = self._local_peak_reserved(pl_module)
        self._reset_cuda_peak_stats(pl_module)
        self._start_time = self._epoch_start_times[epoch]

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Record epoch duration and write aggregate metrics after measured epoch."""
        self._synchronize(pl_module)
        epoch = int(trainer.current_epoch)
        self._trainer_current_epoch = epoch
        end_time = time.perf_counter()
        start_time = self._epoch_start_times.get(epoch)
        if start_time is not None:
            self._epoch_durations_seconds[epoch] = max(end_time - start_time, 1e-12)
        if self.measure_epoch is None:
            return
        if epoch != int(self.measure_epoch):
            return
        self._end_time = end_time
        self._write_if_ready(trainer, pl_module)

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Start timers for the current batch and measurement window."""
        if self.measure_epoch is not None:
            if self._is_epoch_measured(trainer) and self._start_time is None:
                self._synchronize(pl_module)
                self._start_time = time.perf_counter()
                self._batch_start_time = self._start_time
                return
            self._batch_start_time = time.perf_counter()
            return

        if self._seen_steps == self.warmup_steps and self._start_time is None:
            self._synchronize(pl_module)
            self._start_time = time.perf_counter()
            self._batch_start_time = self._start_time
            return
        self._batch_start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Accumulate aggregate counters and append one step diagnostic row."""
        is_measured = self._is_measured_step(trainer)
        if not is_measured and not self.write_step_rows:
            self._seen_steps += 1
            return
        is_final_measured = (
            self.measure_epoch is None
            and is_measured
            and self._measured_steps + 1 == self.target_measured_steps
        )
        if is_final_measured:
            self._synchronize(pl_module)
        now = time.perf_counter()

        step_stats = self._global_step_stats(
            pl_module,
            batch,
            outputs,
        )
        samples, real_atoms, padded_atoms, real_pair_slots, padded_pair_slots, loss = (
            step_stats
        )
        phase = self._phase
        step_seconds = max(now - (self._batch_start_time or now), 1e-12)

        if is_measured:
            self._global_samples += samples
            self._global_real_atoms += real_atoms
            self._global_padded_atoms += padded_atoms
            self._global_real_pair_slots += real_pair_slots
            self._global_padded_pair_slots += padded_pair_slots
            self._measured_steps += 1
            if (
                self.measure_epoch is None
                and self._measured_steps == self.target_measured_steps
            ):
                self._end_time = now
                self._write_if_ready(trainer, pl_module)

        if self.write_step_rows:
            self._append_step_row(
                trainer=trainer,
                batch_idx=batch_idx,
                phase=phase,
                step_seconds=step_seconds,
                samples=samples,
                real_atoms=real_atoms,
                padded_atoms=padded_atoms,
                real_pair_slots=real_pair_slots,
                padded_pair_slots=padded_pair_slots,
                loss=loss,
            )
        self._seen_steps += 1

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Write partial aggregate metrics if training ends early."""
        if self._start_time is not None and self._end_time is None:
            self._synchronize(pl_module)
            self._end_time = time.perf_counter()
        self._write_if_ready(trainer, pl_module)

    @property
    def _total_steps(self) -> int:
        """Return warmup plus measured step count."""
        return self.warmup_steps + self.target_measured_steps

    @property
    def _phase(self) -> str:
        """Return the current benchmark phase name."""
        if self.measure_epoch is not None:
            try:
                epoch = int(self._current_epoch)
            except AttributeError:
                return "warmup"
            if epoch < int(self.measure_epoch):
                return "warmup"
            if epoch == int(self.measure_epoch):
                return "measured"
            return "extra"
        if self._seen_steps < self.warmup_steps:
            return "warmup"
        if self._seen_steps < self._total_steps:
            return "measured"
        return "extra"

    @property
    def _current_epoch(self) -> int:
        """Return the current trainer epoch cached for phase naming."""
        return self.__dict__.get("_trainer_current_epoch", 0)

    def _is_epoch_measured(self, trainer: Trainer) -> bool:
        """Return whether the current batch belongs to the measured epoch."""
        self._trainer_current_epoch = int(trainer.current_epoch)
        return int(trainer.current_epoch) == int(self.measure_epoch)

    def _is_measured_step(self, trainer: Trainer) -> bool:
        """Return whether the current step contributes to aggregate metrics."""
        if self.measure_epoch is not None:
            return self._is_epoch_measured(trainer)
        return self.warmup_steps <= self._seen_steps < self._total_steps

    def _extract_loss(self, outputs: Any) -> Float[""] | None:
        """Extract a scalar loss tensor from Lightning batch outputs."""
        if isinstance(outputs, torch.Tensor):
            return outputs.detach().float().mean()
        if isinstance(outputs, dict):
            value = outputs.get("loss")
            if isinstance(value, torch.Tensor):
                return value.detach().float().mean()
        return None

    def _global_step_stats(
        self,
        pl_module: LightningModule,
        batch: Any,
        outputs: Any,
    ) -> tuple[int, int, int, int, int, float | None]:
        """Return global sample, atom, pair-slot counts, and weighted loss."""
        local_samples = int(batch["num_atoms"].shape[0])
        local_real_atoms = int(batch["atom_mask"].sum().item())
        local_padded_atoms = int(batch["atom_mask"].numel())
        local_real_pair_slots = int(batch["num_atoms"].square().sum().item())
        n_max = int(batch["atom_mask"].shape[1])
        local_padded_pair_slots = int(local_samples * n_max * n_max)
        loss = self._extract_loss(outputs)
        loss_sum = float(loss.item()) * local_samples if loss is not None else 0.0
        loss_weight = float(local_samples) if loss is not None else 0.0
        values = torch.tensor(
            [
                local_samples,
                local_real_atoms,
                local_padded_atoms,
                local_real_pair_slots,
                local_padded_pair_slots,
                loss_sum,
                loss_weight,
            ],
            device=pl_module.device,
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        global_loss = None
        if float(values[6].item()) > 0.0:
            global_loss = float((values[5] / values[6]).item())
        return (
            int(values[0].item()),
            int(values[1].item()),
            int(values[2].item()),
            int(values[3].item()),
            int(values[4].item()),
            global_loss,
        )

    def _append_step_row(
        self,
        trainer: Trainer,
        batch_idx: int,
        phase: str,
        step_seconds: float,
        samples: int,
        real_atoms: int,
        padded_atoms: int,
        real_pair_slots: int,
        padded_pair_slots: int,
        loss: float | None,
    ) -> None:
        """Append one JSONL row from rank zero."""
        if not trainer.is_global_zero:
            return
        payload = {
            "trial_id": self.trial_id,
            "summary": self.summary,
            "step": self._seen_steps + 1,
            "global_step": int(trainer.global_step),
            "batch_idx": int(batch_idx),
            "phase": phase,
            "loss": loss,
            "duration_seconds": step_seconds,
            "epoch": self._current_epoch,
            "global_samples": samples,
            "global_real_atoms": real_atoms,
            "global_padded_atoms": padded_atoms,
            "global_real_pair_slots": real_pair_slots,
            "global_padded_pair_slots": padded_pair_slots,
            "samples_per_sec": samples / step_seconds,
            "steps_per_sec": 1.0 / step_seconds,
            "real_atoms_per_sec": real_atoms / step_seconds,
            "padded_atoms_per_sec": padded_atoms / step_seconds,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.step_output_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _synchronize(self, pl_module: LightningModule) -> None:
        """Synchronize CUDA kernels before reading a benchmark boundary time."""
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)

    def _reset_cuda_peak_stats(self, pl_module: LightningModule) -> None:
        """Reset local CUDA memory peak stats when running on CUDA."""
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def _local_peak_allocated(self, pl_module: LightningModule) -> int:
        """Return local peak allocated CUDA bytes."""
        if pl_module.device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_allocated(pl_module.device))

    def _local_peak_reserved(self, pl_module: LightningModule) -> int:
        """Return local peak reserved CUDA bytes."""
        if pl_module.device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_reserved(pl_module.device))

    def _global_memory_stats(self, pl_module: LightningModule) -> dict[str, int]:
        """Return global max CUDA memory peaks across ranks."""
        measured_allocated = self._local_peak_allocated(pl_module)
        measured_reserved = self._local_peak_reserved(pl_module)
        whole_allocated = max(self._pre_measure_peak_allocated, measured_allocated)
        whole_reserved = max(self._pre_measure_peak_reserved, measured_reserved)
        values = torch.tensor(
            [
                whole_allocated,
                whole_reserved,
                measured_allocated,
                measured_reserved,
                self._pre_measure_peak_allocated,
                self._pre_measure_peak_reserved,
            ],
            device=pl_module.device,
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
        return {
            "peak_allocated_bytes": int(values[0].item()),
            "peak_reserved_bytes": int(values[1].item()),
            "measured_peak_allocated_bytes": int(values[2].item()),
            "measured_peak_reserved_bytes": int(values[3].item()),
            "pre_measure_peak_allocated_bytes": int(values[4].item()),
            "pre_measure_peak_reserved_bytes": int(values[5].item()),
        }

    def _write_if_ready(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Write aggregate metrics once from global rank zero."""
        if self._written:
            return
        if self._start_time is None or self._end_time is None:
            return

        memory_stats = self._global_memory_stats(pl_module)
        if not trainer.is_global_zero:
            self._written = True
            return

        duration_seconds = max(self._end_time - self._start_time, 1e-12)
        payload = {
            "trial_id": self.trial_id,
            "summary": self.summary,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "warmup_steps": self.warmup_steps,
            "target_measured_steps": self.target_measured_steps,
            "measure_epoch": self.measure_epoch,
            "write_step_rows": self.write_step_rows,
            "measured_steps": self._measured_steps,
            "duration_seconds": duration_seconds,
            "epoch_durations_seconds": {
                str(epoch): duration
                for epoch, duration in sorted(self._epoch_durations_seconds.items())
            },
            "warmup_epoch_seconds": self._epoch_durations_seconds.get(0),
            "measured_epoch_seconds": (
                self._epoch_durations_seconds.get(int(self.measure_epoch))
                if self.measure_epoch is not None
                else None
            ),
            "global_samples": self._global_samples,
            "global_real_atoms": self._global_real_atoms,
            "global_padded_atoms": self._global_padded_atoms,
            "global_real_pair_slots": self._global_real_pair_slots,
            "global_padded_pair_slots": self._global_padded_pair_slots,
            "padding_efficiency": (
                self._global_real_atoms / self._global_padded_atoms
                if self._global_padded_atoms > 0
                else 0.0
            ),
            "pair_padding_efficiency": (
                self._global_real_pair_slots / self._global_padded_pair_slots
                if self._global_padded_pair_slots > 0
                else 0.0
            ),
            "samples_per_sec": self._global_samples / duration_seconds,
            "steps_per_sec": self._measured_steps / duration_seconds,
            "real_atoms_per_sec": self._global_real_atoms / duration_seconds,
            "padded_atoms_per_sec": self._global_padded_atoms / duration_seconds,
            "world_size": trainer.world_size,
            "step_output_path": str(self.step_output_path)
            if self.write_step_rows
            else None,
            **memory_stats,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2) + "\n")
        self._written = True
