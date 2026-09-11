"""Callbacks for writing fixed-subset train/validation generations to disk."""

# ruff: noqa: F722,F821

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from lightning import Callback, Trainer
from torch.utils.data import DataLoader, DistributedSampler, Subset

from src.data.components.collate import collate_fn
from src.data.dataset import MaterialDataset
from src.models.flow_module import MaterialFlowModule
from src.utils.tensor_typing import Float, Int

logger = logging.getLogger(__name__)
FLEXIBILITY_LABELS = ("rigid", "flexible")


@dataclass(frozen=True)
class GenerationSubsetRow:
    """One fixed target selected for repeated generation."""

    selection_index: int
    dataset_index: int
    csd_refcode: str
    num_atoms: int
    flexibility: str


def _utc_now() -> str:
    """Return an ISO timestamp for artifact metadata."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload, creating parents first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}.")
    return payload


def _cpu_tensor_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Move tensor values in a dictionary to CPU."""
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in payload.items()
    }


def _distributed_barrier() -> None:
    """Synchronize ranks when distributed training is active."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _metadata_refcode(metadata: dict[str, Any], dataset_index: int) -> str:
    """Return a normalized CSD refcode from one metadata row."""
    value = metadata.get("csd_refcode")
    refcode = "" if value is None else str(value).strip().upper()
    if not refcode:
        raise ValueError(f"Missing csd_refcode for dataset_index={dataset_index}.")
    return refcode


def _metadata_flexibility(metadata: dict[str, Any], dataset_index: int) -> str:
    """Return the persisted rigid/flexible label from one metadata row."""
    value = metadata.get("flexibility")
    flexibility = "" if value is None else str(value).strip().lower()
    if flexibility not in FLEXIBILITY_LABELS:
        raise ValueError(
            f"Missing or invalid flexibility label for dataset_index={dataset_index}."
        )
    return flexibility


def _write_subset_csv(path: Path, rows: list[GenerationSubsetRow]) -> None:
    """Write subset rows to a stable CSV manifest."""
    if not rows:
        raise ValueError("Cannot write an empty generation subset manifest.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _read_subset_csv(path: Path) -> list[GenerationSubsetRow]:
    """Read subset rows from a CSV manifest."""
    rows: list[GenerationSubsetRow] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                GenerationSubsetRow(
                    selection_index=int(row["selection_index"]),
                    dataset_index=int(row["dataset_index"]),
                    csd_refcode=str(row["csd_refcode"]).strip().upper(),
                    num_atoms=int(row["num_atoms"]),
                    flexibility=str(row["flexibility"]),
                ),
            )
    return rows


def _validate_subset_rows(
    rows: list[GenerationSubsetRow],
    targets_per_class: int,
    allow_insufficient_targets: bool,
) -> None:
    """Validate fixed subset rows against configured class counts."""
    if not rows:
        raise ValueError("Generation subset manifest is empty.")
    counts = {
        "rigid": sum(row.flexibility == "rigid" for row in rows),
        "flexible": sum(row.flexibility == "flexible" for row in rows),
    }
    if len({row.dataset_index for row in rows}) != len(rows):
        raise ValueError(
            "Generation subset manifest contains duplicate dataset_index values."
        )
    if allow_insufficient_targets:
        if (
            counts["rigid"] > targets_per_class
            or counts["flexible"] > targets_per_class
        ):
            raise ValueError(f"Subset counts exceed requested target count: {counts}.")
        return
    expected = {"rigid": int(targets_per_class), "flexible": int(targets_per_class)}
    if counts != expected:
        raise ValueError(f"Expected subset counts {expected}, found {counts}.")


def _subset_default_path(dataset: MaterialDataset, split: str) -> Path:
    """Return the default persistent manifest path for one split."""
    return (
        dataset.data_dir
        / dataset.dataset_name
        / "generation_subsets"
        / f"{split}_targets.csv"
    )


def _candidate_row(dataset: MaterialDataset, dataset_pos: int) -> GenerationSubsetRow:
    """Build a candidate manifest row from one dataset position."""
    sample = dataset[dataset_pos]
    dataset_index = int(sample["dataset_index"].item())
    metadata = dict(sample.get("metadata", {}))
    refcode = _metadata_refcode(metadata, dataset_index)
    flexibility = _metadata_flexibility(metadata, dataset_index)
    return GenerationSubsetRow(
        selection_index=-1,
        dataset_index=dataset_index,
        csd_refcode=refcode,
        num_atoms=int(sample["num_atoms"].item()),
        flexibility=str(flexibility),
    )


def _select_subset_rows(
    dataset: MaterialDataset,
    targets_per_class: int,
    allow_insufficient_targets: bool,
) -> list[GenerationSubsetRow]:
    """Select a random rigid/flexible-balanced subset from one dataset."""
    order = list(range(len(dataset)))
    random.SystemRandom().shuffle(order)
    selected: dict[str, list[GenerationSubsetRow]] = {"rigid": [], "flexible": []}
    skipped = 0
    for dataset_pos in order:
        if all(len(rows) >= targets_per_class for rows in selected.values()):
            break
        try:
            row = _candidate_row(dataset, int(dataset_pos))
        except Exception as exc:
            skipped += 1
            logger.warning(
                "Skipping generation subset candidate %d: %s", dataset_pos, exc
            )
            continue
        if row.flexibility not in selected:
            skipped += 1
            continue
        if len(selected[row.flexibility]) >= targets_per_class:
            continue
        selected[row.flexibility].append(row)

    if not allow_insufficient_targets:
        missing = {
            label: int(targets_per_class - len(rows))
            for label, rows in selected.items()
            if len(rows) < targets_per_class
        }
        if missing:
            raise ValueError(
                f"Insufficient generation subset targets: missing={missing}."
            )

    rows = selected["rigid"] + selected["flexible"]
    output = [
        GenerationSubsetRow(
            selection_index=index,
            dataset_index=row.dataset_index,
            csd_refcode=row.csd_refcode,
            num_atoms=row.num_atoms,
            flexibility=row.flexibility,
        )
        for index, row in enumerate(rows)
    ]
    logger.info(
        "Selected %d generation targets with %d skipped candidates.",
        len(output),
        skipped,
    )
    return output


def _write_subset_sidecar(
    csv_path: Path,
    rows: list[GenerationSubsetRow],
    dataset: MaterialDataset,
    split: str,
    targets_per_class: int,
) -> None:
    """Write JSON sidecar metadata for a subset manifest."""
    _write_json(
        csv_path.with_suffix(".json"),
        {
            "created_at": _utc_now(),
            "data_dir": str(dataset.data_dir),
            "dataset_name": dataset.dataset_name,
            "split": split,
            "targets_per_class": int(targets_per_class),
            "num_rows": int(len(rows)),
            "rows": [asdict(row) for row in rows],
        },
    )


def _create_subset_manifest(
    dataset: MaterialDataset,
    split: str,
    path: Path,
    targets_per_class: int,
    allow_insufficient_targets: bool,
) -> None:
    """Create a persistent fixed generation subset manifest."""
    rows = _select_subset_rows(
        dataset=dataset,
        targets_per_class=targets_per_class,
        allow_insufficient_targets=allow_insufficient_targets,
    )
    _validate_subset_rows(rows, targets_per_class, allow_insufficient_targets)
    _write_subset_csv(path, rows)
    _write_subset_sidecar(path, rows, dataset, split, targets_per_class)


class FixedSubsetGenerationCallback(Callback):
    """Generate fixed train/validation subset samples and write rank shards."""

    def __init__(
        self,
        split: str,
        artifact_dir_name: str,
        output_dir: Optional[str] = None,
        manifest_path: Optional[str] = None,
        every_n_epochs: int = 100,
        min_epoch: int = 0,
        targets_per_class: int = 100,
        auto_create_manifest: bool = True,
        allow_insufficient_targets: bool = False,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        num_samples: int = 10,
        max_samples_per_call: Optional[int] = None,
        num_steps: int = 20,
        method: str = "ode",
        sde_noise_scale: float = 1.0,
        sampler_args: Optional[dict[str, Any]] = None,
        conditioning_context: Optional[str] = None,
    ) -> None:
        """Initialize fixed-subset generation settings."""
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'.")
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be >= 1.")
        if targets_per_class < 1:
            raise ValueError("targets_per_class must be >= 1.")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        if max_samples_per_call is not None and max_samples_per_call < 1:
            raise ValueError("max_samples_per_call must be >= 1 when provided.")

        self.split = split
        self.artifact_dir_name = artifact_dir_name
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.manifest_path = None if manifest_path is None else Path(manifest_path)
        self.every_n_epochs = int(every_n_epochs)
        self.min_epoch = int(min_epoch)
        self.targets_per_class = int(targets_per_class)
        self.auto_create_manifest = bool(auto_create_manifest)
        self.allow_insufficient_targets = bool(allow_insufficient_targets)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.num_workers = None if num_workers is None else int(num_workers)
        self.num_samples = int(num_samples)
        self.max_samples_per_call = int(max_samples_per_call or num_samples)
        self.num_steps = int(num_steps)
        self.method = str(method)
        self.sde_noise_scale = float(sde_noise_scale)
        self.sampler_args = dict(sampler_args or {})
        self.conditioning_context = str(conditioning_context or f"{split}_generation")

        self._enabled_for_epoch = False
        self._event_dir: Optional[Path] = None
        self._written_files = 0
        self._selected_rows: list[GenerationSubsetRow] = []

    def _completed_epoch(self, trainer: Trainer) -> int:
        """Return the one-based completed epoch counter."""
        return int(trainer.current_epoch) + 1

    def _should_generate_epoch(self, trainer: Trainer) -> bool:
        """Return whether generation should run at this validation point."""
        if bool(getattr(trainer, "sanity_checking", False)):
            return False
        if self._completed_epoch(trainer) < self.min_epoch:
            return False
        return self._completed_epoch(trainer) % self.every_n_epochs == 0

    def _base_output_dir(self, trainer: Trainer) -> Path:
        """Return the root directory for generation artifacts."""
        if self.output_dir is not None:
            return self.output_dir
        return Path(trainer.default_root_dir) / self.artifact_dir_name

    def _event_name(self, trainer: Trainer) -> str:
        """Return a stable directory name for one generation event."""
        return (
            f"epoch_{int(trainer.current_epoch):06d}"
            f"-step_{int(trainer.global_step):012d}"
        )

    def _build_source_dataset(self, trainer: Trainer) -> MaterialDataset:
        """Build an unaugmented metadata-bearing source dataset for generation."""
        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("Generation callbacks require trainer.datamodule.")
        if self.split == "val":
            dataset = getattr(datamodule, "val_dataset", None)
        else:
            dataset = getattr(datamodule, "train_dataset", None)
        if not isinstance(dataset, MaterialDataset):
            raise NotImplementedError(
                f"{self.split}_generation currently supports one CSD dataset."
            )
        return dataset

    def _prepare_dataset_for_generation(
        self,
        dataset: MaterialDataset,
    ) -> dict[str, Any]:
        """Disable mutable train-only dataset behavior during generation."""
        state = {
            "crystal_rotate": dataset.crystal_rotate,
            "crystal_translate": dataset.crystal_translate,
            "template_rotate": dataset.template_rotate,
            "template_translate": dataset.template_translate,
            "template_torsion_perturb": dataset.template_torsion_perturb,
            "template_jitter": dataset.template_jitter,
            "include_metadata": dataset.include_metadata,
        }
        dataset.crystal_rotate = False
        dataset.crystal_translate = False
        dataset.template_rotate = False
        dataset.template_translate = False
        dataset.template_torsion_perturb = False
        dataset.template_jitter = False
        dataset.include_metadata = True
        return state

    def _restore_dataset_after_generation(
        self,
        dataset: MaterialDataset,
        state: dict[str, Any],
    ) -> None:
        """Restore dataset behavior after generation finishes."""
        dataset.crystal_rotate = bool(state["crystal_rotate"])
        dataset.crystal_translate = bool(state["crystal_translate"])
        dataset.template_rotate = bool(state["template_rotate"])
        dataset.template_translate = bool(state["template_translate"])
        dataset.template_torsion_perturb = bool(state["template_torsion_perturb"])
        dataset.template_jitter = bool(state["template_jitter"])
        dataset.include_metadata = bool(state["include_metadata"])

    def _resolved_manifest_path(self, dataset: MaterialDataset) -> Path:
        """Return the configured or default subset manifest path."""
        if self.manifest_path is not None:
            return self.manifest_path
        return _subset_default_path(dataset, self.split)

    def _read_or_create_subset_rows(
        self,
        trainer: Trainer,
        dataset: MaterialDataset,
        manifest_path: Path,
    ) -> list[GenerationSubsetRow]:
        """Read or create the fixed subset manifest for this split."""
        if trainer.is_global_zero and not manifest_path.is_file():
            if not self.auto_create_manifest:
                raise FileNotFoundError(
                    f"Generation subset manifest not found: {manifest_path}"
                )
            _create_subset_manifest(
                dataset=dataset,
                split=self.split,
                path=manifest_path,
                targets_per_class=self.targets_per_class,
                allow_insufficient_targets=self.allow_insufficient_targets,
            )
        _distributed_barrier()
        rows = _read_subset_csv(manifest_path)
        _validate_subset_rows(
            rows=rows,
            targets_per_class=self.targets_per_class,
            allow_insufficient_targets=self.allow_insufficient_targets,
        )
        return rows

    def _subset_positions(
        self,
        dataset: MaterialDataset,
        rows: list[GenerationSubsetRow],
    ) -> list[int]:
        """Map selected LMDB dataset indices to current dataset positions."""
        position_by_index = {
            int(dataset_index): position
            for position, dataset_index in enumerate(dataset.selected_indices)
        }
        missing = [
            row.dataset_index
            for row in rows
            if int(row.dataset_index) not in position_by_index
        ]
        if missing:
            raise ValueError(
                f"Generation subset manifest contains rows outside split={self.split}: "
                f"{missing[:10]}",
            )
        return [position_by_index[int(row.dataset_index)] for row in rows]

    def _build_subset_dataloader(
        self,
        trainer: Trainer,
    ) -> tuple[DataLoader, Path, list[GenerationSubsetRow]]:
        """Build the rank-local fixed-subset generation dataloader."""
        dataset = self._build_source_dataset(trainer)
        manifest_path = self._resolved_manifest_path(dataset)
        rows = self._read_or_create_subset_rows(trainer, dataset, manifest_path)
        positions = self._subset_positions(dataset, rows)
        subset = Subset(dataset, positions)
        world_size = int(getattr(trainer, "world_size", 1))
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                subset,
                num_replicas=world_size,
                rank=int(getattr(trainer, "global_rank", 0)),
                shuffle=False,
                drop_last=False,
            )
        batch_size = int(
            self.batch_size or getattr(trainer.datamodule.hparams, "batch_size", 1)
        )
        num_workers = int(self.num_workers if self.num_workers is not None else 0)
        conditioning_policy = getattr(
            trainer.datamodule.hparams,
            "conditioning_policy",
            None,
        )
        collate = (
            partial(
                collate_fn,
                conditioning_policy=conditioning_policy,
                conditioning_context=self.conditioning_context,
            )
            if conditioning_policy is not None
            else collate_fn
        )
        dataloader = DataLoader(
            dataset=subset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
            collate_fn=collate,
        )
        return dataloader, manifest_path, rows

    def _write_manifest(
        self,
        trainer: Trainer,
        subset_manifest_path: Path,
        subset_rows: list[GenerationSubsetRow],
    ) -> None:
        """Write event-level metadata on global rank zero."""
        if self._event_dir is None or not trainer.is_global_zero:
            return

        world_size = int(getattr(trainer, "world_size", 1))
        payload = {
            "epoch": int(trainer.current_epoch),
            "completed_epoch": self._completed_epoch(trainer),
            "global_step": int(trainer.global_step),
            "split": self.split,
            "world_size": world_size,
            "expected_ranks": list(range(world_size)),
            "generation_subset": {
                "manifest_path": str(subset_manifest_path),
                "targets_per_class": self.targets_per_class,
                "num_targets": int(len(subset_rows)),
                "dataset_indices": [int(row.dataset_index) for row in subset_rows],
            },
            "sampling": {
                "num_samples": self.num_samples,
                "max_samples_per_call": self.max_samples_per_call,
                "num_steps": self.num_steps,
                "method": self.method,
                "sde_noise_scale": self.sde_noise_scale,
                "sampler_args": self.sampler_args,
                "conditioning_context": self.conditioning_context,
            },
            "created_at": _utc_now(),
        }
        _write_json(self._event_dir / "manifest.json", payload)

    def _device_generation_inputs(
        self,
        batch: dict[str, Any],
        pl_module: MaterialFlowModule,
    ) -> dict[str, Any]:
        """Move model sampling inputs to the module device."""
        return {
            "num_atoms": batch["num_atoms"].to(pl_module.device),
            "conditioning": {
                key: value.to(pl_module.device)
                for key, value in batch["conditioning"].items()
            },
        }

    def _dataset_indices(self, batch: dict[str, Any]) -> Int["b"]:
        """Return stable dataset row ids for artifact collation."""
        if "dataset_index" not in batch:
            raise KeyError(f"{type(self).__name__} requires batch['dataset_index'].")
        return batch["dataset_index"].detach().cpu().to(dtype=torch.long)

    def _metadata_rows(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        """Return one metadata dictionary per source row."""
        batch_size = int(batch["num_atoms"].shape[0])
        rows = batch.get("metadata")
        if rows is None:
            metadata_rows = [{} for _ in range(batch_size)]
        else:
            metadata_rows = [dict(row) for row in rows]

        if "dataset_index" not in batch or not self._selected_rows:
            return metadata_rows
        subset_by_index = {int(row.dataset_index): row for row in self._selected_rows}
        dataset_indices = batch["dataset_index"].detach().cpu().to(dtype=torch.long)
        for row_idx, dataset_index in enumerate(dataset_indices.tolist()):
            subset_row = subset_by_index.get(int(dataset_index))
            if subset_row is None:
                continue
            metadata_rows[row_idx].update(
                {
                    "selection_index": int(subset_row.selection_index),
                    "flexibility": str(subset_row.flexibility),
                },
            )
        return metadata_rows

    def _reference_lattice(
        self,
        batch: dict[str, Any],
        pred_lattice: Float["b d"],
    ) -> Float["b d"]:
        """Select reference lattice representation matching generated samples."""
        if int(pred_lattice.shape[-1]) == 9:
            return rearrange(batch["cell"], "b i j -> b (i j)")
        return batch["lattice"]

    def _reference_payload(
        self,
        batch: dict[str, Any],
        pred_lattice: Float["b d"],
    ) -> dict[str, Any]:
        """Build CPU reference tensors aligned with generated samples."""
        ref_lattice = self._reference_lattice(batch, pred_lattice)
        payload: dict[str, Any] = {
            "cart_coords": batch["cart_coords"].detach().cpu(),
            "lattice": ref_lattice.detach().cpu(),
            "cell": batch["cell"].detach().cpu(),
            "atomic_numbers": batch["conditioning"]["atomic_numbers"].detach().cpu(),
            "atom_mask": batch["atom_mask"].detach().cpu(),
        }
        bond_adj = batch["conditioning"].get("bond_adj")
        if bond_adj is not None:
            payload["bond_adj"] = bond_adj.detach().cpu()
        return payload

    def _write_sample_chunk(
        self,
        trainer: Trainer,
        batch_idx: int,
        sample_offset: int,
        chunk_size: int,
        dataset_indices: Int["b"],
        metadata_rows: list[dict[str, Any]],
        pred: dict[str, Any],
        ref: dict[str, Any],
    ) -> None:
        """Write one generated chunk to this rank's raw directory."""
        if self._event_dir is None:
            raise RuntimeError("Generation event directory is not initialized.")

        rank = int(getattr(trainer, "global_rank", 0))
        raw_dir = self._event_dir / "raw" / f"rank_{rank}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "rank": rank,
            "split": self.split,
            "batch_idx": int(batch_idx),
            "sample_offset": int(sample_offset),
            "chunk_size": int(chunk_size),
            "pred": _cpu_tensor_dict(pred),
            "ref": ref,
            "dataset_index": dataset_indices,
            "metadata": metadata_rows,
        }
        output_path = raw_dir / (
            f"batch_{int(batch_idx):06d}_sample_{int(sample_offset):06d}.pt"
        )
        torch.save(payload, output_path)
        self._written_files += 1

    def _sample_batch(
        self,
        trainer: Trainer,
        batch: dict[str, Any],
        pl_module: MaterialFlowModule,
        batch_idx: int,
    ) -> None:
        """Generate samples for one subset batch and write raw chunks."""
        device_batch = self._device_generation_inputs(batch, pl_module)
        dataset_indices = self._dataset_indices(batch)
        metadata_rows = self._metadata_rows(batch)

        for sample_offset in range(0, self.num_samples, self.max_samples_per_call):
            chunk_size = min(
                self.max_samples_per_call, self.num_samples - sample_offset
            )
            with torch.no_grad():
                pred = pl_module.flow_matching.sample(
                    num_atoms=device_batch["num_atoms"],
                    multiplicity=chunk_size,
                    num_steps=self.num_steps,
                    method=self.method,
                    sde_noise_scale=self.sde_noise_scale,
                    conditioning=device_batch["conditioning"],
                    conditioning_context=self.conditioning_context,
                    sampler_args=self.sampler_args,
                )

            ref = self._reference_payload(batch, pred["lattice"])
            self._write_sample_chunk(
                trainer=trainer,
                batch_idx=batch_idx,
                sample_offset=sample_offset,
                chunk_size=chunk_size,
                dataset_indices=dataset_indices,
                metadata_rows=metadata_rows,
                pred=pred,
                ref=ref,
            )

    def _write_done_marker(self, trainer: Trainer) -> None:
        """Write this rank's completion marker after generation."""
        if self._event_dir is None:
            return
        rank = int(getattr(trainer, "global_rank", 0))
        payload = {
            "rank": rank,
            "split": self.split,
            "epoch": int(trainer.current_epoch),
            "completed_epoch": self._completed_epoch(trainer),
            "global_step": int(trainer.global_step),
            "num_files": int(self._written_files),
            "completed_at": _utc_now(),
        }
        _write_json(self._event_dir / "done" / f"rank_{rank}.json", payload)

    def _run_generation(
        self,
        trainer: Trainer,
        pl_module: MaterialFlowModule,
    ) -> None:
        """Run fixed-subset generation for the current validation point."""
        dataset = self._build_source_dataset(trainer)
        state = self._prepare_dataset_for_generation(dataset)
        try:
            dataloader, manifest_path, subset_rows = self._build_subset_dataloader(
                trainer
            )
            self._selected_rows = subset_rows
            self._event_dir = self._base_output_dir(trainer) / self._event_name(trainer)
            self._event_dir.mkdir(parents=True, exist_ok=True)
            self._write_manifest(trainer, manifest_path, subset_rows)
            for batch_idx, batch in enumerate(dataloader):
                self._sample_batch(trainer, batch, pl_module, int(batch_idx))
            self._write_done_marker(trainer)
        finally:
            self._restore_dataset_after_generation(dataset, state)

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: MaterialFlowModule,
    ) -> None:
        """Reset per-event state before validation starts."""
        self._enabled_for_epoch = self._should_generate_epoch(trainer)
        self._event_dir = None
        self._written_files = 0
        self._selected_rows = []

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: MaterialFlowModule,
    ) -> None:
        """Generate fixed-subset samples after validation loss is computed."""
        if not self._enabled_for_epoch:
            return
        self._run_generation(trainer, pl_module)
        _distributed_barrier()


class ValidationGenerationCallback(FixedSubsetGenerationCallback):
    """Generate validation subset samples and write rank-local shard files."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize validation fixed-subset generation."""
        super().__init__(
            split="val", artifact_dir_name="validation_predictions", **kwargs
        )


class TrainingGenerationCallback(FixedSubsetGenerationCallback):
    """Generate training subset samples and write rank-local shard files."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize training fixed-subset generation."""
        super().__init__(split="train", artifact_dir_name="train_predictions", **kwargs)
