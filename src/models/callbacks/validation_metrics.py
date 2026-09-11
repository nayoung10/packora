"""Callback for computing OXtal validation metrics during validation."""

# ruff: noqa: F722,F821

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from ase.data import chemical_symbols
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from einops import rearrange
from lightning import Callback, Trainer
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.models.flow_module import MaterialFlowModule
from src.utils.structure_io import tensors_to_atoms
from src.utils.tensor_typing import Bool, Float, Int

logger = logging.getLogger(__name__)

ADAPTOR = AseAtomsAdaptor()


@dataclass
class SampleChunk:
    """Container for one sampled validation chunk."""

    rank: int
    sample_offset: int
    chunk_size: int
    dataset_indices: Int["b"]
    metadata_rows: list[dict[str, Any]]
    pred: dict[str, Any]
    ref: dict[str, Any]


@dataclass
class StructurePayload:
    """Container for one unbatched structure tensor payload."""

    coords: Float["n 3"]
    lattice: Float["d"]
    atomic_numbers: Int["n"]
    atom_mask: Bool["n"]
    bond_adj: Optional[Bool["n n"]]
    identifier: str


@dataclass
class TargetAccumulator:
    """Container for one validation target and its generated samples."""

    csd_refcode: str
    target: StructurePayload
    samples: dict[int, StructurePayload] = field(default_factory=dict)


def _safe_identifier(value: object) -> str:
    """Return a CIF-safe identifier string."""
    text = str(value)
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return cleaned or "structure"


def _format_cif_float(value: float) -> str:
    """Return a compact finite CIF numeric value."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError("CIF values must be finite.")
    return f"{scalar:.8f}"


def _cell_from_lattice(lattice: Float["d"]) -> np.ndarray:
    """Return a 3x3 cell matrix from 6D parameters or a flattened cell."""
    lattice_cpu = lattice.detach().cpu().to(dtype=torch.float64)
    if int(lattice_cpu.numel()) == 9:
        return rearrange(lattice_cpu, "(i j) -> i j", i=3, j=3).numpy()
    if int(lattice_cpu.numel()) == 6:
        return cellpar_to_cell(lattice_cpu.numpy()).astype(np.float64)
    raise ValueError(
        f"Expected 6D or 9D lattice, got {int(lattice_cpu.numel())} values."
    )


def _real_atom_indices(atom_mask: Bool["n"]) -> list[int]:
    """Return unpadded atom indices from a mask."""
    mask_cpu = atom_mask.detach().cpu().to(dtype=torch.bool)
    return [int(index) for index in torch.nonzero(mask_cpu, as_tuple=True)[0].tolist()]


def _atom_labels(atomic_numbers: Int["n"], real_indices: list[int]) -> list[str]:
    """Return unique CIF atom labels for real atoms."""
    counts: dict[str, int] = {}
    numbers_cpu = atomic_numbers.detach().cpu().to(dtype=torch.long)
    labels: list[str] = []
    for index in real_indices:
        symbol = chemical_symbols[int(numbers_cpu[index].item())]
        counts[symbol] = counts.get(symbol, 0) + 1
        labels.append(f"{symbol}{counts[symbol]}")
    return labels


def _bond_pairs(
    bond_adj: Optional[Bool["n n"]],
    real_indices: list[int],
) -> list[tuple[int, int]]:
    """Return unique bond pairs in real-atom label coordinates."""
    if bond_adj is None:
        return []

    bond_cpu = bond_adj.detach().cpu().to(dtype=torch.bool)
    pairs: list[tuple[int, int]] = []
    for left_pos, left_index in enumerate(real_indices):
        for right_pos in range(left_pos + 1, len(real_indices)):
            right_index = real_indices[right_pos]
            if bool(bond_cpu[left_index, right_index]) or bool(
                bond_cpu[right_index, left_index]
            ):
                pairs.append((left_pos, right_pos))
    return pairs


def _build_cif(payload: StructurePayload) -> str:
    """Build an essential CIF containing cell, atoms, and intramolecular bonds."""
    real_indices = _real_atom_indices(payload.atom_mask)
    if not real_indices:
        raise ValueError("Cannot build a CIF with zero atoms.")

    cell = _cell_from_lattice(payload.lattice)
    cell_params = cell_to_cellpar(cell).astype(np.float64)
    coords = payload.coords.detach().cpu().to(dtype=torch.float64)
    coords_np = coords[real_indices].numpy()
    frac_coords = np.linalg.solve(cell.T, coords_np.T).T

    labels = _atom_labels(payload.atomic_numbers, real_indices)
    numbers = payload.atomic_numbers.detach().cpu().to(dtype=torch.long)
    bond_pairs = _bond_pairs(payload.bond_adj, real_indices)
    identifier = _safe_identifier(payload.identifier)

    lines = [
        f"data_{identifier}",
        "_symmetry_cell_setting           triclinic",
        "_symmetry_space_group_name_H-M   'P 1'",
        "_symmetry_Int_Tables_number      1",
        "_space_group_name_Hall           'P 1'",
        "loop_",
        "_symmetry_equiv_pos_site_id",
        "_symmetry_equiv_pos_as_xyz",
        "1 x,y,z",
        f"_cell_length_a                   {_format_cif_float(cell_params[0])}",
        f"_cell_length_b                   {_format_cif_float(cell_params[1])}",
        f"_cell_length_c                   {_format_cif_float(cell_params[2])}",
        f"_cell_angle_alpha                {_format_cif_float(cell_params[3])}",
        f"_cell_angle_beta                 {_format_cif_float(cell_params[4])}",
        f"_cell_angle_gamma                {_format_cif_float(cell_params[5])}",
        f"_cell_volume                     {_format_cif_float(abs(float(np.linalg.det(cell))))}",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]

    for atom_pos, atom_index in enumerate(real_indices):
        symbol = chemical_symbols[int(numbers[atom_index].item())]
        frac = frac_coords[atom_pos]
        lines.append(
            " ".join(
                [
                    labels[atom_pos],
                    symbol,
                    _format_cif_float(frac[0]),
                    _format_cif_float(frac[1]),
                    _format_cif_float(frac[2]),
                ],
            ),
        )

    if bond_pairs:
        lines.extend(
            [
                "loop_",
                "_geom_bond_atom_site_label_1",
                "_geom_bond_atom_site_label_2",
                "_geom_bond_site_symmetry_1",
                "_geom_bond_site_symmetry_2",
            ],
        )
        for left_pos, right_pos in bond_pairs:
            lines.append(f"{labels[left_pos]} {labels[right_pos]} 1_555 1_555")

    return "\n".join(lines) + "\n"


def _payload_to_crystal(payload: StructurePayload) -> Any:
    """Convert one tensor payload to a CrystalReader-loaded CCDC Crystal."""
    from eval.oxtal import load_crystal_from_cif_text

    identifier = _safe_identifier(payload.identifier)
    return load_crystal_from_cif_text(_build_cif(payload), identifier=identifier)


def _payload_to_structure(payload: StructurePayload) -> Structure:
    """Convert one tensor payload to a pymatgen Structure."""
    atoms = tensors_to_atoms(
        payload.coords,
        payload.lattice,
        payload.atomic_numbers,
        payload.atom_mask,
    )
    return ADAPTOR.get_structure(atoms)


class ValidationMetricsCallback(Callback):
    """Generate validation samples and compute OXtal metrics."""

    def __init__(
        self,
        every_n_epochs: int = 1,
        min_epoch: int = 0,
        num_samples: int = 30,
        max_samples_per_call: Optional[int] = None,
        num_steps: int = 100,
        method: str = "ode",
        sde_noise_scale: float = 1.0,
        ltol: float = 0.3,
        stol: float = 0.5,
        angle_tol: float = 10.0,
        surrogate_alpha: float = 0.75,
        **kwargs: Any,
    ) -> None:
        """Initialize validation metric callback configuration."""
        super().__init__()
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        if max_samples_per_call is not None and max_samples_per_call < 1:
            raise ValueError("max_samples_per_call must be >= 1 when provided.")

        self.every_n_epochs = int(every_n_epochs)
        self.min_epoch = int(min_epoch)
        self.num_samples = int(num_samples)
        self.max_samples_per_call = int(max_samples_per_call or num_samples)
        self.num_steps = int(num_steps)
        self.method = str(method)
        self.sde_noise_scale = float(sde_noise_scale)
        self.ltol = float(ltol)
        self.stol = float(stol)
        self.angle_tol = float(angle_tol)
        self.surrogate_alpha = float(surrogate_alpha)

        self._enabled_for_epoch = False
        self._local_chunks: list[SampleChunk] = []

    def _should_evaluate_epoch(self, trainer: Trainer) -> bool:
        """Return whether this validation epoch should run metric sampling."""
        if bool(getattr(trainer, "sanity_checking", False)):
            return False
        if int(trainer.current_epoch) < self.min_epoch:
            return False
        return int(trainer.current_epoch) % self.every_n_epochs == 0

    def _reset_epoch_state(self) -> None:
        """Clear per-epoch sampled payload state."""
        self._local_chunks = []

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
        """Return stable dataset row ids for distributed aggregation."""
        if "dataset_index" not in batch:
            raise KeyError("ValidationMetricsCallback requires batch['dataset_index'].")
        return batch["dataset_index"].detach().cpu().to(dtype=torch.long)

    def _metadata_rows(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        """Return one metadata dictionary per source row."""
        batch_size = int(batch["num_atoms"].shape[0])
        rows = batch.get("metadata")
        if rows is None:
            return [{} for _ in range(batch_size)]
        return [dict(row) for row in rows]

    def _csd_refcode(self, metadata: dict[str, Any], dataset_index: int) -> str:
        """Return the required CSD refcode for one validation row."""
        value = metadata.get("csd_refcode")
        refcode = "" if value is None else str(value).strip().upper()
        if not refcode:
            raise ValueError(
                f"Missing csd_refcode metadata for dataset_index={dataset_index}."
            )
        return refcode

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
        return {
            "cart_coords": batch["cart_coords"].detach().cpu(),
            "lattice": ref_lattice.detach().cpu(),
            "atomic_numbers": batch["conditioning"]["atomic_numbers"].detach().cpu(),
            "atom_mask": batch["atom_mask"].detach().cpu(),
            "bond_adj": batch["conditioning"]["bond_adj"].detach().cpu(),
        }

    def _prediction_payload(self, pred: dict[str, Any]) -> dict[str, Any]:
        """Move generated prediction tensors to CPU."""
        return {
            "cart_coords": pred["cart_coords"].detach().cpu(),
            "lattice": pred["lattice"].detach().cpu(),
            "atomic_numbers": pred["atomic_numbers"].detach().cpu(),
            "atom_mask": pred["atom_mask"].detach().cpu(),
        }

    def _sample_batch(
        self,
        batch: dict[str, Any],
        pl_module: MaterialFlowModule,
        rank: int,
    ) -> None:
        """Generate validation samples for one validation batch."""
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
                )

            self._local_chunks.append(
                SampleChunk(
                    rank=int(rank),
                    sample_offset=int(sample_offset),
                    chunk_size=int(chunk_size),
                    dataset_indices=dataset_indices,
                    metadata_rows=metadata_rows,
                    pred=self._prediction_payload(pred),
                    ref=self._reference_payload(batch, pred["lattice"]),
                ),
            )

    def _gather_chunks(self, trainer: Trainer) -> list[SampleChunk]:
        """Gather per-rank sampled chunks to global zero."""
        if int(getattr(trainer, "world_size", 1)) <= 1:
            return list(self._local_chunks)

        if not dist.is_available() or not dist.is_initialized():
            logger.warning(
                "Distributed metrics requested, but torch.distributed is unavailable."
            )
            return list(self._local_chunks) if trainer.is_global_zero else []

        gathered: Optional[list[Optional[list[SampleChunk]]]]
        gathered = (
            [None for _ in range(int(trainer.world_size))]
            if trainer.is_global_zero
            else None
        )
        dist.gather_object(
            self._local_chunks,
            object_gather_list=gathered,
            dst=0,
        )

        if not trainer.is_global_zero or gathered is None:
            return []

        chunks: list[SampleChunk] = []
        for rank_chunks in gathered:
            if rank_chunks:
                chunks.extend(rank_chunks)
        return chunks

    def _assemble_targets(
        self,
        chunks: list[SampleChunk],
    ) -> list[TargetAccumulator]:
        """Group gathered prediction chunks by validation target."""
        targets: dict[int, TargetAccumulator] = {}

        for chunk in chunks:
            batch_size = int(chunk.dataset_indices.shape[0])
            for source_idx in range(batch_size):
                target_key = int(chunk.dataset_indices[source_idx].item())
                metadata = (
                    chunk.metadata_rows[source_idx]
                    if source_idx < len(chunk.metadata_rows)
                    else {}
                )
                identifier = self._csd_refcode(metadata, target_key)
                if target_key not in targets:
                    targets[target_key] = TargetAccumulator(
                        csd_refcode=identifier,
                        target=StructurePayload(
                            coords=chunk.ref["cart_coords"][source_idx],
                            lattice=chunk.ref["lattice"][source_idx],
                            atomic_numbers=chunk.ref["atomic_numbers"][source_idx],
                            atom_mask=chunk.ref["atom_mask"][source_idx],
                            bond_adj=chunk.ref["bond_adj"][source_idx],
                            identifier=identifier,
                        ),
                    )

                for sample_idx in range(chunk.chunk_size):
                    global_sample_idx = int(chunk.sample_offset + sample_idx)
                    pred_row = source_idx * chunk.chunk_size + sample_idx
                    targets[target_key].samples.setdefault(
                        global_sample_idx,
                        StructurePayload(
                            coords=chunk.pred["cart_coords"][pred_row],
                            lattice=chunk.pred["lattice"][pred_row],
                            atomic_numbers=chunk.pred["atomic_numbers"][pred_row],
                            atom_mask=chunk.pred["atom_mask"][pred_row],
                            bond_adj=chunk.ref["bond_adj"][source_idx],
                            identifier=f"{identifier}_sample_{global_sample_idx}",
                        ),
                    )

        return [targets[key] for key in sorted(targets)]

    def _build_oxtal_inputs(
        self,
        targets: list[TargetAccumulator],
    ) -> tuple[list[Any], list[list[Any]], list[str]]:
        """Convert grouped tensor payloads to CCDC OXtal inputs."""
        target_crystals: list[Any] = []
        samples_by_target: list[list[Any]] = []
        target_refcodes: list[str] = []

        for target_index, target in enumerate(targets):
            try:
                target_crystal = _payload_to_crystal(target.target)
            except Exception as exc:
                logger.warning(
                    "Skipping OXtal target %d conversion failure: %s", target_index, exc
                )
                continue

            sample_crystals: list[Any] = []
            for sample_index in sorted(target.samples):
                try:
                    sample_crystals.append(
                        _payload_to_crystal(target.samples[sample_index])
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping OXtal sample target=%d sample=%d conversion failure: %s",
                        target_index,
                        sample_index,
                        exc,
                    )

            if sample_crystals:
                target_crystals.append(target_crystal)
                samples_by_target.append(sample_crystals)
                target_refcodes.append(target.csd_refcode)

        return target_crystals, samples_by_target, target_refcodes

    def _build_surrogate_inputs(
        self,
        targets: list[TargetAccumulator],
    ) -> tuple[list[Structure], list[list[Structure]]]:
        """Convert grouped tensor payloads to pymatgen surrogate inputs."""
        target_structures: list[Structure] = []
        samples_by_target: list[list[Structure]] = []

        for target_index, target in enumerate(targets):
            try:
                target_structure = _payload_to_structure(target.target)
            except Exception as exc:
                logger.warning(
                    "Skipping surrogate target %d conversion failure: %s",
                    target_index,
                    exc,
                )
                continue

            sample_structures: list[Structure] = []
            for sample_index in sorted(target.samples):
                try:
                    sample_structures.append(
                        _payload_to_structure(target.samples[sample_index])
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping surrogate sample target=%d sample=%d conversion failure: %s",
                        target_index,
                        sample_index,
                        exc,
                    )

            if sample_structures:
                target_structures.append(target_structure)
                samples_by_target.append(sample_structures)

        return target_structures, samples_by_target

    def _log_metrics(
        self,
        pl_module: MaterialFlowModule,
        metrics: dict[str, object],
    ) -> None:
        """Log scalar validation metrics through Lightning."""
        for key, value in metrics.items():
            pl_module.log(
                f"val_metrics/{key}",
                float(value),
                rank_zero_only=True,
                sync_dist=False,
            )

    def _evaluate_oxtal(
        self,
        targets: list[TargetAccumulator],
        pl_module: MaterialFlowModule,
    ) -> None:
        """Evaluate and log CCDC OXtal metrics."""
        start = perf_counter()
        target_crystals, samples_by_target, target_refcodes = self._build_oxtal_inputs(
            targets
        )
        if not target_crystals:
            logger.warning(
                "Skipping OXtal metrics because no CCDC Crystal inputs were valid."
            )
            return

        from eval.oxtal import evaluate as evaluate_oxtal

        result = evaluate_oxtal(
            target_crystals,
            samples_by_target,
            target_refcodes=target_refcodes,
        )
        elapsed = perf_counter() - start
        self._log_metrics(pl_module, result["summary"])
        self._log_metrics(pl_module, {"oxtal_time_seconds": elapsed})
        logger.info("OXtal validation metrics: %s", result["summary"])

    def _evaluate_surrogates(
        self,
        targets: list[TargetAccumulator],
        pl_module: MaterialFlowModule,
    ) -> None:
        """Evaluate and log StructureMatcher surrogate metrics."""
        start = perf_counter()
        target_structures, samples_by_target = self._build_surrogate_inputs(targets)
        if not target_structures:
            logger.warning(
                "Skipping surrogate metrics because no pymatgen inputs were valid."
            )
            return

        from eval.oxtal_surrogates import evaluate as evaluate_surrogates
        from eval.oxtal_surrogates import make_structure_matcher

        matcher = make_structure_matcher(
            stol=self.stol,
            ltol=self.ltol,
            angle_tol=self.angle_tol,
        )
        result = evaluate_surrogates(
            target_structures,
            samples_by_target,
            alpha=self.surrogate_alpha,
            matcher=matcher,
        )
        elapsed = perf_counter() - start
        self._log_metrics(pl_module, result["summary"])
        self._log_metrics(pl_module, {"surrogate_time_seconds": elapsed})
        logger.info("Surrogate validation metrics: %s", result["summary"])

    def _evaluate_metrics(
        self,
        chunks: list[SampleChunk],
        pl_module: MaterialFlowModule,
    ) -> None:
        """Assemble, evaluate, and log validation metrics on global zero."""
        targets = self._assemble_targets(chunks)
        if not targets:
            logger.warning(
                "Validation metrics skipped because no samples were collected."
            )
            return

        logger.info(
            "Computing validation metrics for %d targets with %d samples each.",
            len(targets),
            self.num_samples,
        )
        self._evaluate_oxtal(targets, pl_module)
        self._evaluate_surrogates(targets, pl_module)

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: MaterialFlowModule,
    ) -> None:
        """Prepare per-epoch callback state before validation starts."""
        self._enabled_for_epoch = self._should_evaluate_epoch(trainer)
        self._reset_epoch_state()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: MaterialFlowModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Sample generated structures for one validation batch."""
        if not self._enabled_for_epoch:
            return
        self._sample_batch(batch, pl_module, int(getattr(trainer, "global_rank", 0)))

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: MaterialFlowModule,
    ) -> None:
        """Gather generated samples and evaluate metrics on global zero."""
        if not self._enabled_for_epoch:
            return

        chunks = self._gather_chunks(trainer)
        if trainer.is_global_zero:
            self._evaluate_metrics(chunks, pl_module)

        if int(getattr(trainer, "world_size", 1)) > 1:
            trainer.strategy.barrier()

        self._reset_epoch_state()
