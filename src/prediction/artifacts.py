"""Prediction artifact collation and manifest helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.prediction.config import (
    SourceSpec,
    resolve_conditioning_context,
    resolve_conditioning_policy,
    resolve_max_num_atoms,
    resolve_model_overrides,
)
from src.prediction.io import PredictionBundle, expand_prediction_rows
from src.prediction.sampling import (
    effective_spacegroup_cfg_weight,
    effective_sampling_method,
    sample_chunk_size,
    sampler_args_dict,
    samples_per_datapoint,
    spacegroup_cfg_weight,
    steering_args_dict,
    time_epsilon,
)
from src.utils.pylogger import RankedLogger
from src.utils.tensor_typing import Int

log = RankedLogger(__name__, rank_zero_only=True)


def to_container_or_none(value: Any) -> Any | None:
    """Convert an optional OmegaConf node to plain containers."""
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True)


def collect_raw_payloads(
    output_dir: Path,
) -> tuple[
    list[dict[str, torch.Tensor]],
    list[dict[str, torch.Tensor]],
    list[Int["b"]],  # noqa: F821
    list[Int["b"]],  # noqa: F821
    list[dict[str, Any]],
]:
    """Load per-rank batch payloads written by PredictionBundleWriter."""
    payload_paths = sorted((output_dir / "raw").glob("rank_*/batch_*.pt"))
    if not payload_paths:
        raise FileNotFoundError(
            f"No prediction payload files found in {(output_dir / 'raw')}"
        )

    pred_batches: list[dict[str, torch.Tensor]] = []
    ref_batches: list[dict[str, torch.Tensor]] = []
    dataset_index_batches: list[Int["b"]] = []  # noqa: F821
    sample_index_batches: list[Int["b"]] = []  # noqa: F821
    metadata_rows: list[dict[str, Any]] = []

    for path in payload_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        (
            pred_batch,
            ref_batch,
            dataset_index_batch,
            sample_index_batch,
            metadata_batch,
        ) = expand_prediction_rows(
            pred=payload["pred"],
            ref=payload["ref"],
            dataset_index=payload["dataset_index"],
            metadata_rows=list(payload.get("metadata", [])),
            sample_offset=int(payload["sample_offset"]),
        )
        pred_batches.append(pred_batch)
        ref_batches.append(ref_batch)
        dataset_index_batches.append(dataset_index_batch)
        sample_index_batches.append(sample_index_batch)
        metadata_rows.extend(metadata_batch)

    return (
        pred_batches,
        ref_batches,
        dataset_index_batches,
        sample_index_batches,
        metadata_rows,
    )


def write_manifest(
    output_dir: Path,
    cfg: DictConfig,
    run_cfg: DictConfig,
    bundle: PredictionBundle,
    source: SourceSpec,
    elapsed_seconds: float,
) -> None:
    """Write a lightweight run manifest for traceability."""
    max_num_atoms = resolve_max_num_atoms(cfg, run_cfg)
    payload = {
        "ckpt_path": str(Path(cfg.ckpt_path).resolve()),
        "source": {
            "data_dir": source.data_dir,
            "dataset_name": source.dataset_name,
            "split": source.split,
            "benchmark": source.benchmark,
            "max_num_atoms": max_num_atoms,
        },
        "sampling": {
            "method": effective_sampling_method(cfg),
            "requested_method": cfg.sampling.method,
            "sde_noise_scale": float(cfg.sampling.sde_noise_scale),
            "sampler_args": sampler_args_dict(cfg),
            "conditioning_context": resolve_conditioning_context(cfg, run_cfg),
            "conditioning_policy": to_container_or_none(
                resolve_conditioning_policy(cfg, run_cfg)
            ),
            "num_steps": int(cfg.sampling.num_steps),
            "batch_size": int(cfg.sampling.batch_size),
            "samples_per_datapoint": samples_per_datapoint(cfg),
            "sample_chunk_size": sample_chunk_size(cfg),
            "time_epsilon": time_epsilon(cfg),
            "batch_sampler": to_container_or_none(cfg.sampling.get("batch_sampler")),
            "num_prediction_rows": int(bundle.dataset_indices.shape[0]),
            "steering_args": steering_args_dict(cfg),
            "spacegroup_cfg": {
                "weight": spacegroup_cfg_weight(cfg),
                "effective_weight": effective_spacegroup_cfg_weight(cfg, run_cfg),
            },
        },
        "output": {
            "metadata_fields": sorted(bundle.metadata.keys()),
            "keep_raw": bool(cfg.output.get("keep_raw", False)),
        },
        "eval_with_ema": bool(cfg.get("eval_with_ema", True)),
        "model_overrides": to_container_or_none(cfg.get("model_overrides")),
        "resolved_model_overrides": to_container_or_none(resolve_model_overrides(cfg)),
        "elapsed_seconds": elapsed_seconds,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def cleanup_raw_payloads(output_dir: Path, cfg: DictConfig) -> None:
    """Remove raw rank payloads after successful final artifact writes."""
    if bool(cfg.output.get("keep_raw", False)):
        return
    raw_dir = output_dir / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
        log.info(f"Removed raw prediction payloads from {raw_dir}")
