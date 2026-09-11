"""Prediction checkpoint loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.models.flow_module import MaterialFlowModule
from src.utils.checkpoint_state import remap_state_dict_keys
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

CheckpointWeightSource = Literal["raw", "ema"]


def select_checkpoint_weights(
    checkpoint: Mapping[str, Any],
    source: CheckpointWeightSource,
) -> Mapping[str, Any]:
    """Return raw or EMA weights from a training checkpoint."""
    if source == "raw":
        weights = checkpoint.get("state_dict")
    elif source == "ema":
        ema_state = checkpoint.get("ema")
        if not isinstance(ema_state, Mapping):
            raise KeyError("EMA weights requested, but checkpoint has no EMA state.")
        weights = ema_state.get("ema_weights")
    else:
        raise ValueError(
            f"Unsupported checkpoint weight source {source!r}; expected 'raw' or 'ema'."
        )

    if not isinstance(weights, Mapping):
        raise KeyError(f"Checkpoint is missing {source} model weights.")
    return weights


def resolve_run_config_path(
    ckpt_path: Path,
    run_config_path: Path | None = None,
) -> Path:
    """Resolve the Hydra run config associated with a checkpoint."""
    resolved_path = (
        run_config_path
        if run_config_path is not None
        else ckpt_path.parent.parent / ".hydra" / "config.yaml"
    )
    resolved_path = resolved_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Run config not found at {resolved_path}. "
            "Keep the checkpoint under <run_dir>/checkpoints/ or provide an "
            "explicit run config path."
        )
    return resolved_path


def load_run_config(
    ckpt_path: Path,
    run_config_path: Path | None = None,
) -> tuple[DictConfig, Path]:
    """Load the Hydra run config associated with a checkpoint."""
    resolved_path = resolve_run_config_path(ckpt_path, run_config_path)
    return OmegaConf.load(resolved_path), resolved_path


def resolve_model_config(
    run_cfg: DictConfig,
    model_overrides: DictConfig | None = None,
) -> DictConfig:
    """Return a standalone resolved model config with optional overrides."""
    model_cfg = OmegaConf.create(
        OmegaConf.to_container(run_cfg.model, resolve=True),
    )
    if "center_cart_coords" not in model_cfg:
        model_cfg.center_cart_coords = True
    if model_overrides is not None and len(model_overrides) > 0:
        model_cfg = OmegaConf.merge(model_cfg, model_overrides)
    return model_cfg


def instantiate_model_from_run_config(
    run_cfg: DictConfig,
    model_overrides: DictConfig | None = None,
) -> tuple[MaterialFlowModule, DictConfig]:
    """Instantiate a model and return its effective resolved config."""
    model_cfg = resolve_model_config(run_cfg, model_overrides)
    model: MaterialFlowModule = hydra.utils.instantiate(model_cfg)
    return model, model_cfg


def load_checkpoint_weights(
    model: MaterialFlowModule,
    ckpt_path: Path,
    eval_with_ema: bool = True,
) -> None:
    """Load raw or EMA checkpoint weights into an instantiated model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    target_keys = set(model.state_dict().keys())
    model.load_state_dict(
        remap_state_dict_keys(select_checkpoint_weights(ckpt, "raw"), target_keys),
        strict=True,
    )
    if eval_with_ema:
        try:
            ema_weights = select_checkpoint_weights(ckpt, "ema")
        except KeyError:
            log.warning(f"EMA requested but checkpoint has no EMA state: {ckpt_path}")
            log.info(f"Loaded raw weights from checkpoint: {ckpt_path}")
            return
        model.load_state_dict(
            remap_state_dict_keys(ema_weights, target_keys),
            strict=False,
        )
        log.info(f"Loaded EMA weights from checkpoint: {ckpt_path}")
        return
    log.info(f"Loaded raw weights from checkpoint: {ckpt_path}")


def load_model(
    ckpt_path: Path,
    eval_with_ema: bool = True,
    model_overrides: DictConfig | None = None,
    run_config_path: Path | None = None,
) -> tuple[MaterialFlowModule, DictConfig]:
    """Instantiate model from run config and load checkpoint weights."""
    # Rebuild model exactly as it was trained
    run_cfg, _ = load_run_config(ckpt_path, run_config_path)
    model, _ = instantiate_model_from_run_config(run_cfg, model_overrides)
    load_checkpoint_weights(model, ckpt_path, eval_with_ema=eval_with_ema)
    return model, run_cfg


def load_model_preserving_rng(
    ckpt_path: Path,
    eval_with_ema: bool = True,
    model_overrides: DictConfig | None = None,
    run_config_path: Path | None = None,
) -> tuple[MaterialFlowModule, DictConfig]:
    """Load a checkpoint without advancing the caller's PyTorch RNG streams."""
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return load_model(
            ckpt_path,
            eval_with_ema=eval_with_ema,
            model_overrides=model_overrides,
            run_config_path=run_config_path,
        )
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
