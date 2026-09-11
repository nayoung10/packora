"""Prediction sampling config helpers."""

from __future__ import annotations

import math
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.prediction.config import (
    resolve_conditioning_context,
    resolve_conditioning_policy,
)


def spacegroup_cfg_weight(cfg: DictConfig) -> float | None:
    """Return the validated configured space-group CFG weight."""
    spacegroup_cfg = cfg.sampling.get("spacegroup_cfg")
    if spacegroup_cfg is None:
        return None

    args = dict(OmegaConf.to_container(spacegroup_cfg, resolve=True))
    unknown = sorted(set(args).difference({"weight"}))
    if unknown:
        raise ValueError(
            "sampling.spacegroup_cfg only supports weight. "
            f"Unsupported: {', '.join(unknown)}."
        )

    value = args.get("weight")
    if value is None:
        return None
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("sampling.spacegroup_cfg.weight must be finite and >= 0.")
    return weight


def effective_spacegroup_cfg_weight(cfg: DictConfig, run_cfg: DictConfig) -> float:
    """Return the supported weight after applying the space-group policy switch."""
    weight = spacegroup_cfg_weight(cfg)
    policy = resolve_conditioning_policy(cfg, run_cfg)
    context = resolve_conditioning_context(cfg, run_cfg)
    mode = str(policy.contexts[context].spacegroup)
    if mode == "off":
        return 0.0
    if mode != "on":
        raise ValueError(
            f"Prediction space-group conditioning mode must be on/off, got {mode!r}."
        )
    if weight is None:
        return 1.0
    if weight != 1.0:
        raise NotImplementedError(
            "Space-group CFG is currently unsupported; use "
            "sampling.spacegroup_cfg.weight=1 for ordinary conditional inference "
            "or conditioning_policy.spacegroup=off for unconditional inference."
        )
    return weight


def autoguidance_args_dict(cfg: DictConfig) -> dict[str, Any] | None:
    """Return validated autoguidance args when enabled."""
    autoguidance = cfg.sampling.get("autoguidance")
    if autoguidance is None:
        return None

    args = dict(OmegaConf.to_container(autoguidance, resolve=True))
    unknown = sorted(set(args).difference({"enabled", "bad_ckpt_path", "weight"}))
    if unknown:
        raise ValueError(
            "sampling.autoguidance only supports enabled, bad_ckpt_path, and weight. "
            f"Unsupported: {', '.join(unknown)}."
        )

    enabled = bool(args.get("enabled", False))
    if not enabled:
        return None

    bad_ckpt_path = args.get("bad_ckpt_path")
    if bad_ckpt_path is None or str(bad_ckpt_path).strip() == "":
        raise ValueError(
            "sampling.autoguidance.bad_ckpt_path is required when autoguidance is enabled."
        )

    weight = float(args.get("weight", 1.0))
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("sampling.autoguidance.weight must be finite and >= 0.")

    return {
        "bad_ckpt_path": str(bad_ckpt_path),
        "weight": weight,
    }


def steering_args_dict(cfg: DictConfig) -> dict[str, Any] | None:
    """Return resolved FK steering args as a plain dictionary."""
    steering_args = cfg.sampling.get("steering_args")
    if steering_args is None:
        return None
    return dict(OmegaConf.to_container(steering_args, resolve=True))


def sampler_args_dict(cfg: DictConfig) -> dict[str, Any]:
    """Return resolved sampler args as a plain dictionary."""
    sampler_args = cfg.sampling.get("sampler_args")
    if sampler_args is None:
        return {}
    return dict(OmegaConf.to_container(sampler_args, resolve=True))


def samples_per_datapoint(cfg: DictConfig) -> int:
    """Return the requested samples per source datapoint."""
    value = int(cfg.sampling.samples_per_datapoint)
    if value < 1:
        raise ValueError("sampling.samples_per_datapoint must be >= 1.")
    return value


def sample_chunk_size(cfg: DictConfig) -> int:
    """Return the prediction sample chunk size."""
    value = cfg.sampling.get("sample_chunk_size")
    if value is None:
        value = samples_per_datapoint(cfg)
    value = int(value)
    if value < 1:
        raise ValueError("sampling.sample_chunk_size must be >= 1.")
    return value


def time_epsilon(cfg: DictConfig) -> float:
    """Return the endpoint epsilon used for ODE and SDE sampling."""
    value = float(cfg.sampling.time_epsilon)
    if value <= 0.0 or value >= 0.5:
        raise ValueError("sampling.time_epsilon must satisfy 0 < value < 0.5.")
    return value


def uses_mlip_fk_steering(cfg: DictConfig) -> bool:
    """Return whether prediction uses MLIP-backed FK steering."""
    steering_args = steering_args_dict(cfg)
    if steering_args is None:
        return False
    return bool(
        steering_args.get("fk_steering", False)
        and steering_args.get("energy_fn", "mlip") == "mlip"
    )


def effective_sampling_method(cfg: DictConfig) -> str:
    """Return the sampling method after FK steering overrides."""
    method = str(cfg.sampling.method)
    steering_args = steering_args_dict(cfg)
    if steering_args is not None and steering_args.get("fk_steering", False):
        return "sde"
    return method
