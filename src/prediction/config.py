"""Prediction config resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

from src.data.components.conditioning import CONDITIONING_GROUPS

PREDICTION_CONDITIONING_MODES = {"on", "off"}


@dataclass(frozen=True)
class SourceSpec:
    """Resolved data source used for prediction and manifest writing."""

    data_dir: str
    dataset_name: str
    split: str
    benchmark: str | None = None


def resolve_with_fallback(
    cfg_value: Optional[str], run_cfg_value: Optional[str]
) -> str:
    """Resolve one source field from predict config or training run config."""
    if cfg_value is not None:
        return str(cfg_value)
    if run_cfg_value is None:
        raise ValueError(
            "Unable to resolve source value from both predict and run configs."
        )
    return str(run_cfg_value)


def resolve_source_spec(cfg: DictConfig, run_cfg: DictConfig) -> SourceSpec:
    """Resolve all source fields once for dataset building and output metadata."""
    benchmark = cfg.source.get("benchmark")
    if benchmark is not None:
        return SourceSpec(
            data_dir=str(cfg.source.get("data_dir") or cfg.paths.data_dir),
            dataset_name=str(cfg.source.get("dataset_name") or "csd_benchmarks"),
            split=str(benchmark),
            benchmark=str(benchmark),
        )

    data_dir = resolve_with_fallback(
        cfg.source.get("data_dir"),
        run_cfg.data.get("data_dir"),
    )
    dataset_name = resolve_with_fallback(
        cfg.source.get("dataset_name"),
        run_cfg.data.get("dataset_name"),
    )
    split = str(cfg.source.get("split", "test"))
    return SourceSpec(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split=split,
    )


def resolve_max_num_atoms(cfg: DictConfig, run_cfg: DictConfig) -> int | None:
    """Resolve the prediction atom-count limit from predict or training config."""
    max_num_atoms = cfg.data.get("max_num_atoms")
    if cfg.source.get("benchmark") is not None and max_num_atoms is None:
        return None
    if max_num_atoms is None:
        max_num_atoms = run_cfg.data.get("max_num_atoms")
    if max_num_atoms is None:
        return None
    return int(max_num_atoms)


def center_cart_coords_from_run_config(run_cfg: DictConfig) -> bool:
    """Return the checkpoint-owned Cartesian centering mode."""
    return bool(run_cfg.model.get("center_cart_coords", True))


def validate_matching_centering_modes(
    run_cfg: DictConfig,
    bad_run_cfg: DictConfig,
) -> None:
    """Require main and autoguidance checkpoints to share a coordinate gauge."""
    if center_cart_coords_from_run_config(
        bad_run_cfg
    ) != center_cart_coords_from_run_config(run_cfg):
        raise ValueError(
            "Autoguidance checkpoints must use the same "
            "model.center_cart_coords setting."
        )


def resolve_conditioning_context(cfg: DictConfig, run_cfg: DictConfig) -> str:
    """Return the resolved conditioning policy context for prediction."""
    context_key = str(cfg.sampling.get("conditioning_context", "predict"))
    contexts = run_cfg.conditioning.context_by_role
    return str(contexts.get(context_key, context_key))


def plain_mapping(value: Any, config_name: str) -> dict[str, Any]:
    """Convert an optional Hydra mapping node to a plain dictionary."""
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        raw = OmegaConf.to_container(value, resolve=True)
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise ValueError(f"{config_name} must be a mapping.")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_name} must be a mapping.")
    return dict(raw)


def prediction_conditioning_mode(value: Any, field_name: str) -> str:
    """Normalize one prediction conditioning mode to on/off."""
    if isinstance(value, bool):
        return "on" if value else "off"
    mode = str(value).lower()
    if mode not in PREDICTION_CONDITIONING_MODES:
        raise ValueError(
            f"conditioning_policy.{field_name} must be 'on' or 'off', got {value!r}."
        )
    return mode


def resolve_conditioning_policy(cfg: DictConfig, run_cfg: DictConfig) -> DictConfig:
    """Return the saved conditioning policy with explicit prediction modes applied."""
    policy = OmegaConf.create(
        OmegaConf.to_container(run_cfg.conditioning.policy, resolve=True)
    )
    override = plain_mapping(cfg.get("conditioning_policy"), "conditioning_policy")
    if not override:
        return policy

    unknown_fields = sorted(set(override) - set(CONDITIONING_GROUPS))
    if unknown_fields:
        raise ValueError(
            "conditioning_policy only supports direct prediction fields: "
            f"{', '.join(CONDITIONING_GROUPS)}. Unsupported: {', '.join(unknown_fields)}."
        )

    context = resolve_conditioning_context(cfg, run_cfg)
    if context not in policy.contexts:
        raise ValueError(f"Unknown prediction conditioning context: {context}")
    context_modes = {
        group: str(policy.contexts[context][group]) for group in CONDITIONING_GROUPS
    }
    for group in CONDITIONING_GROUPS:
        if group in override:
            context_modes[group] = prediction_conditioning_mode(override[group], group)
    policy.contexts[context] = context_modes
    return policy


def prediction_compile_enabled(value: Any) -> bool:
    """Normalize the prediction compile toggle."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"model_overrides.compile must be true or false, got {value!r}.")


def resolve_model_overrides(cfg: DictConfig) -> DictConfig:
    """Translate public prediction model toggles to model constructor overrides."""
    override = plain_mapping(cfg.get("model_overrides"), "model_overrides")
    if not override:
        return OmegaConf.create({})

    unknown_fields = sorted(set(override) - {"compile"})
    if unknown_fields:
        raise ValueError(
            "model_overrides only supports the direct prediction field 'compile'. "
            f"Unsupported: {', '.join(unknown_fields)}."
        )

    if "compile" not in override or override["compile"] is None:
        return OmegaConf.create({})
    if prediction_compile_enabled(override["compile"]):
        return OmegaConf.create({"compile_target": "net"})
    return OmegaConf.create({"compile_target": None})
