"""Artifact and runtime settings for single-structure prediction."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelSpec:
    """Describe one selectable Packora checkpoint."""

    id: str
    label: str
    checkpoint_id: str
    checkpoint_path: Path


@dataclass(frozen=True)
class PredictionSettings:
    """Hold local artifact and sampling settings for prediction."""

    device: str
    max_atoms: int
    num_steps: int
    model_manifest_path: Path | None
    z_prior_path: Path | None
    models: dict[str, ModelSpec]

    @classmethod
    def resolve(
        cls,
        *,
        device: str | None = None,
        max_atoms: int | None = None,
        num_steps: int | None = None,
        model_manifest_path: Path | None = None,
        checkpoint_paths: Mapping[str, Path] | None = None,
        z_prior_path: Path | None = None,
    ) -> "PredictionSettings":
        """Resolve explicit artifact paths over environment configuration."""
        manifest_value = model_manifest_path or os.environ.get("PACKORA_MODEL_MANIFEST")
        manifest_path = (
            None if manifest_value is None else Path(manifest_value).resolve()
        )
        if manifest_path is None:
            models = load_direct_model_specs(checkpoint_paths)
        else:
            models = load_model_specs(
                manifest_path,
                checkpoint_paths=checkpoint_paths,
            )
        if not models:
            raise ValueError(
                "No checkpoints are configured. Pass checkpoint_paths, set "
                "PACKORA_M_CHECKPOINT or PACKORA_L_CHECKPOINT, or configure a "
                "model manifest."
            )
        resolved_device = device
        if resolved_device is None:
            gpu_device = int(os.environ.get("PACKORA_GPU_DEVICE", "0"))
            resolved_device = f"cuda:{gpu_device}"
        return cls(
            device=str(resolved_device),
            max_atoms=(
                int(max_atoms)
                if max_atoms is not None
                else int(os.environ.get("PACKORA_MAX_ATOMS", "512"))
            ),
            num_steps=(
                int(num_steps)
                if num_steps is not None
                else int(os.environ.get("PACKORA_NUM_STEPS", "200"))
            ),
            model_manifest_path=manifest_path,
            z_prior_path=_optional_artifact_path(
                z_prior_path,
                environment_key="PACKORA_Z_PRIOR",
            ),
            models=models,
        )

    def validate_for(
        self,
        model_id: str,
        *,
        require_z_prior: bool,
        require_checkpoint: bool = True,
        require_cuda: bool = True,
    ) -> list[str]:
        """Return errors relevant to one requested model and Z mode."""
        errors: list[str] = []
        if self.max_atoms < 1 or self.max_atoms > 512:
            errors.append("max_atoms must be between 1 and 512.")
        if self.num_steps < 1:
            errors.append("num_steps must be positive.")
        model = self.models.get(model_id)
        if model is None:
            errors.append(f"Unknown model: {model_id}")
        elif require_checkpoint and not model.checkpoint_path.is_file():
            errors.append(
                f"Checkpoint for {model.label} not found: {model.checkpoint_path}"
            )
        if require_z_prior and self.z_prior_path is None:
            errors.append(
                "Empirical Z prior is not configured. Pass z_prior_path or set "
                "PACKORA_Z_PRIOR."
            )
        elif require_z_prior and not self.z_prior_path.is_file():
            errors.append(f"Empirical Z prior not found: {self.z_prior_path}")
        if require_cuda:
            errors.extend(_cuda_errors(self.device))
        return errors


def load_model_specs(
    manifest_path: Path,
    checkpoint_paths: Mapping[str, Path] | None = None,
) -> dict[str, ModelSpec]:
    """Load model specs with explicit and environment checkpoint overrides."""
    payload = _read_json(manifest_path)
    methods = payload.get("methods")
    if not isinstance(methods, dict):
        raise ValueError(f"Model manifest has no methods mapping: {manifest_path}")
    explicit = {} if checkpoint_paths is None else dict(checkpoint_paths)
    models: dict[str, ModelSpec] = {}
    for model_id, (method_id, label) in _MODEL_MAPPING.items():
        raw = methods.get(method_id)
        if not isinstance(raw, dict):
            raise ValueError(f"Model manifest is missing {method_id!r}.")
        fallback = _manifest_artifact_path(manifest_path, raw["checkpoint_path"])
        env_path = os.environ.get(_checkpoint_env_key(model_id))
        checkpoint_path = explicit.get(
            model_id, Path(env_path) if env_path else fallback
        )
        models[model_id] = ModelSpec(
            id=model_id,
            label=label,
            checkpoint_id=str(raw["checkpoint_id"]),
            checkpoint_path=Path(checkpoint_path).resolve(),
        )
    unknown = sorted(set(explicit) - set(_MODEL_MAPPING))
    if unknown:
        raise ValueError(f"Unknown checkpoint model overrides: {', '.join(unknown)}")
    return models


def load_direct_model_specs(
    checkpoint_paths: Mapping[str, Path] | None,
) -> dict[str, ModelSpec]:
    """Build model specs directly from arguments and environment paths."""
    explicit = {} if checkpoint_paths is None else dict(checkpoint_paths)
    unknown = sorted(set(explicit) - set(_MODEL_MAPPING))
    if unknown:
        raise ValueError(f"Unknown checkpoint model overrides: {', '.join(unknown)}")
    models: dict[str, ModelSpec] = {}
    for model_id, (_, label) in _MODEL_MAPPING.items():
        env_path = os.environ.get(_checkpoint_env_key(model_id))
        checkpoint_path = explicit.get(
            model_id, None if env_path is None else Path(env_path)
        )
        if checkpoint_path is None:
            continue
        models[model_id] = ModelSpec(
            id=model_id,
            label=label,
            checkpoint_id=model_id,
            checkpoint_path=Path(checkpoint_path).resolve(),
        )
    return models


def _manifest_artifact_path(manifest_path: Path, value: Any) -> Path:
    """Resolve one manifest artifact relative to its containing directory."""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _checkpoint_env_key(model_id: str) -> str:
    """Return the per-model checkpoint environment variable name."""
    suffix = model_id.removeprefix("packora-").upper().replace("-", "_")
    return f"PACKORA_{suffix}_CHECKPOINT"


_MODEL_MAPPING = {
    "packora-m": ("medium", "Packora-M"),
    "packora-l": ("large", "Packora-L"),
}


def _optional_artifact_path(
    explicit_path: Path | None,
    *,
    environment_key: str,
) -> Path | None:
    """Resolve one optional explicit or environment artifact path."""
    value = explicit_path or os.environ.get(environment_key)
    return None if value is None else Path(value).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}.")
    return payload


def _cuda_errors(device_name: str) -> list[str]:
    """Return CUDA availability errors for one device string."""
    try:
        import torch
    except ImportError:
        return ["PyTorch is unavailable."]
    device = torch.device(device_name)
    if device.type != "cuda":
        return [f"Packora prediction requires a CUDA device, received {device_name!r}."]
    if not torch.cuda.is_available():
        return ["CUDA is unavailable."]
    index = 0 if device.index is None else int(device.index)
    if index >= torch.cuda.device_count():
        return [f"GPU {index} is outside the available device range."]
    return []
