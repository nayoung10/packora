"""Persistent checkpoint loading and single-sample Packora prediction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Protocol

import lightning as L
import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from src.data.components.collate import collate_fn
from src.data.components.transforms import augment_template
from src.prediction.checkpoint import load_model
from src.prediction.api.chemistry import FeaturizedInput
from src.prediction.api.settings import ModelSpec, PredictionSettings


class InferenceError(RuntimeError):
    """Raised when Packora prediction returns an invalid structure."""


@dataclass(frozen=True)
class PredictionArrays:
    """Contain one raw generated structure and its inference provenance."""

    cart_coords: np.ndarray
    cell: np.ndarray
    atomic_numbers: np.ndarray
    elapsed_seconds: float
    checkpoint_id: str
    model_label: str


class PredictionBackend(Protocol):
    """Define the swappable prediction backend contract."""

    def predict(
        self,
        model_id: str,
        featurized: FeaturizedInput,
        seed: int,
    ) -> PredictionArrays:
        """Generate one raw crystal structure."""


@dataclass
class LoadedModel:
    """Hold one GPU-resident model and its saved training configuration."""

    model: Any
    run_cfg: DictConfig
    spec: ModelSpec


class ModelRegistry:
    """Lazy-load and cache configured Packora checkpoints on one GPU."""

    def __init__(self, settings: PredictionSettings) -> None:
        """Initialize an empty model cache for one configured CUDA device."""
        self.settings = settings
        self.device = torch.device(settings.device)
        self._loaded: dict[str, LoadedModel] = {}

    def _load(self, model_id: str) -> LoadedModel:
        """Return a cached model or load its EMA weights without compilation."""
        if model_id in self._loaded:
            return self._loaded[model_id]
        spec = self.settings.models.get(model_id)
        if spec is None:
            raise InferenceError(f"Unknown model: {model_id}")
        model, run_cfg = load_model(
            spec.checkpoint_path,
            eval_with_ema=True,
            model_overrides=OmegaConf.create({"compile_target": None}),
        )
        model.to(self.device)
        model.eval()
        loaded = LoadedModel(model=model, run_cfg=run_cfg, spec=spec)
        self._loaded[model_id] = loaded
        return loaded

    @staticmethod
    def _conditioning_policy(run_cfg: DictConfig) -> tuple[DictConfig, str]:
        """Return the saved policy with benchmark prediction modes applied."""
        policy = OmegaConf.create(
            OmegaConf.to_container(run_cfg.conditioning.policy, resolve=True)
        )
        context = str(run_cfg.conditioning.context_by_role.get("predict", "predict"))
        policy.contexts[context] = {
            "template": "on",
            "stereochemistry": "on",
            "spacegroup": "off",
        }
        return policy, context

    @staticmethod
    def _sample(featurized: FeaturizedInput) -> dict[str, Any]:
        """Build one unpadded in-memory sample for the shared collator."""
        num_atoms = featurized.total_atom_count
        conditioning = featurized.conditioning.to_tensors()
        conditioning = replace(
            conditioning,
            template_coords=augment_template(
                conditioning.template_coords,
                conditioning.membership,
                rotate=True,
                translate=True,
            ),
        )
        return {
            "cart_coords": torch.zeros(num_atoms, 3, dtype=torch.float32),
            "frac_coords": torch.zeros(num_atoms, 3, dtype=torch.float32),
            "lattice": torch.tensor(
                [1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
                dtype=torch.float32,
            ),
            "cell": torch.eye(3, dtype=torch.float32),
            "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
            "conditioning": conditioning,
        }

    def predict(
        self,
        model_id: str,
        featurized: FeaturizedInput,
        seed: int,
    ) -> PredictionArrays:
        """Generate one benchmark-configured raw Packora prediction."""
        loaded = self._load(model_id)
        policy, context = self._conditioning_policy(loaded.run_cfg)
        L.seed_everything(int(seed), workers=True)
        batch = collate_fn(
            [self._sample(featurized)],
            conditioning_policy=policy,
            conditioning_context=context,
        )
        num_atoms = batch["num_atoms"].to(self.device)
        conditioning = {
            key: value.to(self.device) for key, value in batch["conditioning"].items()
        }
        torch.cuda.synchronize(self.device)
        started = perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            output = loaded.model.flow_matching.sample(
                num_atoms=num_atoms,
                multiplicity=1,
                num_steps=self.settings.num_steps,
                method="heun",
                conditioning=conditioning,
                conditioning_context=context,
                time_epsilon=1e-3,
                use_inference_cache=True,
            )
        torch.cuda.synchronize(self.device)
        elapsed_seconds = perf_counter() - started

        atom_mask = output["atom_mask"][0].detach().cpu().to(dtype=torch.bool)
        coords = (
            output["cart_coords"][0][atom_mask]
            .detach()
            .cpu()
            .to(dtype=torch.float32)
            .numpy()
        )
        atomic_numbers = (
            output["atomic_numbers"][0][atom_mask]
            .detach()
            .cpu()
            .to(dtype=torch.long)
            .numpy()
            .astype(np.int64, copy=False)
        )
        flat_cell = output["lattice"][0].detach().cpu().to(dtype=torch.float32).numpy()
        if flat_cell.shape != (9,):
            raise InferenceError(
                f"Expected a flattened 3x3 cell, received {flat_cell.shape}."
            )
        cell = rearrange(flat_cell, "(i j) -> i j", i=3, j=3)
        expected_numbers = featurized.conditioning.atomic_numbers.astype(
            np.int64,
            copy=False,
        )
        if not np.array_equal(atomic_numbers, expected_numbers):
            raise InferenceError("Prediction atom ordering differs from conditioning.")
        if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(cell)):
            raise InferenceError("Prediction contains non-finite coordinates or cell.")
        if abs(float(np.linalg.det(cell))) < 1e-8:
            raise InferenceError("Prediction returned a singular unit cell.")
        return PredictionArrays(
            cart_coords=np.asarray(coords, dtype=np.float64),
            cell=np.asarray(cell, dtype=np.float64),
            atomic_numbers=atomic_numbers,
            elapsed_seconds=elapsed_seconds,
            checkpoint_id=loaded.spec.checkpoint_id,
            model_label=loaded.spec.label,
        )
