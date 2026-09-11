from __future__ import annotations

from functools import partial
from typing import Any, Optional

import lightning as L
import torch
import torch.nn as nn
from lightning import Callback
from torch import Tensor

from src.models.components.interpolants import Interpolant
from src.models.components.scalers import Scaler
from src.models.components.time_samplers import TimeSampler
from src.models.flow_model import MaterialFlowMatching
from src.utils.loss_utils import stratify_loss_by_time
from src.utils.tensor_typing import Float


class MaterialFlowModule(L.LightningModule):
    """Lightning wrapper: loss weighting, logging, optimizer config."""

    def __init__(
        self,
        net: nn.Module,
        interpolant: Interpolant,
        scaler: Scaler,
        prior: dict,
        time_sampler: TimeSampler,
        optimizer: partial,
        scheduler: Optional[partial] = None,
        periodic_pair_distance: Optional[nn.Module] = None,
        center_cart_coords: bool = True,
        self_conditioning_probability: float = 0.5,
        loss_type: str = "l1",
        loss_weights: Optional[dict[str, float]] = None,
        num_time_bins: int = 5,
        ema_decay: Optional[float] = None,
        ema_warm_start: bool = True,
        eval_with_ema: bool = False,
        activation: str = "silu",
        norm_type: Optional[str] = None,
        norm_eps: Optional[float] = None,
        lattice_repr: Optional[str] = None,
        lattice_dim: Optional[int] = None,
        conditioning_contexts: Optional[dict[str, str]] = None,
        compile_target: Optional[str] = None,
        compile_backend: Optional[str] = None,
        compile_dynamic: Optional[bool] = None,
        compile_mode: Optional[str] = None,
        compile_fullgraph: bool = False,
        compile_cache_size_limit: Optional[int] = None,
    ) -> None:
        """Initialize flow module with model components and training config."""
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "interpolant",
                "scaler",
                "time_sampler",
                "periodic_pair_distance",
            ]
        )

        self.flow_matching = MaterialFlowMatching(
            net=net,
            interpolant=interpolant,
            scaler=scaler,
            prior=prior,
            time_sampler=time_sampler,
            periodic_pair_distance=periodic_pair_distance,
            center_cart_coords=center_cart_coords,
            self_conditioning_probability=self_conditioning_probability,
        )
        self._configure_compile_cache(compile_cache_size_limit)
        self._compile_model_part(
            compile_target=compile_target,
            compile_backend=compile_backend,
            compile_dynamic=compile_dynamic,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
        )
        self.loss_type = loss_type
        self.num_time_bins = num_time_bins
        self.ema_decay = ema_decay
        self.ema_warm_start = ema_warm_start
        self.use_ema = ema_decay is not None
        self._ema_initial_weights: Optional[dict[str, Any]] = None

        self.loss_weights = loss_weights or {}
        self.conditioning_contexts = {
            "train_loss": "train_loss",
            "val_loss": "val_loss",
            "predict": "predict",
        }
        self.conditioning_contexts.update(conditioning_contexts or {})

        # Build weight-to-loss mapping based on lattice representation
        self._weight_to_loss = {
            "coords": "loss_coords",
        }
        if scaler.lattice_repr == "cell":
            self._weight_to_loss["lattice_cell"] = "loss_lattice_cell"
            self.loss_weights.setdefault("lattice_cell", 1.0)
        elif scaler.lattice_repr == "ltri":
            self._weight_to_loss["lattice_ltri"] = "loss_lattice_ltri"
            self.loss_weights.setdefault("lattice_ltri", 1.0)
        else:
            self._weight_to_loss["lattice_lengths"] = "loss_lattice_lengths"
            self._weight_to_loss["lattice_angles"] = "loss_lattice_angles"
            self.loss_weights.setdefault("lattice_lengths", 1.0)
            self.loss_weights.setdefault("lattice_angles", 1.0)
        self.loss_weights.setdefault("coords", 1.0)
        if periodic_pair_distance is not None:
            self._weight_to_loss["periodic_pair_distance"] = (
                "loss_periodic_pair_distance"
            )
            self.loss_weights.setdefault("periodic_pair_distance", 1.0)

    def _configure_compile_cache(
        self,
        compile_cache_size_limit: Optional[int],
    ) -> None:
        """Set TorchDynamo's graph cache size limit when requested."""
        if compile_cache_size_limit is None:
            return
        torch._dynamo.config.cache_size_limit = int(compile_cache_size_limit)

    def _compile_model_part(
        self,
        compile_target: Optional[str],
        compile_backend: Optional[str],
        compile_dynamic: Optional[bool],
        compile_mode: Optional[str],
        compile_fullgraph: bool,
    ) -> None:
        """Compile the configured model component when requested."""
        if compile_target is None:
            return
        compile_kwargs: dict[str, object] = {"fullgraph": bool(compile_fullgraph)}
        if compile_backend is not None:
            compile_kwargs["backend"] = compile_backend
        if compile_dynamic is not None:
            compile_kwargs["dynamic"] = bool(compile_dynamic)
        if compile_mode is not None:
            compile_kwargs["mode"] = compile_mode

        if compile_target == "net":
            self.flow_matching.net = torch.compile(
                self.flow_matching.net,
                **compile_kwargs,
            )
            return
        raise ValueError(f"Unknown compile target: {compile_target}")

    def _shared_step(self, batch: dict, prefix: str) -> Float[""]:  # noqa: F722
        """Forward pass, loss computation, and logging."""
        context_key = "train_loss" if prefix == "train" else "val_loss"
        preds = self.flow_matching(
            batch,
            conditioning_context=self.conditioning_contexts[context_key],
        )
        t = preds.pop("t")
        loss_dict = self.flow_matching.compute_loss(
            batch,
            preds,
            t=t,
            loss_type=self.loss_type,
        )

        # Weighted total loss
        total = sum(
            self.loss_weights[wk] * loss_dict[lk]
            for wk, lk in self._weight_to_loss.items()
        )

        # Log losses
        sync_dist = prefix == "val"
        batch_size = int(batch["num_atoms"].shape[0])
        self.log(
            f"{prefix}/loss",
            total,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
        for wk, lk in self._weight_to_loss.items():
            self.log(
                f"{prefix}/{lk}",
                loss_dict[lk],
                sync_dist=sync_dist,
                batch_size=batch_size,
            )

        # Time-stratified logging (train only)
        if prefix == "train":
            for wk, lk in self._weight_to_loss.items():
                stratified = stratify_loss_by_time(
                    t,
                    loss_dict[f"per_sample_{wk}"],
                    num_bins=self.num_time_bins,
                    loss_name=lk,
                )
                self.log_dict({f"train/{k}": v for k, v in stratified.items()})

        return total

    def training_step(self, batch: dict, batch_idx: int) -> Float[""]:  # noqa: F722
        """Execute one training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> Float[""]:  # noqa: F722
        """Execute one validation step."""
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> dict:
        """Build optimizer and optional LR scheduler from Hydra partials."""
        optimizer = self.hparams.optimizer(self.named_parameters())
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}
        scheduler = self.hparams.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _sample_prediction_chunk(
        self,
        batch: dict[str, Any],
        chunk_size: int,
    ) -> dict[str, Tensor]:
        """Sample one prediction chunk for a collated batch."""
        return self.flow_matching.sample(
            num_atoms=batch["num_atoms"],
            multiplicity=int(chunk_size),
            num_steps=getattr(self, "_gen_num_steps", 100),
            method=getattr(self, "_gen_method", "ode"),
            sde_noise_scale=getattr(self, "_gen_sde_noise_scale", 1.0),
            conditioning=batch["conditioning"],
            conditioning_context=getattr(
                self,
                "_gen_conditioning_context",
                self.conditioning_contexts["predict"],
            ),
            steering_args=getattr(self, "_gen_steering_args", None),
            sampler_args=getattr(self, "_gen_sampler_args", None),
            spacegroup_guidance_weight=getattr(
                self,
                "_gen_spacegroup_guidance_weight",
                None,
            ),
            time_epsilon=getattr(self, "_gen_time_epsilon", 1e-3),
            use_inference_cache=getattr(self, "_gen_use_inference_cache", True),
        )

    def predict_step(
        self,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> list[dict[str, Any]] | None:
        """Predict crystal structures via flow matching sampling, conditioned on atom types."""
        samples_per_datapoint = int(getattr(self, "_gen_samples_per_datapoint", 1))
        sample_chunk_size = int(
            getattr(self, "_gen_sample_chunk_size", samples_per_datapoint)
        )
        if samples_per_datapoint < 1:
            raise ValueError("_gen_samples_per_datapoint must be >= 1.")
        if sample_chunk_size < 1:
            raise ValueError("_gen_sample_chunk_size must be >= 1.")

        prediction_writer = getattr(self, "_gen_prediction_writer", None)
        chunks: list[dict[str, Any]] | None = [] if prediction_writer is None else None
        for sample_offset in range(0, samples_per_datapoint, sample_chunk_size):
            chunk_size = min(
                sample_chunk_size,
                samples_per_datapoint - sample_offset,
            )
            chunk = {
                "pred": self._sample_prediction_chunk(batch, chunk_size),
                "sample_offset": int(sample_offset),
                "chunk_size": int(chunk_size),
            }
            if prediction_writer is None:
                if chunks is None:
                    raise RuntimeError("Internal prediction chunk state is invalid.")
                chunks.append(chunk)
            else:
                trainer = self.__dict__.get("_trainer")
                if trainer is None:
                    trainer = self.trainer
                prediction_writer.write_prediction_chunk(
                    trainer=trainer,
                    prediction=chunk,
                    batch_indices=None,
                    batch=batch,
                    batch_idx=batch_idx,
                )
        return chunks

    def set_ema_initial_weights(self, state_dict: dict[str, Any]) -> None:
        """Set explicit weights for initializing the next EMA trajectory."""
        if not self.use_ema:
            raise ValueError("Cannot initialize EMA weights when EMA is disabled.")
        self._ema_initial_weights = state_dict

    def configure_callbacks(self) -> list[Callback]:
        """Configure model-level callbacks (EMA if enabled)."""
        if self.use_ema:
            from src.models.callbacks.ema import EMA

            initial_weights = self._ema_initial_weights
            self._ema_initial_weights = None
            return [
                EMA(
                    self.ema_decay,
                    eval_with_ema=self.hparams.eval_with_ema,
                    warm_start=self.ema_warm_start,
                    initial_weights=initial_weights,
                )
            ]
        return []
