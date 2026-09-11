"""Predict CSP structures from a checkpoint and save rich evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Tuple

import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.callbacks.generation_writer import PredictionBundleWriter  # noqa: E402
from src.prediction.artifacts import (  # noqa: E402
    cleanup_raw_payloads,
    collect_raw_payloads,
    write_manifest,
)
from src.prediction.checkpoint import load_model, load_model_preserving_rng  # noqa: E402
from src.prediction.config import (  # noqa: E402
    resolve_conditioning_context,
    resolve_model_overrides,
    resolve_source_spec,
    validate_matching_centering_modes,
)
from src.prediction.io import (  # noqa: E402
    build_prediction_bundle,
    save_prediction_bundle,
)
from src.prediction.runtime import (  # noqa: E402
    PredictionDataModule,
    build_dataset,
    build_trainer,
)
from src.prediction.sampling import (  # noqa: E402
    autoguidance_args_dict,
    effective_spacegroup_cfg_weight,
    effective_sampling_method,
    sample_chunk_size,
    sampler_args_dict,
    samples_per_datapoint,
    spacegroup_cfg_weight,
    steering_args_dict,
    time_epsilon,
)
from src.utils import RankedLogger, extras, task_wrapper  # noqa: E402
from src.utils.structure_io import save_structures_as_cif  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def predict(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run CSP prediction and save collated tensors for evaluation."""
    seed = cfg.get("seed")
    if seed is not None:
        L.seed_everything(int(seed), workers=True)

    ckpt_path = Path(cfg.ckpt_path).resolve()
    model_overrides = resolve_model_overrides(cfg)
    model, run_cfg = load_model(
        ckpt_path,
        eval_with_ema=bool(cfg.get("eval_with_ema", True)),
        model_overrides=model_overrides,
    )
    spacegroup_weight = effective_spacegroup_cfg_weight(cfg, run_cfg)
    autoguidance_args = autoguidance_args_dict(cfg)
    if autoguidance_args is not None:
        bad_ckpt_path = Path(autoguidance_args["bad_ckpt_path"]).resolve()
        bad_model, bad_run_cfg = load_model_preserving_rng(
            bad_ckpt_path,
            eval_with_ema=bool(cfg.get("eval_with_ema", True)),
            model_overrides=model_overrides,
        )
        validate_matching_centering_modes(run_cfg, bad_run_cfg)
        model.flow_matching.set_autoguidance(
            bad_net=bad_model.flow_matching.net,
            weight=float(autoguidance_args["weight"]),
        )
        log.info(
            f"Enabled autoguidance with bad checkpoint {bad_ckpt_path} "
            f"and weight={float(autoguidance_args['weight']):g}"
        )
    source = resolve_source_spec(cfg, run_cfg)

    # Sampling controls are attached to module attributes used in sample()
    model._gen_num_steps = int(cfg.sampling.num_steps)
    effective_method = effective_sampling_method(cfg)
    if effective_method != str(cfg.sampling.method):
        log.warning(
            f"FK steering requires stochastic sampling; overriding sampling.method={cfg.sampling.method} to sde.",
        )
    model._gen_method = effective_method
    model._gen_sde_noise_scale = float(cfg.sampling.sde_noise_scale)
    model._gen_sampler_args = sampler_args_dict(cfg)
    model._gen_spacegroup_guidance_weight = spacegroup_weight
    model._gen_conditioning_context = resolve_conditioning_context(cfg, run_cfg)
    model._gen_samples_per_datapoint = samples_per_datapoint(cfg)
    model._gen_sample_chunk_size = sample_chunk_size(cfg)
    model._gen_time_epsilon = time_epsilon(cfg)
    model._gen_steering_args = steering_args_dict(cfg)
    model._gen_use_inference_cache = bool(cfg.sampling.get("use_inference_cache", True))
    requested_spacegroup_weight = spacegroup_cfg_weight(cfg)
    if (
        requested_spacegroup_weight is not None
        and requested_spacegroup_weight != spacegroup_weight
    ):
        log.info(
            "Ignoring space-group CFG weight because conditioning_policy.spacegroup=off."
        )
    model.eval()

    dataset = build_dataset(cfg, run_cfg, source)
    datamodule = PredictionDataModule(cfg=cfg, run_cfg=run_cfg, dataset=dataset)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = PredictionBundleWriter(output_dir=output_dir)
    model._gen_prediction_writer = writer
    trainer = build_trainer(cfg, writer)

    start = perf_counter()
    # Trainer.predict enables distributed inference when trainer.devices > 1
    trainer.predict(model=model, datamodule=datamodule, return_predictions=False)

    # Ensure all ranks finished writing before rank-zero collation
    if trainer.world_size > 1:
        trainer.strategy.barrier()

    if not trainer.is_global_zero:
        return {}, {}

    (
        pred_batches,
        ref_batches,
        dataset_index_batches,
        sample_index_batches,
        metadata_rows,
    ) = collect_raw_payloads(output_dir)
    bundle = build_prediction_bundle(
        pred_batches=pred_batches,
        ref_batches=ref_batches,
        dataset_index_batches=dataset_index_batches,
        sample_index_batches=sample_index_batches,
        metadata_rows=metadata_rows,
    )

    # Persist one canonical prediction bundle for evaluation
    if bool(cfg.output.get("save_pt", True)):
        save_prediction_bundle(output_dir / "predictions.pt", bundle)
        log.info(f"Saved prediction bundle to {output_dir / 'predictions.pt'}")

    if bool(cfg.output.get("save_cif", False)):
        save_structures_as_cif(output_dir / "cif", bundle.pred)

    elapsed_seconds = perf_counter() - start
    if bool(cfg.output.get("save_manifest", True)):
        write_manifest(
            output_dir=output_dir,
            cfg=cfg,
            run_cfg=run_cfg,
            bundle=bundle,
            source=source,
            elapsed_seconds=elapsed_seconds,
        )

    cleanup_raw_payloads(output_dir, cfg)

    log.info(
        f"Prediction complete: {bundle.dataset_indices.shape[0]} samples in {elapsed_seconds:.2f} sec"
    )
    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    """Run prediction entrypoint."""
    extras(cfg)
    predict(cfg)


if __name__ == "__main__":
    main()
