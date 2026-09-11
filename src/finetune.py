"""Fine-tune Packora from checkpoint weights with fresh training state."""

from pathlib import Path
from typing import Any, Optional

import hydra
import lightning as L
import rootutils
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import Callback, LightningDataModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.flow_module import MaterialFlowModule  # noqa: E402
from src.prediction.checkpoint import (  # noqa: E402
    CheckpointWeightSource,
    load_run_config,
    resolve_model_config,
    select_checkpoint_weights,
)
from src.train import (  # noqa: E402
    configure_distributed_sampling,
    configure_wandb_watch,
    validate_effective_batch_size,
    validate_lattice_config,
)
from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    write_run_info,
)
from src.utils.checkpoint_state import remap_state_dict_keys  # noqa: E402
from src.utils.omegaconf_resolvers import register_omegaconf_resolvers  # noqa: E402

register_omegaconf_resolvers()
log = RankedLogger(__name__, rank_zero_only=True)


def _selected_checkpoint_path(cfg: DictConfig) -> tuple[Path, bool]:
    """Return the selected checkpoint path and whether it is a full resume."""
    resume_path = cfg.get("ckpt_path")
    init_path = cfg.get("init_ckpt_path")
    selected = resume_path if resume_path is not None else init_path
    if selected is None:
        raise ValueError("Set init_ckpt_path for a new stage or ckpt_path to resume.")
    ckpt_path = Path(str(selected)).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path, resume_path is not None


def _model_overrides(cfg: DictConfig) -> DictConfig:
    """Build explicit stage-2 optimizer and scheduler overrides."""
    learning_rate = float(cfg.finetune.learning_rate)
    warmup_steps = int(cfg.finetune.warmup_steps)
    if learning_rate <= 0.0:
        raise ValueError("finetune.learning_rate must be > 0.")
    if warmup_steps < 0:
        raise ValueError("finetune.warmup_steps must be >= 0.")
    return OmegaConf.create(
        {
            "ema_warm_start": bool(cfg.finetune.ema_warm_start),
            "optimizer": {"lr": learning_rate},
            "scheduler": {
                "target_lr": learning_rate,
                "warmup_no_steps": warmup_steps,
            },
        }
    )


def initialize_stage2_weights(
    model: MaterialFlowModule,
    ckpt_path: Path,
    raw_source: CheckpointWeightSource,
    ema_source: CheckpointWeightSource,
) -> None:
    """Initialize stage-2 raw and EMA weights from independent sources."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    target_keys = set(model.state_dict().keys())
    raw_weights = remap_state_dict_keys(
        select_checkpoint_weights(checkpoint, raw_source),
        target_keys,
    )
    ema_weights = remap_state_dict_keys(
        select_checkpoint_weights(checkpoint, ema_source),
        target_keys,
    )

    # Validate and clone EMA weights before restoring the selected raw weights
    model.load_state_dict(ema_weights, strict=True)
    ema_initial_weights = {
        key: value.detach().clone().cpu() for key, value in model.state_dict().items()
    }
    model.load_state_dict(raw_weights, strict=True)
    model.set_ema_initial_weights(ema_initial_weights)
    log.info(f"Initialized stage-2 raw weights from pretrained {raw_source} weights.")
    log.info(f"Initialized stage-2 EMA weights from pretrained {ema_source} weights.")


def prepare_finetune_config(cfg: DictConfig) -> tuple[Path, bool]:
    """Inject the checkpoint's effective model config into the finetune config."""
    ckpt_path, is_resume = _selected_checkpoint_path(cfg)
    configured_run_path = cfg.get("run_config_path")
    run_config_path = (
        Path(str(configured_run_path)).expanduser()
        if configured_run_path is not None
        else None
    )
    run_cfg, resolved_run_path = load_run_config(ckpt_path, run_config_path)
    overrides = None if is_resume else _model_overrides(cfg)
    model_cfg = resolve_model_config(run_cfg, overrides)

    with open_dict(cfg):
        cfg.model = model_cfg
        cfg.resolved_run_config_path = str(resolved_run_path)
    return ckpt_path, is_resume


@rank_zero_only
def save_effective_run_config(cfg: DictConfig) -> None:
    """Persist the reconstructed model config in standard Hydra run metadata."""
    output_dir = Path(str(HydraConfig.get().runtime.output_dir))
    config_path = output_dir / ".hydra" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=config_path)
    log.info(f"Saved effective finetune config to {config_path}")


@task_wrapper
def finetune(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run stage-2 training from selected weights or a full-state resume."""
    if cfg.get("model") is None:
        ckpt_path, is_resume = prepare_finetune_config(cfg)
    else:
        ckpt_path, is_resume = _selected_checkpoint_path(cfg)

    configure_distributed_sampling(cfg)
    validate_lattice_config(cfg)
    validate_effective_batch_size(cfg)

    if cfg.get("seed") is not None:
        L.seed_everything(int(cfg.seed), workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: MaterialFlowModule = hydra.utils.instantiate(cfg.model)
    if not is_resume:
        initialize_stage2_weights(
            model,
            ckpt_path,
            raw_source=str(cfg.finetune.init_weight_source.raw),
            ema_source=str(cfg.finetune.init_weight_source.ema),
        )

    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log_hyperparameters(object_dict)
        configure_wandb_watch(cfg, model, logger)

    write_run_info(Path(cfg.paths.output_dir))
    resume_path = str(ckpt_path) if is_resume else None
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_path)

    return trainer.callback_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Run the stage-2 finetuning entrypoint."""
    prepare_finetune_config(cfg)
    extras(cfg)
    save_effective_run_config(cfg)
    finetune(cfg)
    return None


if __name__ == "__main__":
    main()
