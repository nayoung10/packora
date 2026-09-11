from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    write_run_info,
)
from src.utils.omegaconf_resolvers import register_omegaconf_resolvers  # noqa: E402

register_omegaconf_resolvers()
log = RankedLogger(__name__, rank_zero_only=True)


def _count_devices(devices: Any) -> int:
    """Return the number of trainer devices from a supported config value."""
    if isinstance(devices, int):
        return devices
    if isinstance(devices, str) and devices.isdigit():
        return int(devices)
    if isinstance(devices, (list, tuple)):
        return len(devices)
    raise ValueError(f"Unsupported trainer.devices value for batch sizing: {devices}")


def configure_distributed_sampling(cfg: DictConfig) -> None:
    """Disable Lightning sampler injection when the datamodule owns DDP sharding."""
    has_configured_sampler = cfg.data.get("sampler") is not None
    if not has_configured_sampler and not cfg.data.get(
        "length_bucketed_batches", False
    ):
        return
    with open_dict(cfg):
        cfg.trainer.use_distributed_sampler = False


def validate_lattice_config(cfg: DictConfig) -> None:
    """Validate cross-component lattice representation constraints."""
    if cfg.model.get("lattice_repr") != "ltri":
        return
    if cfg.data.get("crystal_rotate", False):
        raise ValueError(
            "model.lattice_repr='ltri' is incompatible with data.crystal_rotate=true. "
            "The ltri representation canonicalizes lattice orientation, while rotated "
            "Cartesian coordinates remain in the rotated frame."
        )


def validate_effective_batch_size(cfg: DictConfig) -> None:
    """Ensure resolved per-device batch size matches the requested effective size."""
    effective_batch_size = cfg.data.get("effective_batch_size")
    if effective_batch_size is None:
        return

    devices = _count_devices(cfg.trainer.devices)
    num_nodes = int(cfg.trainer.get("num_nodes", 1))
    accumulate_grad_batches = int(cfg.trainer.get("accumulate_grad_batches", 1))
    actual_batch_size = (
        int(cfg.data.batch_size) * devices * num_nodes * accumulate_grad_batches
    )
    if actual_batch_size == int(effective_batch_size):
        return

    divisor = devices * num_nodes * accumulate_grad_batches
    raise ValueError(
        "data.effective_batch_size must be divisible by "
        "trainer.devices * trainer.num_nodes * trainer.accumulate_grad_batches. "
        f"Got effective_batch_size={effective_batch_size}, divisor={divisor}, "
        f"resolved data.batch_size={cfg.data.batch_size}, actual={actual_batch_size}."
    )


def configure_wandb_watch(
    cfg: DictConfig,
    model: LightningModule,
    loggers: List[Logger],
) -> None:
    """Enable optional W&B model watch from Hydra config."""
    logger_cfg = cfg.get("logger")
    if not logger_cfg:
        return
    watch_cfg = logger_cfg.get("wandb_watch")
    if not watch_cfg or not watch_cfg.get("enabled", False):
        return
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.watch(
                model,
                log=watch_cfg.get("log", "gradients"),
                log_freq=watch_cfg.get("log_freq", 500),
                log_graph=watch_cfg.get("log_graph", False),
            )


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    configure_distributed_sampling(cfg)
    validate_lattice_config(cfg)
    validate_effective_batch_size(cfg)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
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
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
        configure_wandb_watch(cfg, model, logger)

    if cfg.get("train"):
        write_run_info(Path(cfg.paths.output_dir))
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    configure_distributed_sampling(cfg)
    validate_lattice_config(cfg)
    validate_effective_batch_size(cfg)

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
