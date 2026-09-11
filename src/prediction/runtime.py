"""Prediction runtime construction helpers."""

from __future__ import annotations

from functools import partial

import hydra
import torch
from lightning import LightningDataModule, Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.data.components.collate import collate_fn
from src.data.dataset import MaterialDataset
from src.models.callbacks.generation_writer import PredictionBundleWriter
from src.prediction.config import (
    SourceSpec,
    center_cart_coords_from_run_config,
    resolve_conditioning_context,
    resolve_conditioning_policy,
    resolve_max_num_atoms,
)
from src.prediction.sampling import uses_mlip_fk_steering
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PredictionDataModule(LightningDataModule):
    """Build the prediction dataloader inside Lightning's hook context."""

    def __init__(
        self,
        cfg: DictConfig,
        run_cfg: DictConfig,
        dataset: Dataset,
    ) -> None:
        """Initialize prediction dataloader inputs."""
        super().__init__()
        self.cfg = cfg
        self.run_cfg = run_cfg
        self.dataset = dataset
        batch_sampler_cfg = cfg.sampling.get("batch_sampler")
        if batch_sampler_cfg is not None:
            # Register the subclass before Lightning snapshots BatchSampler types
            hydra.utils.get_class(str(batch_sampler_cfg._target_))

    def predict_dataloader(self) -> DataLoader:
        """Build the dataloader while Lightning captures sampler arguments."""
        return build_dataloader(self.cfg, self.run_cfg, self.dataset)


def build_dataset(
    cfg: DictConfig, run_cfg: DictConfig, source: SourceSpec
) -> MaterialDataset:
    """Build a prediction dataset that includes references and metadata."""
    # Reuse training indexing so tensor conventions stay aligned
    indexing_cfg = run_cfg.data.get("indexing")
    subset_cfg = cfg.data.get("subset")
    max_num_atoms = resolve_max_num_atoms(cfg, run_cfg)

    dataset = MaterialDataset(
        data_dir=source.data_dir,
        dataset_name=source.dataset_name,
        split=source.split,
        subset=subset_cfg,
        crystal_rotate=False,
        crystal_translate=False,
        center_cart_coords=center_cart_coords_from_run_config(run_cfg),
        template_rotate=bool(cfg.data.get("template_rotate", False)),
        template_translate=bool(cfg.data.get("template_translate", False)),
        indexing=indexing_cfg,
        max_num_atoms=max_num_atoms,
        include_metadata=True,
    )
    log.info(
        f"Loaded {len(dataset)} samples from {source.data_dir}/{source.dataset_name} "
        f"split={source.split} max_num_atoms={max_num_atoms}"
    )
    return dataset


def build_dataloader(
    cfg: DictConfig,
    run_cfg: DictConfig,
    dataset: Dataset,
) -> DataLoader:
    """Build dataloader for Trainer.predict CSP inference."""
    num_workers = int(cfg.sampling.get("num_workers", 0))
    persistent_workers = bool(cfg.sampling.get("persistent_workers", False))
    conditioning_policy = resolve_conditioning_policy(cfg, run_cfg)
    conditioning_context = resolve_conditioning_context(cfg, run_cfg)
    batch_sampler_cfg = cfg.sampling.get("batch_sampler")
    pad_to_atom_buckets = None
    batch_sampler = None
    if batch_sampler_cfg is not None:
        atom_buckets = [
            int(bucket) for bucket in batch_sampler_cfg.get("atom_buckets", [])
        ]
        pad_to_atom_buckets = atom_buckets if atom_buckets else None
        batch_sampler = hydra.utils.instantiate(
            batch_sampler_cfg,
            dataset=dataset,
            batch_size=int(cfg.sampling.batch_size),
            _recursive_=False,
        )
    collate = partial(
        collate_fn,
        conditioning_policy=conditioning_policy,
        conditioning_context=conditioning_context,
        pad_to_atom_buckets=pad_to_atom_buckets,
    )
    loader_kwargs = {
        "dataset": dataset,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.sampling.get("pin_memory", False)),
        "persistent_workers": persistent_workers and num_workers > 0,
        "collate_fn": collate,
    }
    if batch_sampler is None:
        loader_kwargs["batch_size"] = int(cfg.sampling.batch_size)
    else:
        loader_kwargs["batch_sampler"] = batch_sampler
        loader_kwargs.pop("shuffle")
    return DataLoader(**loader_kwargs)


def build_trainer(cfg: DictConfig, writer: PredictionBundleWriter) -> Trainer:
    """Instantiate prediction trainer with safe CPU fallback when CUDA is unavailable."""
    # Resolve interpolations before detaching trainer config
    trainer_cfg = OmegaConf.create(OmegaConf.to_container(cfg.trainer, resolve=True))
    if cfg.sampling.get("batch_sampler") is not None:
        trainer_cfg["use_distributed_sampler"] = False
    accelerator = str(trainer_cfg.get("accelerator", "cpu"))
    if accelerator == "gpu" and not torch.cuda.is_available():
        log.warning(
            "trainer.accelerator=gpu requested but CUDA is unavailable, using CPU."
        )
        trainer_cfg["accelerator"] = "cpu"
        trainer_cfg["devices"] = 1

    return hydra.utils.instantiate(
        trainer_cfg,
        callbacks=[writer],
        logger=False,
        inference_mode=not uses_mlip_fk_steering(cfg),
    )
