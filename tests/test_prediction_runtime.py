"""Tests for Lightning prediction runtime construction."""

from unittest.mock import patch

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from src.prediction.runtime import PredictionDataModule


class _AtomCountDataset(Dataset):
    """Minimal prediction dataset with atom-count metadata."""

    def __init__(self, lengths: list[int]) -> None:
        """Initialize the fake dataset with atom counts."""
        self.num_atoms_by_index = np.asarray(lengths, dtype=np.int64)
        self.selected_indices = list(range(len(lengths)))

    def __len__(self) -> int:
        """Return the fake dataset size."""
        return len(self.selected_indices)

    def __getitem__(self, index: int) -> int:
        """Return the index as fake prediction data."""
        return index


class _IdentityPredictor(L.LightningModule):
    """Return prediction batches unchanged."""

    def predict_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """Return one prediction batch unchanged."""
        del batch_idx, dataloader_idx
        return batch


def _collate_indices(batch: list[int], **_: object) -> torch.Tensor:
    """Collate fake dataset indices for the identity predictor."""
    return torch.tensor(batch)


def _prediction_cfg() -> DictConfig:
    """Return a minimal fixed-shape prediction config."""
    return OmegaConf.create(
        {
            "sampling": {
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "batch_sampler": {
                    "_target_": (
                        "src.data.components.samplers."
                        "DistributedFixedShapeBucketBatchSampler"
                    ),
                    "atom_buckets": [300, 512],
                    "drop_last": False,
                    "pad_to_full_batch": False,
                    "seed": 42,
                    "balance_across_ranks": True,
                },
            },
        },
    )


def test_prediction_datamodule_preserves_large_atom_bucket() -> None:
    """Lightning prediction reconstruction retains custom atom buckets."""
    dataset = _AtomCountDataset([280, 504])
    datamodule = PredictionDataModule(
        cfg=_prediction_cfg(),
        run_cfg=OmegaConf.create({}),
        dataset=dataset,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        use_distributed_sampler=False,
    )

    with (
        patch(
            "src.prediction.runtime.resolve_conditioning_policy",
            return_value=OmegaConf.create({}),
        ),
        patch(
            "src.prediction.runtime.resolve_conditioning_context",
            return_value="predict",
        ),
        patch("src.prediction.runtime.collate_fn", side_effect=_collate_indices),
    ):
        predictions = trainer.predict(
            model=_IdentityPredictor(),
            datamodule=datamodule,
            return_predictions=True,
        )

    assert predictions is not None
    assert sorted(int(batch.item()) for batch in predictions) == [0, 1]
    batch_sampler = trainer.predict_dataloaders.batch_sampler._batch_sampler
    assert batch_sampler.atom_buckets == [300, 512]
    assert batch_sampler.seed == 42
