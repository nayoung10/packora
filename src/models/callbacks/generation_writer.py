"""Callback for writing generated structures to disk during prediction."""

from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class PredictionBundleWriter(BasePredictionWriter):
    """Write raw per-batch prediction payloads for later bundle collation."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize writer for multi-rank prediction bundle payloads."""
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)

    def _dataset_index(
        self,
        batch: dict[str, Any],
        batch_indices: Optional[Sequence[int]],
    ) -> torch.Tensor:
        """Return dataset indices for one prediction batch."""
        if "dataset_index" in batch:
            return batch["dataset_index"].cpu()
        if batch_indices is not None:
            return torch.as_tensor(batch_indices, dtype=torch.long)
        raise ValueError("Prediction writer requires dataset_index or batch_indices.")

    def _reference_payload(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Return CPU reference tensors for one prediction batch."""
        return {
            "cart_coords": batch["cart_coords"].cpu(),
            "lattice": batch["lattice"].cpu(),
            "cell": batch["cell"].cpu(),
            "atomic_numbers": batch["conditioning"]["atomic_numbers"].cpu(),
            "atom_mask": batch["atom_mask"].cpu(),
        }

    def write_prediction_chunk(
        self,
        trainer: Any,
        prediction: dict[str, Any],
        batch_indices: Optional[Sequence[int]],
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Write one sample chunk payload immediately."""
        rank = trainer.global_rank
        save_dir = self.output_dir / "raw" / f"rank_{rank}"
        save_dir.mkdir(parents=True, exist_ok=True)

        sample_offset = int(prediction["sample_offset"])
        payload = {
            "pred": {key: value.cpu() for key, value in prediction["pred"].items()},
            "sample_offset": sample_offset,
            "chunk_size": int(prediction["chunk_size"]),
            "ref": self._reference_payload(batch),
            "dataset_index": self._dataset_index(batch, batch_indices),
            "metadata": batch.get(
                "metadata",
                [{} for _ in range(int(batch["num_atoms"].shape[0]))],
            ),
        }
        torch.save(
            payload,
            save_dir / f"batch_{batch_idx:06d}_sample_{sample_offset:06d}.pt",
        )

    def write_on_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        prediction: Sequence[dict[str, Any]] | None,
        batch_indices: Optional[Sequence[int]],
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Write raw sample-chunk payloads and defer row expansion to collation."""
        if prediction is None:
            return
        for chunk in prediction:
            self.write_prediction_chunk(
                trainer=trainer,
                prediction=chunk,
                batch_indices=batch_indices,
                batch=batch,
                batch_idx=batch_idx,
            )

    def write_on_epoch_end(
        self,
        trainer: Any,
        pl_module: Any,
        predictions: Sequence[Sequence[dict[str, torch.Tensor]]],
        batch_indices: Optional[Sequence[Sequence[int]]],
    ) -> None:
        """No-op because this writer stores bundle payloads per batch."""
        return
