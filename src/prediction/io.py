"""Utilities for collation and serialization of CSP prediction artifacts."""

# ruff: noqa: F821

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange, repeat

from src.utils.tensor_typing import Int


@dataclass(frozen=True)
class PredictionBundle:
    """Container for collated prediction rows with aligned references and metadata.

    Shapes use `B` = final number of rows after sort + dedup and `N` = padded max atom count.
    Tensor dicts are expected to include:
    - `cart_coords`: `Float[B, N, 3]`
    - `lattice`: `Float[B, 6]`
    - `cell`: `Float[B, 9]`
    - `atomic_numbers`: `Int[B, N]`
    - `atom_mask`: `Bool[B, N]`
    """

    pred: dict[str, torch.Tensor]  # predicted tensors, shape conventions above
    ref: dict[str, torch.Tensor]  # reference tensors, same row order/shape conventions
    dataset_indices: np.ndarray  # np.int64, shape (B,), original dataset row ids
    sample_indices: (
        np.ndarray
    )  # np.int64, shape (B,), generated sample ids per source row
    metadata: dict[str, list[Any]]  # columnar metadata, each column length must equal B


def pad_and_cat(preds: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length tensors to common shape and concatenate on batch dim."""
    if not preds:
        raise ValueError("Cannot pad and concatenate an empty list.")

    keys = list(preds[0].keys())

    # Track max size for each non-batch axis per key
    max_sizes: dict[str, list[int]] = {}
    for pred in preds:
        for key in keys:
            shape_tail = list(pred[key].shape[1:])
            if key not in max_sizes:
                max_sizes[key] = shape_tail
            else:
                for dim in range(len(shape_tail)):
                    max_sizes[key][dim] = max(max_sizes[key][dim], shape_tail[dim])

    def _pad(tensor: torch.Tensor, target: list[int]) -> torch.Tensor:
        """Pad tensor to target size for all axes after batch dimension."""
        pad_args: list[int] = []
        for dim in reversed(range(len(target))):
            pad_args.extend([0, target[dim] - tensor.shape[dim + 1]])
        if any(value > 0 for value in pad_args):
            return torch.nn.functional.pad(tensor, pad_args)
        return tensor

    return {
        key: torch.cat([_pad(pred[key], max_sizes[key]) for pred in preds], dim=0)
        for key in keys
    }


def metadata_rows_to_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert row-oriented metadata dictionaries to column-oriented lists."""
    if not rows:
        return {}

    keys: list[str] = sorted({key for row in rows for key in row.keys()})
    return {key: [row.get(key) for row in rows] for key in keys}


def expand_source_major_tensor(
    tensor: torch.Tensor,
    multiplicity: int,
) -> torch.Tensor:
    """Repeat one batch-major tensor in source-major row order."""
    repeated = repeat(tensor, "b ... -> b m ...", m=multiplicity)
    return rearrange(repeated, "b m ... -> (b m) ...")


def build_sample_index(
    batch_size: int,
    multiplicity: int,
) -> Int["b"]:
    """Build source-major sample indices for one expanded prediction batch."""
    if multiplicity < 1:
        raise ValueError("multiplicity must be >= 1.")

    sample_index = repeat(
        torch.arange(multiplicity, dtype=torch.long),
        "m -> b m",
        b=batch_size,
    )
    return rearrange(sample_index, "b m -> (b m)")


def expand_prediction_rows(
    pred: dict[str, torch.Tensor],
    ref: dict[str, torch.Tensor],
    dataset_index: Int["b"],
    metadata_rows: list[dict[str, Any]],
    sample_offset: int = 0,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    Int["b"],
    Int["b"],
    list[dict[str, Any]],
]:
    """Expand one raw batch payload to source-major generated rows."""
    batch_size = int(dataset_index.shape[0])
    if batch_size == 0:
        raise ValueError("dataset_index must contain at least one row.")

    if not metadata_rows:
        metadata_rows = [{} for _ in range(batch_size)]
    if len(metadata_rows) != batch_size:
        raise ValueError("metadata row count must match dataset_index length.")

    row_count = int(pred["cart_coords"].shape[0])
    if row_count % batch_size != 0:
        raise ValueError(
            "Prediction row count must be divisible by the source batch size."
        )

    multiplicity = row_count // batch_size
    sample_index = build_sample_index(batch_size, multiplicity) + int(sample_offset)
    expanded_ref = {
        key: expand_source_major_tensor(value, multiplicity)
        for key, value in ref.items()
    }
    expanded_dataset_index = expand_source_major_tensor(dataset_index, multiplicity)
    expanded_metadata = [row for row in metadata_rows for _ in range(multiplicity)]
    return (
        pred,
        expanded_ref,
        expanded_dataset_index,
        sample_index,
        expanded_metadata,
    )


def save_prediction_bundle(path: Path, bundle: PredictionBundle) -> None:
    """Serialize a PredictionBundle to disk using torch.save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pred": bundle.pred,
        "ref": bundle.ref,
        "dataset_indices": bundle.dataset_indices,
        "sample_indices": bundle.sample_indices,
        "metadata": bundle.metadata,
    }
    torch.save(payload, path)


def _as_int64_numpy(values: Any, field_name: str) -> np.ndarray:
    """Normalize numpy-or-tensor integer arrays to np.int64."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be a 1D array.")
    return array.astype(np.int64, copy=False)


def load_prediction_bundle(path: Path) -> PredictionBundle:
    """Load PredictionBundle payload and validate required keys and lengths."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction bundle not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "pred",
        "ref",
        "dataset_indices",
        "metadata",
    }
    missing = required.difference(payload.keys())
    if missing:
        raise KeyError(f"Prediction bundle missing required keys: {sorted(missing)}")

    dataset_indices = _as_int64_numpy(payload["dataset_indices"], "dataset_indices")
    sample_indices_payload = payload.get("sample_indices")
    if sample_indices_payload is None:
        sample_indices = np.zeros(dataset_indices.shape[0], dtype=np.int64)
    else:
        sample_indices = _as_int64_numpy(sample_indices_payload, "sample_indices")
    metadata = dict(payload["metadata"])

    n_rows = int(dataset_indices.shape[0])
    if int(sample_indices.shape[0]) != n_rows:
        raise ValueError("sample_indices length must match dataset_indices length.")

    for key, values in metadata.items():
        if len(values) != n_rows:
            raise ValueError(
                f"metadata column '{key}' length ({len(values)}) does not match {n_rows} rows.",
            )

    return PredictionBundle(
        pred=dict(payload["pred"]),
        ref=dict(payload["ref"]),
        dataset_indices=dataset_indices,
        sample_indices=sample_indices,
        metadata=metadata,
    )


def reorder_tensor_dict(
    payload: dict[str, torch.Tensor],
    order: Int["b"],
) -> dict[str, torch.Tensor]:
    """Reorder each tensor in payload along batch axis using one index tensor."""
    return {key: value.index_select(0, order) for key, value in payload.items()}


def reorder_metadata_columns(
    metadata: dict[str, list[Any]],
    order: Int["b"],
) -> dict[str, list[Any]]:
    """Reorder metadata columns using one shared row order."""
    order_list = [int(index) for index in order.detach().cpu().tolist()]
    return {
        key: [values[idx] for idx in order_list] for key, values in metadata.items()
    }


def _select_metadata_rows(
    metadata: dict[str, list[Any]],
    row_ids: Int["b"],
) -> dict[str, list[Any]]:
    """Select metadata rows by integer row ids."""
    keep = [int(index) for index in row_ids.detach().cpu().tolist()]
    return {key: [values[idx] for idx in keep] for key, values in metadata.items()}


def build_prediction_bundle(
    pred_batches: list[dict[str, torch.Tensor]],
    ref_batches: list[dict[str, torch.Tensor]],
    dataset_index_batches: list[Int["b"]],
    sample_index_batches: list[Int["b"]],
    metadata_rows: list[dict[str, Any]],
) -> PredictionBundle:
    """Collate, sort, and package tensors + metadata into one bundle."""
    preds = pad_and_cat(pred_batches)
    refs = pad_and_cat(ref_batches)

    dataset_indices = torch.cat(dataset_index_batches, dim=0).to(dtype=torch.long)
    sample_indices = torch.cat(sample_index_batches, dim=0).to(dtype=torch.long)
    metadata_cols = metadata_rows_to_columns(metadata_rows)

    # Stable row ordering keeps tensor rows and metadata joins deterministic
    order_np = np.lexsort(
        (
            sample_indices.detach().cpu().numpy(),
            dataset_indices.detach().cpu().numpy(),
        ),
    )
    order = torch.from_numpy(order_np.astype(np.int64))
    preds = reorder_tensor_dict(preds, order)
    refs = reorder_tensor_dict(refs, order)
    dataset_indices = dataset_indices.index_select(0, order)
    sample_indices = sample_indices.index_select(0, order)
    metadata_cols = reorder_metadata_columns(metadata_cols, order)

    # Distributed samplers may pad with duplicated row identities on some ranks
    if dataset_indices.numel() > 0:
        keep_mask = torch.ones_like(dataset_indices, dtype=torch.bool)
        same_dataset = dataset_indices[1:] == dataset_indices[:-1]
        same_sample = sample_indices[1:] == sample_indices[:-1]
        keep_mask[1:] = ~(same_dataset & same_sample)
        keep_rows = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
        preds = reorder_tensor_dict(preds, keep_rows)
        refs = reorder_tensor_dict(refs, keep_rows)
        dataset_indices = dataset_indices.index_select(0, keep_rows)
        sample_indices = sample_indices.index_select(0, keep_rows)
        metadata_cols = _select_metadata_rows(metadata_cols, keep_rows)

    dataset_indices_np = dataset_indices.detach().cpu().numpy().astype(np.int64)
    sample_indices_np = sample_indices.detach().cpu().numpy().astype(np.int64)

    return PredictionBundle(
        pred=preds,
        ref=refs,
        dataset_indices=dataset_indices_np,
        sample_indices=sample_indices_np,
        metadata=metadata_cols,
    )
