# ruff: noqa: F722,F821

from typing import Any, Dict, List, Optional

import torch

from src.data.components.conditioning import (
    ConditioningPolicy,
    prepare_model_conditioning,
)
from src.data.components.schema import CONDITIONING_FIELDS
from src.data.types import ConditioningTensors
from src.utils.tensor_typing import Shaped


def _pad_atom_dim(
    tensor: Shaped["..."],
    n_max: int,
    kind: str,
    fill: Any,
) -> Shaped["..."]:
    """Pad a per-atom tensor along its atom dimension to size n_max."""
    n = tensor.shape[0]
    if kind == "scalar":
        out = torch.full((n_max,), fill, dtype=tensor.dtype)
        out[:n] = tensor
    elif kind == "vec3":
        out = torch.full((n_max, 3), fill, dtype=tensor.dtype)
        out[:n] = tensor
    elif kind == "pair":
        out = torch.full((n_max, n_max), fill, dtype=tensor.dtype)
        out[:n, :n] = tensor
    else:
        raise ValueError(f"Unknown pad kind: {kind}")
    return out


def _resolve_padded_atom_count(
    num_atoms: Shaped["b"],
    pad_to_atom_buckets: Optional[list[int]],
) -> int:
    """Return the atom dimension used for this collated batch."""
    max_atoms = int(num_atoms.max().item())
    if pad_to_atom_buckets is None:
        return max_atoms

    buckets = sorted({int(bucket) for bucket in pad_to_atom_buckets})
    if not buckets:
        raise ValueError("pad_to_atom_buckets must not be empty.")
    for bucket in buckets:
        if max_atoms <= bucket:
            return bucket
    raise ValueError(
        f"Batch has {max_atoms} atoms, above largest bucket {buckets[-1]}."
    )


def collate_fn(
    batch: List[Dict[str, Any]],
    drop_fields: Optional[list[str]] = None,
    conditioning_policy: Optional[dict[str, Any]] = None,
    conditioning_context: Optional[str] = None,
    pad_to_atom_buckets: Optional[list[int]] = None,
) -> Dict[str, Any]:
    """Batch material structures by padding to the batch max or a fixed bucket.

    Output shapes (B = batch size, N = max num_atoms in this batch, K = num
    conditioning fields):

        cart_coords      (B, N, 3)     float — pad value 0.0
        frac_coords      (B, N, 3)     float — pad value 0.0
        lattice          (B, 6)        float
        cell             (B, 3, 3)     float
        atom_mask        (B, N)        bool  — True for real atoms
        num_atoms        (B,)          long
        dataset_index    (B,)          long  — only if input had `dataset_index`
        indices          (B, N)        long  — only if input had `indices`

        conditioning: dict with
            atomic_numbers  (B, N)     long  — pad value 0
            template_coords (B, N, 3)  float
            template_present (B,)       bool
            formal_charges  (B, N)     long
            atom_chirality  (B, N)     long
            membership      (B, N)     long  — pad value -1
            bond_adj        (B, N, N)  bool
            bond_type_adj   (B, N, N)  long
            bond_stereochemistry_adj (B, N, N) long
            spacegroup_number (B,)     long
            template_mask, stereochemistry_mask, spacegroup_mask (B,) float
    """
    dropped = set(drop_fields or [])
    prepare_conditioning = conditioning_policy is not None
    num_atoms = torch.stack([s["num_atoms"] for s in batch])  # (B,)
    n_max = _resolve_padded_atom_count(num_atoms, pad_to_atom_buckets)  # N
    has_indices = "indices" in batch[0]
    has_dataset_index = "dataset_index" in batch[0]
    has_metadata = "metadata" in batch[0]

    # Main fields (coords, indices, atom_mask)
    cart_coords = torch.stack(
        [  # (B, N, 3)
            _pad_atom_dim(s["cart_coords"], n_max, "vec3", 0.0) for s in batch
        ]
    )
    frac_coords = None
    if "frac_coords" not in dropped:
        frac_coords = torch.stack(
            [  # (B, N, 3)
                _pad_atom_dim(s["frac_coords"], n_max, "vec3", 0.0) for s in batch
            ]
        )
    atom_mask = torch.stack(
        [  # (B, N)
            torch.cat(
                [
                    torch.ones(int(s["num_atoms"].item()), dtype=torch.bool),
                    torch.zeros(n_max - int(s["num_atoms"].item()), dtype=torch.bool),
                ]
            )
            for s in batch
        ]
    )

    result: Dict[str, Any] = {
        "cart_coords": cart_coords,  # (B, N, 3)
        "cell": torch.stack([s["cell"] for s in batch]),  # (B, 3, 3)
        "atom_mask": atom_mask,  # (B, N)
        "num_atoms": num_atoms,  # (B,)
    }
    if frac_coords is not None:
        result["frac_coords"] = frac_coords  # (B, N, 3)
    if "lattice" not in dropped:
        result["lattice"] = torch.stack([s["lattice"] for s in batch])  # (B, 6)

    # (Optional) Indices for positional encoding
    if has_indices:
        result["indices"] = torch.stack(
            [  # (B, N)
                _pad_atom_dim(s["indices"], n_max, "scalar", 0) for s in batch
            ]
        )

    if has_dataset_index and "dataset_index" not in dropped:
        result["dataset_index"] = torch.stack([s["dataset_index"] for s in batch])

    # (Optional) auxiliary conditioning
    sample_conds: List[ConditioningTensors] = [s["conditioning"] for s in batch]
    conditioning: Dict[str, Shaped["..."]] = {}
    for spec in CONDITIONING_FIELDS:
        if not prepare_conditioning and f"conditioning.{spec.name}" in dropped:
            continue
        if spec.kind == "global":
            conditioning[spec.name] = torch.stack(
                [getattr(c, spec.name) for c in sample_conds]
            )
        else:
            conditioning[spec.name] = torch.stack(
                [  # (B, ...)
                    _pad_atom_dim(getattr(c, spec.name), n_max, spec.kind, spec.fill)
                    for c in sample_conds
                ]
            )
    if prepare_conditioning:
        context = str(conditioning_context or "predict")
        policy = ConditioningPolicy.from_config(conditioning_policy)
        conditioning = prepare_model_conditioning(
            conditioning=conditioning,
            atom_mask=atom_mask,
            policy=policy,
            context=context,
            drop_fields=dropped,
        )
    result["conditioning"] = conditioning
    if has_metadata:
        result["metadata"] = [s["metadata"] for s in batch]

    return result
