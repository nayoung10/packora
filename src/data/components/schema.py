"""Conditioning field schema for organic crystal batching.

Each entry says how to pad one field of `ConditioningTensors` when stacking
samples into a batch. The canonical per-sample structure (which fields
exist, their dtypes, etc.) is defined in `src/data/types.py`. This module
is just the padding metadata used by `collate_fn`.
"""

from dataclasses import dataclass
from typing import Any, Tuple

# Re-export for callers that used the old import path.


@dataclass(frozen=True)
class FieldSpec:
    name: str  # field name on ConditioningTensors
    kind: str  # "scalar" (N,) | "vec3" (N, 3) | "pair" (N, N) | "global" ()
    fill: Any  # neutral value used when padding to batch's max-N


# Order matches FIELD_NAMES in src/data/types.py — used by collate_fn.
CONDITIONING_FIELDS: Tuple[FieldSpec, ...] = (
    FieldSpec("atomic_numbers", "scalar", 0),
    FieldSpec("template_coords", "vec3", 0.0),
    FieldSpec("template_present", "global", False),
    FieldSpec("formal_charges", "scalar", 0),
    FieldSpec("atom_chirality", "scalar", 0),
    FieldSpec("membership", "scalar", -1),
    FieldSpec("bond_adj", "pair", False),
    FieldSpec("bond_type_adj", "pair", 0),
    FieldSpec("bond_stereochemistry_adj", "pair", 0),
    FieldSpec("stereochemistry_present", "global", False),
    FieldSpec("spacegroup_number", "global", 0),
    FieldSpec("spacegroup_present", "global", False),
)
