"""Single representation embedder.

Builds a `[B, N, dim]` tensor `c` from atom-level conditioning.
"""

# ruff: noqa: F722,F821

import torch
import torch.nn as nn
from einops import rearrange

from src.data.constants import (
    COVALENT_RADIUS_CORDERO_ANGSTROM,
    ELECTRONEGATIVITY_PAULING,
    ELEMENT_BLOCK,
    ELEMENT_GROUP,
    ELEMENT_PERIOD,
    NUM_ATOM_TYPES,
    VDW_RADIUS_ALVAREZ_ANGSTROM,
)
from src.data.components.conditioning import (
    NUM_ATOM_CHIRALITY_MODEL_TYPES,
    NUM_SPACEGROUP_MODEL_TYPES,
)
from src.utils.tensor_typing import Float, Int


def _float_lookup(values: list[float]) -> Float["z"]:
    """Build a float element lookup."""
    return torch.tensor(values, dtype=torch.float32)


def _int_lookup(values: list[int]) -> Int["z"]:
    """Build an integer element lookup."""
    return torch.tensor(values, dtype=torch.long)


class SingleConditionEmbedder(nn.Module):
    """Build single representation c from atom-level conditioning."""

    def __init__(
        self,
        dim: int,
        num_atom_types: int = NUM_ATOM_TYPES,
        num_chirality_types: int = NUM_ATOM_CHIRALITY_MODEL_TYPES,
        num_spacegroup_types: int = NUM_SPACEGROUP_MODEL_TYPES,
    ) -> None:
        """Initialize atom-level conditioning embedding layers."""
        super().__init__()

        # Always-on atom-type table (index 0 = padding)
        self.embed_atom = nn.Embedding(num_atom_types, dim, padding_idx=0)

        # Element-level categorical descriptors
        self.embed_period = nn.Embedding(8, dim, padding_idx=0)
        self.embed_group = nn.Embedding(19, dim, padding_idx=0)
        self.embed_block = nn.Embedding(5, dim, padding_idx=0)

        # Element-level scalar descriptors
        self.embed_covalent_radius = nn.Linear(1, dim, bias=False)
        self.embed_vdw_radius = nn.Linear(1, dim, bias=False)
        self.embed_electronegativity = nn.Linear(1, dim, bias=False)

        # Template coordinates
        self.embed_template = nn.Linear(3, dim, bias=False)

        # Formal charges as raw scalar -> linear (Boltz-style)
        self.embed_charges = nn.Linear(1, dim, bias=False)

        # Optional categorical descriptors
        self.embed_chirality = nn.Embedding(num_chirality_types, dim, padding_idx=0)
        self.embed_spacegroup = nn.Embedding(num_spacegroup_types, dim, padding_idx=0)

        self.register_buffer(
            "covalent_radius_lookup",
            _float_lookup(COVALENT_RADIUS_CORDERO_ANGSTROM),
            persistent=False,
        )
        self.register_buffer(
            "vdw_radius_lookup",
            _float_lookup(VDW_RADIUS_ALVAREZ_ANGSTROM),
            persistent=False,
        )
        self.register_buffer(
            "electronegativity_lookup",
            _float_lookup(ELECTRONEGATIVITY_PAULING),
            persistent=False,
        )
        self.register_buffer(
            "period_lookup", _int_lookup(ELEMENT_PERIOD), persistent=False
        )
        self.register_buffer(
            "group_lookup", _int_lookup(ELEMENT_GROUP), persistent=False
        )
        self.register_buffer(
            "block_lookup", _int_lookup(ELEMENT_BLOCK), persistent=False
        )

    def forward(self, noisy_batch: dict) -> Float["b n d"]:
        """Embed atom types and add organic conditioning terms."""
        conditioning = noisy_batch["conditioning"]
        conditioning_masks = noisy_batch["conditioning_masks"]
        atom_types = conditioning["atomic_numbers"]
        atom_mask = noisy_batch["atom_mask"]

        # Atom types, always present
        c = self.embed_atom(atom_types)

        # Periodic-table categorical descriptors
        c = c + self.embed_period(self.period_lookup[atom_types])
        c = c + self.embed_group(self.group_lookup[atom_types])
        c = c + self.embed_block(self.block_lookup[atom_types])

        # Periodic-table scalar descriptors
        covalent_radius = rearrange(
            self.covalent_radius_lookup[atom_types], "b n -> b n 1"
        )
        vdw_radius = rearrange(self.vdw_radius_lookup[atom_types], "b n -> b n 1")
        electronegativity = rearrange(
            self.electronegativity_lookup[atom_types], "b n -> b n 1"
        )
        c = c + self.embed_covalent_radius(covalent_radius)
        c = c + self.embed_vdw_radius(vdw_radius)
        c = c + self.embed_electronegativity(electronegativity)

        # Formal charges
        charges_scalar = rearrange(
            conditioning["formal_charges"].float(), "b n -> b n 1"
        )
        c = c + self.embed_charges(charges_scalar)

        # Optional template coords
        template_gate = rearrange(conditioning_masks["template"], "b -> b 1 1")
        template = self.embed_template(conditioning["template_coords"])
        c = c + template * template_gate

        # Optional atom chirality labels
        c = c + self.embed_chirality(conditioning["atom_chirality"])

        # Optional global space-group number
        spacegroup = rearrange(
            self.embed_spacegroup(conditioning["spacegroup_number"]),
            "b d -> b 1 d",
        )
        c = c + spacegroup * rearrange(atom_mask.to(c.dtype), "b n -> b n 1")

        return c
