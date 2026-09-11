"""Pairwise representation embedder.

Builds a `[B, N, N, dim_pair]` tensor `z` from organic pair conditioning plus
the single representation `s`.
"""

# ruff: noqa: F722,F821

import torch.nn as nn
from einops import rearrange

from src.data.components.conditioning import (
    NUM_BOND_STEREOCHEMISTRY_MODEL_TYPES,
    NUM_BOND_TYPE_MODEL_TYPES,
)
from src.models.components.activations import ActivationType, build_activation
from src.models.components.norms import NormType, build_norm
from src.utils.tensor_typing import Bool, Float


class PairwiseConditionEmbedder(nn.Module):
    """Build pairwise representation z from single repr + optional conditioning."""

    def __init__(
        self,
        dim_single: int,
        dim_pair: int,
        norm_type: NormType = "layernorm",
        activation: ActivationType = "silu",
    ) -> None:
        """Initialize pairwise conditioning embedding layers."""
        super().__init__()
        self.dim_pair = dim_pair

        # Template-coord branch
        self.embed_template_d = nn.Linear(3, dim_pair, bias=False)
        self.embed_template_dist = nn.Linear(1, dim_pair, bias=False)

        # Bond branch
        self.embed_bond = nn.Embedding(
            NUM_BOND_TYPE_MODEL_TYPES,
            dim_pair,
            padding_idx=0,
        )
        self.embed_stereochemistry = nn.Embedding(
            NUM_BOND_STEREOCHEMISTRY_MODEL_TYPES,
            dim_pair,
            padding_idx=0,
        )

        # Single -> pair projection (always on)
        self.proj_q = nn.Linear(dim_single, dim_pair, bias=False)
        self.proj_k = nn.Linear(dim_single, dim_pair, bias=False)

        # Refinement MLP (per-position)
        self.mlp = nn.Sequential(
            build_norm(dim_pair, norm_type=norm_type),
            nn.Linear(dim_pair, dim_pair),
            build_activation(activation),
            nn.Linear(dim_pair, dim_pair),
        )

    def forward(
        self,
        s: Float["b n d"],
        atom_mask: Bool["b n"],
        conditioning: dict,
        conditioning_masks: dict[str, Float["b"]],
    ) -> Float["b n n dp"]:
        """Compute pairwise representation z from s + conditioning."""
        # Pair mask v where atoms are real and from the same molecule
        membership = conditioning["membership"]
        same_group = rearrange(membership, "b i -> b i 1") == rearrange(
            membership, "b j -> b 1 j"
        )
        real_pair = rearrange(atom_mask, "b i -> b i 1") & rearrange(
            atom_mask, "b j -> b 1 j"
        )
        valid_mol_pair = same_group & real_pair
        pair_gate = rearrange(valid_mol_pair.to(s.dtype), "b i j -> b i j 1")

        # Template-coord branch
        t = conditioning["template_coords"]
        d = rearrange(t, "b i c -> b i 1 c") - rearrange(t, "b j c -> b 1 j c")
        d_norm = 1.0 / (1.0 + d.square().sum(-1, keepdim=True))
        template = self.embed_template_d(d) + self.embed_template_dist(d_norm)
        template_gate = rearrange(conditioning_masks["template"], "b -> b 1 1 1")
        template = template * (pair_gate * template_gate)

        # Bond branch
        bond = self.embed_bond(conditioning["bond_type_adj"]) * pair_gate

        # Stereochemistry branch
        bond_adj = conditioning.get("bond_adj", conditioning["bond_type_adj"] > 0)
        valid_bond_pair = bond_adj & valid_mol_pair
        stereochemistry_gate = rearrange(
            valid_bond_pair.to(s.dtype),
            "b i j -> b i j 1",
        )
        stereochemistry = (
            self.embed_stereochemistry(conditioning["bond_stereochemistry_adj"])
            * stereochemistry_gate
        )
        z = template + bond + stereochemistry

        # Single-to-pair projection
        q = self.proj_q(s)
        k = self.proj_k(s)
        z = z + rearrange(q, "b i d -> b i 1 d") + rearrange(k, "b j d -> b 1 j d")

        # Residual MLP
        z = z + self.mlp(z)

        return z
