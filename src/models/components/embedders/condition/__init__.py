# ruff: noqa: F722,F821

from typing import Optional

import torch.nn as nn

from src.utils.tensor_typing import Bool, Float


class ConditionEmbedder(nn.Module):
    """Build condition-only single and pairwise representations."""

    def __init__(
        self,
        single_embedder: nn.Module,
        pairwise_embedder: Optional[nn.Module] = None,
    ) -> None:
        """Initialize single and optional pairwise condition embedders."""
        super().__init__()
        self.single_embedder = single_embedder
        self.pairwise_embedder = pairwise_embedder

    def forward(
        self,
        noisy_batch: dict,
    ) -> tuple[Float["b n d"], Optional[Float["b n n dp"]]]:
        """Embed condition-only token and optional pair features."""
        conditioning = noisy_batch["conditioning"]
        conditioning_masks = {
            "template": conditioning["template_mask"],
            "stereochemistry": conditioning["stereochemistry_mask"],
            "spacegroup": conditioning["spacegroup_mask"],
        }
        conditioned_batch = dict(noisy_batch)
        conditioned_batch["conditioning_masks"] = conditioning_masks

        c = self.single_embedder(conditioned_batch)
        z = self._embed_pairwise(
            c,
            conditioned_batch["atom_mask"],
            conditioned_batch["conditioning"],
            conditioning_masks,
        )
        return c, z

    def _embed_pairwise(
        self,
        c: Float["b n d"],
        atom_mask: Bool["b n"],
        conditioning: dict,
        conditioning_masks: dict[str, Float["b"]],
    ) -> Optional[Float["b n n dp"]]:
        """Embed pairwise conditioning when configured."""
        if self.pairwise_embedder is None:
            return None
        return self.pairwise_embedder(
            s=c,
            atom_mask=atom_mask,
            conditioning=conditioning,
            conditioning_masks=conditioning_masks,
        )


__all__ = ["ConditionEmbedder"]
