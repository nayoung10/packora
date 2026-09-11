# ruff: noqa: F722,F821

from typing import Optional

import torch.nn as nn

from src.utils.tensor_typing import Float


class SingleEmbedder(nn.Module):
    """Create initial single representation from condition and noisy inputs."""

    def __init__(
        self,
        single_condition_embedder: nn.Module,
        coord_embedder: nn.Module,
        lattice_embedder: nn.Module,
        positional_embedder: Optional[nn.Module] = None,
    ) -> None:
        """Initialize condition, coordinate, lattice, and position embedders."""
        super().__init__()
        self.single_condition_embedder = single_condition_embedder
        self.coord_embedder = coord_embedder
        self.lattice_embedder = lattice_embedder
        self.positional_embedder = positional_embedder

    def forward(
        self,
        noisy_batch: dict,
    ) -> tuple[Float["b n d"], Float["b n d"]]:
        """Build condition-only c and noisy-input single representation s."""
        c = self.condition_embedding(noisy_batch)
        s = self.base_embedding(noisy_batch, c) + self.noisy_embedding(noisy_batch)
        return c, s

    def condition_embedding(self, noisy_batch: dict) -> Float["b n d"]:
        """Build the condition-only single representation."""
        return self.single_condition_embedder(noisy_batch)

    def base_embedding(
        self,
        noisy_batch: dict,
        c: Float["b n d"],
    ) -> Float["b n d"]:
        """Build the time-independent single representation."""
        s = c
        if self.positional_embedder is not None:
            s = s + self.positional_embedder(noisy_batch["indices"])
        return s

    def noisy_embedding(self, noisy_batch: dict) -> Float["b n d"]:
        """Build the noisy coordinate and lattice single representation."""
        num_atoms = noisy_batch["x_t"].shape[1]
        return self.coord_embedder(noisy_batch["x_t"]) + self.lattice_embedder(
            noisy_batch["l_t"],
            num_atoms,
        )


class PairwiseEmbedder(nn.Module):
    """Create initial pairwise representation from condition and geometry."""

    def __init__(
        self,
        pairwise_condition_embedder: Optional[nn.Module] = None,
        geometry_pair_embedder: Optional[nn.Module] = None,
    ) -> None:
        """Initialize optional condition and geometry pair embedders."""
        super().__init__()
        self.pairwise_condition_embedder = pairwise_condition_embedder
        self.geometry_pair_embedder = geometry_pair_embedder

    @property
    def dim_pair(self) -> int | None:
        """Return pair representation width when condition pair features exist."""
        if self.pairwise_condition_embedder is None:
            return None
        return int(self.pairwise_condition_embedder.dim_pair)

    def forward(
        self,
        noisy_batch: dict,
        c: Float["b n d"],
    ) -> Optional[Float["b n n dp"]]:
        """Build pair representation z from condition and optional geometry."""
        z = self.condition_embedding(noisy_batch, c)
        return self.add_geometry(noisy_batch, z)

    def condition_embedding(
        self,
        noisy_batch: dict,
        c: Float["b n d"],
    ) -> Optional[Float["b n n dp"]]:
        """Build the time-independent pair condition representation."""
        if self.pairwise_condition_embedder is None:
            return None

        return self.pairwise_condition_embedder(
            s=c,
            atom_mask=noisy_batch["atom_mask"],
            conditioning=noisy_batch["conditioning"],
            conditioning_masks=noisy_batch["conditioning_masks"],
        )

    def geometry_embedding(self, noisy_batch: dict) -> Optional[Float["b n n dp"]]:
        """Build the noisy geometry pair representation when configured."""
        if self.geometry_pair_embedder is None:
            return None
        if self.pairwise_condition_embedder is None:
            raise RuntimeError(
                "Geometry pair embedding requires pairwise condition features."
            )
        return self.geometry_pair_embedder(noisy_batch)

    def add_geometry(
        self,
        noisy_batch: dict,
        z: Optional[Float["b n n dp"]],
    ) -> Optional[Float["b n n dp"]]:
        """Add noisy geometry features to a pair representation."""
        if self.geometry_pair_embedder is not None:
            geom = self.geometry_embedding(noisy_batch)
            if z is None or geom is None:
                raise RuntimeError(
                    "Geometry pair embedding requires pairwise condition features."
                )
            z = z + geom
        return z


class InputEmbedder(nn.Module):
    """Combine single and pairwise embedders into model input representations."""

    def __init__(
        self,
        single_embedder: SingleEmbedder,
        pairwise_embedder: Optional[PairwiseEmbedder] = None,
    ) -> None:
        """Initialize single and optional pairwise input embedders."""
        super().__init__()
        self.single_embedder = single_embedder
        self.pairwise_embedder = pairwise_embedder

    def _conditioning_masks(self, conditioning: dict) -> dict[str, Float["b"]]:
        """Build conditioning masks expected by condition embedders."""
        return {
            "template": conditioning["template_mask"],
            "stereochemistry": conditioning["stereochemistry_mask"],
            "spacegroup": conditioning["spacegroup_mask"],
        }

    def forward(
        self,
        noisy_batch: dict,
    ) -> tuple[Float["b n d"], Optional[Float["b n n dp"]]]:
        """Build token and pair representations from a noisy batch."""
        c, s, z = self.forward_parts(noisy_batch)
        return s, z

    def forward_parts(
        self,
        noisy_batch: dict,
    ) -> tuple[
        Float["b n d"],
        Float["b n d"],
        Optional[Float["b n n dp"]],
    ]:
        """Build condition, token, and pair representations from a noisy batch."""
        conditioned_batch = dict(noisy_batch)
        conditioned_batch["conditioning_masks"] = self._conditioning_masks(
            noisy_batch["conditioning"]
        )

        c, s = self.single_embedder(conditioned_batch)
        z = None
        if self.pairwise_embedder is not None:
            z = self.pairwise_embedder(conditioned_batch, c)
        return c, s, z

    def base_parts(
        self,
        noisy_batch: dict,
    ) -> tuple[
        dict,
        Float["b n d"],
        Float["b n d"],
        Optional[Float["b n n dp"]],
    ]:
        """Build time-independent input representations."""
        conditioned_batch = dict(noisy_batch)
        conditioned_batch["conditioning_masks"] = self._conditioning_masks(
            noisy_batch["conditioning"]
        )
        c = self.single_embedder.condition_embedding(conditioned_batch)
        s = self.single_embedder.base_embedding(conditioned_batch, c)
        z = None
        if self.pairwise_embedder is not None:
            z = self.pairwise_embedder.condition_embedding(conditioned_batch, c)
        return conditioned_batch, c, s, z

    def base_representations(
        self,
        noisy_batch: dict,
    ) -> tuple[Float["b n d"], Optional[Float["b n n dp"]]]:
        """Build time-independent single and pair representations."""
        _, _, s, z = self.base_parts(noisy_batch)
        return s, z

    def noisy_single_embedding(self, noisy_batch: dict) -> Float["b n d"]:
        """Build noisy coordinate and lattice single features."""
        return self.single_embedder.noisy_embedding(noisy_batch)

    def add_geometry(
        self,
        noisy_batch: dict,
        z: Optional[Float["b n n dp"]],
    ) -> Optional[Float["b n n dp"]]:
        """Add noisy geometry pair features when configured."""
        if self.pairwise_embedder is None:
            return z
        return self.pairwise_embedder.add_geometry(noisy_batch, z)


__all__ = ["InputEmbedder", "PairwiseEmbedder", "SingleEmbedder"]
