# ruff: noqa: F722,F821

from abc import ABC, abstractmethod
from typing import Any

from src.utils.tensor_typing import Bool, Float


class Scaler(ABC):
    """Base class for coordinate/lattice unit scaling."""

    lattice_repr: str = "params"

    def setup(self, stats: dict[str, Any]) -> None:
        """Optional hook to configure from dataset statistics."""
        pass

    @abstractmethod
    def scale_coords(
        self,
        cart_coords: Float["b n 3"],
        atom_mask: Bool["b n"] | None = None,
    ) -> Float["b n 3"]:
        """Scale coordinates from raw units to model units."""
        ...

    @abstractmethod
    def scale_lattice(
        self,
        lattice: Float["b d"],
    ) -> Float["b d"]:
        """Scale lattice from raw units to model units."""
        ...

    @abstractmethod
    def unscale_coords(
        self,
        cart_coords: Float["b n 3"],
        atom_mask: Bool["b n"] | None = None,
    ) -> Float["b n 3"]:
        """Scale coordinates from model units back to raw units."""
        ...

    @abstractmethod
    def unscale_lattice(
        self,
        lattice: Float["b d"],
    ) -> Float["b d"]:
        """Scale lattice from model units back to raw units."""
        ...
