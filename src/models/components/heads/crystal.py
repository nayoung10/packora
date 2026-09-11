import torch.nn as nn

from src.utils.tensor_typing import Bool, Float


class CrystalHeads(nn.Module):
    """Predict coordinates and lattice from token representations."""

    def __init__(
        self,
        coord_head: nn.Module,
        lattice_head: nn.Module,
    ) -> None:
        """Initialize coordinate and lattice heads."""
        super().__init__()
        self.coord_head = coord_head
        self.lattice_head = lattice_head

    def forward(
        self,
        s: Float['b n d'],
        mask: Bool['b n'],
    ) -> dict[str, Float['...']]:
        """Run coordinate and lattice prediction heads."""
        return {
            "coords": self.coord_head(s, mask),
            "lattice": self.lattice_head(s, mask),
        }
