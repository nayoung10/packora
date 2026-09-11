import torch
import torch.nn as nn

from src.models.components.norms import NormType, build_norm
from src.utils.tensor_typing import Bool, Float


class LinearCoordsHead(nn.Module):
    """Predict Cartesian coordinate velocities from per-atom representations."""

    def __init__(
        self,
        dim: int,
        center: bool = True,
        zero_init: bool = True,
        norm_type: NormType = "layernorm",
    ) -> None:
        """Initialize linear coordinate head."""
        super().__init__()
        self.norm = build_norm(dim, norm_type=norm_type)
        self.proj = nn.Linear(dim, 3)
        self.center = center

        if zero_init:
            nn.init.constant_(self.proj.weight, 0)
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, s: Float['b n d'], mask: Bool['b n']) -> Float['b n 3']:
        """Project (B, N, D) to (B, N, 3) coordinate predictions."""
        coords = self.proj(self.norm(s))

        # Remove center of mass using masked mean
        if self.center:
            mask_expanded = mask[:, :, None].float()
            coord_mean = (
                torch.sum(coords * mask_expanded, dim=1, keepdim=True)
                / torch.sum(mask_expanded, dim=1, keepdim=True)
            )
            coords = coords - coord_mean

        return coords
