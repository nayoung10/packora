import torch.nn as nn

from src.models.components.activations import ActivationType, build_activation
from src.models.components.norms import NormType, build_norm
from src.utils.tensor_typing import Float, Bool


class MeanPoolLatticeHead(nn.Module):
    """Predict lattice by masked mean pooling over atom representations."""

    def __init__(
        self,
        dim: int,
        output_dim: int = 6,
        zero_init: bool = True,
        norm_type: NormType = "layernorm",
        activation: ActivationType = "silu",
    ) -> None:
        """Initialize masked mean lattice head."""
        super().__init__()
        self.norm = build_norm(dim, norm_type=norm_type)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            build_activation(activation),
            nn.Linear(dim, output_dim),
        )

        if zero_init:
            nn.init.constant_(self.mlp[-1].weight, 0)
            nn.init.constant_(self.mlp[-1].bias, 0)

    # TODO: Add attention-weighted pooling variant as an alternative to mean pooling

    def forward(
        self, s: Float['b n d'], atom_mask: Bool['b n']
    ) -> Float['b d']:
        """Pool over atoms and predict (B, D) lattice."""
        s = self.norm(s)

        # Masked mean pooling: zero out padded positions, divide by real atom count
        mask_expanded = atom_mask.unsqueeze(-1).float()
        s_masked = s * mask_expanded
        num_atoms = mask_expanded.sum(dim=1).clamp(min=1)
        s_pooled = s_masked.sum(dim=1) / num_atoms

        return self.mlp(s_pooled)
