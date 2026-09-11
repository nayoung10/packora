from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.models.components.activations import ActivationType, build_activation
from src.models.components.attention import AttentionPairBias
from src.models.components.norms import NormType, build_norm, folded_norm_linear_params
from src.utils.tensor_typing import Bool, Float

#################################################################################
#                               Transformer blocks                              #
#################################################################################


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        activation: ActivationType = "silu",
        norm_layer: type[nn.Module] | None = None,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = build_activation(activation)
        self.has_drop = drop > 0.0
        self.has_norm = norm_layer is not None
        self.drop1 = nn.Dropout(drop) if self.has_drop else nn.Identity()
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop) if self.has_drop else nn.Identity()

    def forward(self, x: Float["... d"]) -> Float["... d"]:
        """Apply the feed-forward projection."""
        x = self.fc1(x)
        x = self.act(x)
        if self.has_drop:
            x = self.drop1(x)
        if self.has_norm:
            x = self.norm(x)
        x = self.fc2(x)
        if self.has_drop:
            x = self.drop2(x)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

AttentionImpl = Literal["manual", "pytorch", "xformers"]
MlpType = Literal["standard", "swiglu"]


def _round_swiglu_hidden(dim: int, multiple_of: int) -> int:
    """Round SwiGLU hidden width using the Slowrun 8d/3 convention."""
    if multiple_of <= 0:
        raise ValueError("swiglu_multiple_of must be positive.")
    hidden = 8 * dim // 3
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network with rounded 8d/3 hidden width."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        activation: ActivationType = "silu",
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features

        self.gate = nn.Linear(in_features, hidden_features, bias=bias)
        self.up = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = build_activation(activation)
        self.has_drop = drop > 0.0
        self.drop1 = nn.Dropout(drop) if self.has_drop else nn.Identity()
        self.down = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop) if self.has_drop else nn.Identity()

    def forward(self, x: Float["... d"]) -> Float["... d"]:
        """Apply the gated feed-forward projection."""
        x = self.act(self.gate(x)) * self.up(x)
        if self.has_drop:
            x = self.drop1(x)
        x = self.down(x)
        if self.has_drop:
            x = self.drop2(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self,
        heads,
        dim,
        dim_pair=None,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_impl: AttentionImpl = "pytorch",
        force_fp32_attention: bool = True,
        norm_type: NormType = "layernorm",
        block_norm_eps: float = 1e-6,
        attention_norm_eps: float = 1e-5,
        qk_norm: bool = False,
        use_xsa: bool = False,
        use_cuequivariance: bool = False,
        activation: ActivationType = "silu",
        mlp_type: MlpType = "standard",
        swiglu_multiple_of: int = 256,
    ) -> None:
        super().__init__()
        self.norm1 = build_norm(
            dim,
            norm_type=norm_type,
            elementwise_affine=False,
            eps=block_norm_eps,
        )
        self.attn = AttentionPairBias(
            c_s=dim,
            c_z=dim_pair,
            num_heads=heads,
            attention_impl=attention_impl,
            force_fp32_attention=force_fp32_attention,
            norm_type=norm_type,
            norm_eps=attention_norm_eps,
            qk_norm=qk_norm,
            use_xsa=use_xsa,
            use_cuequivariance=use_cuequivariance,
        )
        self.norm2 = build_norm(
            dim,
            norm_type=norm_type,
            elementwise_affine=False,
            eps=block_norm_eps,
        )
        if mlp_type == "standard":
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                activation=activation,
                drop=dropout,
            )
        elif mlp_type == "swiglu":
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=_round_swiglu_hidden(dim, swiglu_multiple_of),
                activation=activation,
                drop=dropout,
            )
        else:
            raise ValueError(f"Unknown mlp_type: {mlp_type}")
        self.adaLN_modulation = nn.Sequential(
            build_activation(activation), nn.Linear(dim, 6 * dim, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT encoder blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x,
        c,
        mask,
        z=None,
        normalized_pair_bias=None,
        projected_pair_bias=None,
    ):
        """Apply one DiT block."""
        # Generate modulation parameters (shift, scale gate) from condition c
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # Attention block
        _x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            s=_x,
            mask=mask,
            z=z,
            normalized_pair_bias=normalized_pair_bias,
            projected_pair_bias=projected_pair_bias,
        )

        # MLP block
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class DiT(nn.Module):
    """Transformer with DiT blocks"""

    def __init__(
        self,
        depth,
        heads,
        dim,
        dim_pair=None,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_impl: AttentionImpl = "pytorch",
        force_fp32_attention: bool = True,
        share_pair_bias_norm: bool = False,
        activation_checkpointing=False,
        norm_type: NormType = "layernorm",
        block_norm_eps: float = 1e-6,
        attention_norm_eps: float = 1e-5,
        qk_norm: bool = False,
        use_xsa: bool = False,
        use_cuequivariance: bool = False,
        activation: ActivationType = "silu",
        mlp_type: MlpType = "standard",
        swiglu_multiple_of: int = 256,
    ) -> None:
        super().__init__()

        self.activation_checkpointing = activation_checkpointing
        self.share_pair_bias_norm = share_pair_bias_norm
        self.layers = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_pair=dim_pair,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_impl=attention_impl,
                    force_fp32_attention=force_fp32_attention,
                    norm_type=norm_type,
                    block_norm_eps=block_norm_eps,
                    attention_norm_eps=attention_norm_eps,
                    qk_norm=qk_norm,
                    use_xsa=use_xsa,
                    use_cuequivariance=use_cuequivariance,
                    activation=activation,
                    mlp_type=mlp_type,
                    swiglu_multiple_of=swiglu_multiple_of,
                )
                for _ in range(depth)
            ]
        )

    def _project_pair_biases(
        self,
        normalized_pair_bias: Float["b n n c_z"],
    ) -> Float["l b h n n"]:
        """Project shared normalized pair features for all transformer layers."""
        linear_weights = []
        linear_biases = []
        for layer in self.layers:
            norm = layer.attn.proj_z[0]
            linear = layer.attn.proj_z[1]
            weight, bias = folded_norm_linear_params(
                norm,
                linear,
                layer.attn.norm_type,
            )
            linear_weights.append(weight)
            if bias is not None:
                linear_biases.append(bias)
        linear_weight = torch.stack(linear_weights)
        weight = rearrange(linear_weight, "l h c -> (l h) c")
        bias = None
        if linear_biases:
            bias = rearrange(torch.stack(linear_biases), "l h -> (l h)")
        projected = F.linear(normalized_pair_bias, weight, bias)
        return rearrange(projected, "b i j (l h) -> l b h i j", l=len(self.layers))

    def precompute_pair_biases(self, z: Float["b n n dp"]) -> Float["l b h n n"] | None:
        """Normalize pair features once and project each block's attention bias."""
        if not self.share_pair_bias_norm or len(self.layers) == 0:
            return None
        normalized_pair_bias = self.layers[0].attn.normalize_pair_bias(z)
        return self._project_pair_biases(normalized_pair_bias)

    def forward(
        self,
        x: Float["b n d"],
        c: Float["b d"],
        mask: Bool["b n"],
        z: Float["b n n dp"] | None = None,
        projected_pair_biases: Float["l b h n n"] | None = None,
    ) -> Float["b n d"]:
        """Run transformer blocks with cached or freshly projected pair biases."""
        # Training and uncached inference compute biases for this forward call
        if projected_pair_biases is None and z is not None:
            projected_pair_biases = self.precompute_pair_biases(z)

        for layer_idx, layer in enumerate(self.layers):
            projected_pair_bias = (
                None
                if projected_pair_biases is None
                else projected_pair_biases[layer_idx]
            )
            if self.activation_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    c,
                    mask,
                    z,
                    None,
                    projected_pair_bias,
                    use_reentrant=False,
                )
            else:
                x = layer(x, c, mask, z, None, projected_pair_bias)
        return x
