# ruff: noqa: F722,F821

import math
import warnings
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn
from einops import rearrange

from src.models.components import initialize as init
from src.models.components.activations import ActivationType, build_activation
from src.models.components.attention import AttentionPairBias
from src.models.components.norms import NormType, build_norm
from src.utils.tensor_typing import Bool, Float

_cueq_triangle_warned = False
_deepspeed_triangle_attention_warned = False
_deepspeed_evo_attention_checked = False
_deepspeed_evo_attention_error: str | None = None
_deepspeed_evo_attention: Callable[..., Float["b i j h d"]] | None = None

TriangleAttentionBackend = Literal["deepspeed", "pytorch"]


def _warn_cueq_triangle(reason: str) -> None:
    """Warn once when cuEquivariance triangle acceleration cannot be used."""
    global _cueq_triangle_warned
    if _cueq_triangle_warned:
        return
    warnings.warn(
        f"cuEquivariance triangle_multiplicative_update unavailable: {reason}. "
        "Using PyTorch fallback.",
        stacklevel=3,
    )
    _cueq_triangle_warned = True


def _warn_deepspeed_triangle_attention(reason: str) -> None:
    """Warn once when DeepSpeed triangle attention cannot be used."""
    global _deepspeed_triangle_attention_warned
    if _deepspeed_triangle_attention_warned:
        return
    warnings.warn(
        f"DeepSpeed4Science EvoformerAttention unavailable: {reason}. "
        "Using PyTorch fallback.",
        stacklevel=3,
    )
    _deepspeed_triangle_attention_warned = True


def _load_deepspeed_evo_attention() -> Callable[..., Float["b i j h d"]] | None:
    """Return cached DeepSpeed4Science Evoformer attention when available."""
    global _deepspeed_evo_attention
    global _deepspeed_evo_attention_checked
    global _deepspeed_evo_attention_error
    if _deepspeed_evo_attention_checked:
        if _deepspeed_evo_attention_error is not None:
            _warn_deepspeed_triangle_attention(_deepspeed_evo_attention_error)
        return _deepspeed_evo_attention

    try:
        from deepspeed.ops.deepspeed4science import (  # noqa: PLC0415
            DS4Sci_EvoformerAttention,
        )
    except Exception as exc:  # pragma: no cover - depends on optional setup
        raise RuntimeError(
            "DeepSpeed4Science EvoformerAttention import failed while "
            "triangle_attention_backend='deepspeed': "
            f"{exc}. Configure DeepSpeed, CUDA_HOME, CUDA_PATH, and CUTLASS_PATH, "
            "or set model.net.pairmixer.triangle_attention_backend=pytorch to use "
            "the PyTorch implementation."
        ) from exc

    _deepspeed_evo_attention_checked = True
    _deepspeed_evo_attention_error = None
    _deepspeed_evo_attention = DS4Sci_EvoformerAttention
    return _deepspeed_evo_attention


@torch.compiler.disable
def _cueq_triangle_update(
    x: Float["b n n d"],
    direction: Literal["outgoing", "incoming"],
    mask: Bool["b n n"],
    norm_in: nn.Module,
    p_in: nn.Linear,
    g_in: nn.Linear,
    norm_out: nn.Module,
    p_out: nn.Linear,
    g_out: nn.Linear,
    eps: float,
) -> Optional[Float["b n n d"]]:
    """Run cuEquivariance triangle update when available and compatible."""
    if not isinstance(norm_in, nn.LayerNorm) or not isinstance(norm_out, nn.LayerNorm):
        _warn_cueq_triangle("LayerNorm is required")
        return None
    try:
        from cuequivariance_torch import triangle_multiplicative_update  # noqa: PLC0415
    except ImportError:
        _warn_cueq_triangle("package is not installed")
        return None
    return triangle_multiplicative_update(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in.weight,
        norm_in_bias=norm_in.bias,
        p_in_weight=p_in.weight,
        g_in_weight=g_in.weight,
        norm_out_weight=norm_out.weight,
        norm_out_bias=norm_out.bias,
        p_out_weight=p_out.weight,
        g_out_weight=g_out.weight,
        eps=eps,
    )


@torch.compiler.disable
def _deepspeed_triangle_attention(
    q: Float["b i h j d"],
    k: Float["b i h j d"],
    v: Float["b i h j d"],
    mask_bias: Float["b i 1 1 j"],
    triangle_bias: Float["b 1 h i j"],
) -> Optional[Float["b i j h d"]]:
    """Run DeepSpeed4Science Evoformer attention when available."""
    global _deepspeed_evo_attention
    global _deepspeed_evo_attention_checked
    global _deepspeed_evo_attention_error
    ds4s_evo_attention = _load_deepspeed_evo_attention()
    if ds4s_evo_attention is None:
        return None

    q_ds = rearrange(q, "b i h j d -> b i j h d")
    k_ds = rearrange(k, "b i h j d -> b i j h d")
    v_ds = rearrange(v, "b i h j d -> b i j h d")
    orig_dtype = q_ds.dtype
    kernel_dtype = (
        orig_dtype if orig_dtype in {torch.bfloat16, torch.float16} else torch.bfloat16
    )
    try:
        out = ds4s_evo_attention(
            q_ds.to(dtype=kernel_dtype),
            k_ds.to(dtype=kernel_dtype),
            v_ds.to(dtype=kernel_dtype),
            [
                mask_bias.to(dtype=kernel_dtype),
                triangle_bias.to(dtype=kernel_dtype),
            ],
        )
    except Exception as exc:  # pragma: no cover - depends on optional CUDA extension
        _deepspeed_evo_attention = None
        _deepspeed_evo_attention_checked = True
        _deepspeed_evo_attention_error = str(exc)
        _warn_deepspeed_triangle_attention(_deepspeed_evo_attention_error)
        return None
    return out.to(dtype=orig_dtype)


def _pairmixer_dropout_mask(
    dropout: float,
    z: Float["b n n d"],
    training: bool,
) -> Float["..."]:
    """Create a Pairmixer-style structured dropout mask."""
    return _pairmixer_oriented_dropout_mask(dropout, z, training, columnwise=False)


def _pairmixer_oriented_dropout_mask(
    dropout: float,
    z: Float["b n n d"],
    training: bool,
    columnwise: bool = False,
) -> Float["..."]:
    """Create rowwise or columnwise structured dropout masks."""
    base = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    if dropout <= 0.0 or not training:
        return torch.ones_like(base)
    keep_prob = 1.0 - dropout
    keep_mask = torch.rand(base.shape, device=z.device, dtype=torch.float32) < keep_prob
    return keep_mask.to(dtype=z.dtype) / keep_prob


class Transition(nn.Module):
    """Apply a normalized gated feed-forward transition."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        norm_type: NormType = "layernorm",
        norm_eps: float = 1e-5,
        activation: ActivationType = "silu",
    ) -> None:
        """Initialize transition projections."""
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.norm = build_norm(dim, norm_type=norm_type, eps=norm_eps)
        self.proj_a = nn.Linear(dim, hidden_dim, bias=False)
        self.proj_b = nn.Linear(dim, hidden_dim, bias=False)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=False)
        self.activation = build_activation(activation)
        init.lecun_normal_init_(self.proj_a.weight)
        init.lecun_normal_init_(self.proj_b.weight)
        init.final_init_(self.proj_out.weight)

    def forward(self, x: Float["... d"]) -> Float["... d"]:
        """Apply the transition."""
        x = self.norm(x)
        return self.proj_out(self.activation(self.proj_a(x)) * self.proj_b(x))


class TriangleMultiplication(nn.Module):
    """Apply one gated triangle multiplicative update."""

    def __init__(
        self,
        dim_pair: int,
        direction: Literal["outgoing", "incoming"],
        norm_type: NormType = "layernorm",
        norm_eps: float = 1e-5,
        use_cuequivariance: bool = False,
    ) -> None:
        """Initialize triangle multiplication projections."""
        super().__init__()
        self.direction = direction
        self.norm_eps = norm_eps
        self.use_cuequivariance = use_cuequivariance
        self.norm_in = build_norm(dim_pair, norm_type=norm_type, eps=norm_eps)
        self.p_in = nn.Linear(dim_pair, 2 * dim_pair, bias=False)
        self.g_in = nn.Linear(dim_pair, 2 * dim_pair, bias=False)
        self.norm_out = build_norm(dim_pair, norm_type=norm_type, eps=norm_eps)
        self.p_out = nn.Linear(dim_pair, dim_pair, bias=False)
        self.g_out = nn.Linear(dim_pair, dim_pair, bias=False)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize triangle update weights with Pairmixer-style defaults."""
        for norm in (self.norm_in, self.norm_out):
            if hasattr(norm, "weight") and norm.weight is not None:
                init.bias_init_one_(norm.weight)
            if hasattr(norm, "bias") and norm.bias is not None:
                init.bias_init_zero_(norm.bias)
        init.lecun_normal_init_(self.p_in.weight)
        init.gating_init_(self.g_in.weight)
        init.final_init_(self.p_out.weight)
        init.gating_init_(self.g_out.weight)

    def _torch_update(
        self,
        x: Float["b n n d"],
        mask: Bool["b n n"],
    ) -> Float["b n n d"]:
        """Run the PyTorch triangle update implementation."""
        x_norm = self.norm_in(x)
        x_proj = self.p_in(x_norm) * self.g_in(x_norm).sigmoid()
        x_proj = x_proj * mask.to(dtype=x_proj.dtype).unsqueeze(-1)
        a, b = torch.chunk(x_proj.float(), 2, dim=-1)
        if self.direction == "outgoing":
            update = torch.einsum("b i k d, b j k d -> b i j d", a, b)
        else:
            update = torch.einsum("b k i d, b k j d -> b i j d", a, b)
        update = update.to(dtype=x.dtype)
        return self.p_out(self.norm_out(update)) * self.g_out(x_norm).sigmoid()

    def forward(
        self,
        x: Float["b n n d"],
        mask: Bool["b n n"],
    ) -> Float["b n n d"]:
        """Apply triangle multiplication with optional cuEquivariance acceleration."""
        if self.use_cuequivariance:
            cueq_update = _cueq_triangle_update(
                x=x,
                direction=self.direction,
                mask=mask,
                norm_in=self.norm_in,
                p_in=self.p_in,
                g_in=self.g_in,
                norm_out=self.norm_out,
                p_out=self.p_out,
                g_out=self.g_out,
                eps=self.norm_eps,
            )
            if cueq_update is not None:
                return cueq_update
        return self._torch_update(x, mask)


class TriangleSelfAttention(nn.Module):
    """Apply one gated triangle self-attention update."""

    def __init__(
        self,
        dim_pair: int,
        num_heads: int = 4,
        head_dim: int | None = 32,
        starting: bool = True,
        backend: TriangleAttentionBackend = "deepspeed",
        chunk_size: int | None = None,
        norm_type: NormType = "layernorm",
        norm_eps: float = 1e-5,
        inf: float = 1e9,
    ) -> None:
        """Initialize triangle self-attention projections."""
        super().__init__()
        if num_heads <= 0:
            raise ValueError("triangle attention num_heads must be positive.")
        if head_dim is None:
            if dim_pair % num_heads != 0:
                raise ValueError(
                    "dim_pair must be divisible by triangle attention num_heads "
                    "when head_dim is not set."
                )
            head_dim = dim_pair // num_heads
        if head_dim <= 0:
            raise ValueError("triangle attention head_dim must be positive.")
        if backend not in {"deepspeed", "pytorch"}:
            raise ValueError(f"Unknown triangle attention backend: {backend}")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("triangle attention chunk_size must be positive or None.")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.starting = starting
        self.backend = backend
        self.chunk_size = chunk_size
        self.inf = inf
        hidden_dim = num_heads * head_dim
        self.norm = build_norm(dim_pair, norm_type=norm_type, eps=norm_eps)
        self.proj_q = nn.Linear(dim_pair, hidden_dim, bias=False)
        self.proj_k = nn.Linear(dim_pair, hidden_dim, bias=False)
        self.proj_v = nn.Linear(dim_pair, hidden_dim, bias=False)
        self.proj_bias = nn.Linear(dim_pair, num_heads, bias=False)
        self.proj_g = nn.Linear(dim_pair, hidden_dim, bias=False)
        self.proj_o = nn.Linear(hidden_dim, dim_pair, bias=False)
        self.used_deepspeed = False
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize attention weights with Genesis Pairmixer-style defaults."""
        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            init.bias_init_one_(self.norm.weight)
        if hasattr(self.norm, "bias") and self.norm.bias is not None:
            init.bias_init_zero_(self.norm.bias)
        init.glorot_uniform_init_(self.proj_q.weight)
        init.glorot_uniform_init_(self.proj_k.weight)
        init.glorot_uniform_init_(self.proj_v.weight)
        init.normal_init_(self.proj_bias.weight)
        init.gating_init_(self.proj_g.weight)
        init.final_init_(self.proj_o.weight)

    def _attention(
        self,
        x_norm: Float["b i j dp"],
        mask_bias: Float["b i 1 1 j"],
        triangle_bias: Float["b 1 h i j"],
    ) -> Float["b i j dp"]:
        """Run triangle attention on normalized pair features."""
        q = rearrange(
            self.proj_q(x_norm),
            "b i j (h d) -> b i h j d",
            h=self.num_heads,
        )
        k = rearrange(
            self.proj_k(x_norm),
            "b i j (h d) -> b i h j d",
            h=self.num_heads,
        )
        v = rearrange(
            self.proj_v(x_norm),
            "b i j (h d) -> b i h j d",
            h=self.num_heads,
        )
        if self.backend == "deepspeed":
            ds_out = _deepspeed_triangle_attention(q, k, v, mask_bias, triangle_bias)
            if ds_out is not None:
                self.used_deepspeed = True
                out = rearrange(ds_out, "b i j h d -> b i j (h d)")
                return self.proj_o(self.proj_g(x_norm).sigmoid() * out)

        logits = torch.einsum("b i h q d, b i h k d -> b i h q k", q.float(), k.float())
        logits = logits / math.sqrt(self.head_dim)
        logits = logits + mask_bias.float() + triangle_bias.float()
        weights = logits.softmax(dim=-1)
        out = torch.einsum("b i h q k, b i h k d -> b i h q d", weights, v.float())
        out = rearrange(out.to(dtype=x_norm.dtype), "b i h j d -> b i j (h d)")
        return self.proj_o(self.proj_g(x_norm).sigmoid() * out)

    def _forward_oriented(
        self,
        x: Float["b i j dp"],
        mask: Bool["b i j"],
    ) -> Float["b i j dp"]:
        """Apply attention in starting-node orientation."""
        x_norm = self.norm(x)
        mask_bias = (1.0 - mask.to(dtype=x.dtype)) * -self.inf
        mask_bias = rearrange(mask_bias, "b i j -> b i 1 1 j")
        triangle_bias = rearrange(self.proj_bias(x_norm), "b i j h -> b 1 h i j")
        if self.chunk_size is None:
            return self._attention(x_norm, mask_bias, triangle_bias)

        chunks = []
        num_rows = x_norm.shape[1]
        for start in range(0, num_rows, self.chunk_size):
            end = min(start + self.chunk_size, num_rows)
            chunks.append(
                self._attention(
                    x_norm[:, start:end],
                    mask_bias[:, start:end],
                    triangle_bias,
                )
            )
        return torch.cat(chunks, dim=1)

    def forward(
        self,
        x: Float["b n n dp"],
        mask: Bool["b n n"],
    ) -> Float["b n n dp"]:
        """Apply triangle self-attention."""
        self.used_deepspeed = False
        if self.starting:
            return self._forward_oriented(x, mask)
        x_t = rearrange(x, "b i j d -> b j i d")
        mask_t = rearrange(mask, "b i j -> b j i")
        out = self._forward_oriented(x_t, mask_t)
        return rearrange(out, "b j i d -> b i j d")


class PairmixerBlock(nn.Module):
    """Update pair features with optional single attention update."""

    def __init__(
        self,
        dim_single: int,
        dim_pair: int,
        num_heads: int,
        update_single: bool = False,
        dropout: float = 0.0,
        norm_type: NormType = "layernorm",
        norm_eps: float = 1e-5,
        activation: ActivationType = "silu",
        attention_impl: Literal["manual", "pytorch", "xformers"] = "pytorch",
        force_fp32_attention: bool = True,
        use_cuequivariance: bool = False,
        use_cuequivariance_triangle: bool | None = None,
        use_cuequivariance_attention: bool | None = None,
        use_triangle_attention: bool = False,
        triangle_attention_backend: TriangleAttentionBackend = "deepspeed",
        triangle_attention_num_heads: int = 4,
        triangle_attention_head_dim: int | None = 32,
        triangle_attention_chunk_size: int | None = None,
    ) -> None:
        """Initialize one Pairmixer block."""
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if use_cuequivariance_triangle is None:
            use_cuequivariance_triangle = use_cuequivariance
        if use_cuequivariance_attention is None:
            use_cuequivariance_attention = use_cuequivariance
        self.update_single = update_single
        self.use_triangle_attention = use_triangle_attention
        self.dropout = float(dropout)
        self.tri_out = TriangleMultiplication(
            dim_pair=dim_pair,
            direction="outgoing",
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_cuequivariance=use_cuequivariance_triangle,
        )
        self.tri_in = TriangleMultiplication(
            dim_pair=dim_pair,
            direction="incoming",
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_cuequivariance=use_cuequivariance_triangle,
        )
        if self.use_triangle_attention:
            self.tri_att_start = TriangleSelfAttention(
                dim_pair=dim_pair,
                num_heads=triangle_attention_num_heads,
                head_dim=triangle_attention_head_dim,
                starting=True,
                backend=triangle_attention_backend,
                chunk_size=triangle_attention_chunk_size,
                norm_type=norm_type,
                norm_eps=norm_eps,
            )
            self.tri_att_end = TriangleSelfAttention(
                dim_pair=dim_pair,
                num_heads=triangle_attention_num_heads,
                head_dim=triangle_attention_head_dim,
                starting=False,
                backend=triangle_attention_backend,
                chunk_size=triangle_attention_chunk_size,
                norm_type=norm_type,
                norm_eps=norm_eps,
            )
        self.transition_z = Transition(
            dim=dim_pair,
            norm_type=norm_type,
            norm_eps=norm_eps,
            activation=activation,
        )
        if self.update_single:
            self.attention = AttentionPairBias(
                c_s=dim_single,
                c_z=dim_pair,
                num_heads=num_heads,
                attention_impl=attention_impl,
                force_fp32_attention=force_fp32_attention,
                norm_type=norm_type,
                norm_eps=norm_eps,
                use_cuequivariance=use_cuequivariance_attention,
            )
            self.transition_s = Transition(
                dim=dim_single,
                norm_type=norm_type,
                norm_eps=norm_eps,
                activation=activation,
            )

    def forward(
        self,
        s: Float["b n ds"],
        z: Float["b n n dz"],
        atom_mask: Bool["b n"],
    ) -> tuple[Float["b n ds"], Float["b n n dz"]]:
        """Apply one Pairmixer block."""
        pair_mask = rearrange(atom_mask, "b i -> b i 1") & rearrange(
            atom_mask,
            "b j -> b 1 j",
        )
        dropout = _pairmixer_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_out(z, pair_mask)
        dropout = _pairmixer_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_in(z, pair_mask)
        if self.use_triangle_attention:
            dropout = _pairmixer_oriented_dropout_mask(self.dropout, z, self.training)
            z = z + dropout * self.tri_att_start(z, pair_mask)
            dropout = _pairmixer_oriented_dropout_mask(
                self.dropout,
                z,
                self.training,
                columnwise=True,
            )
            z = z + dropout * self.tri_att_end(z, pair_mask)
        z = z + self.transition_z(z)
        if self.update_single:
            s = s + self.attention(s=s, mask=atom_mask, z=z)
            s = s + self.transition_s(s)
        return s, z


class Pairmixer(nn.Module):
    """Stack Pairmixer blocks."""

    def __init__(
        self,
        dim_single: int,
        dim_pair: int,
        num_heads: int,
        num_blocks: int = 1,
        update_single: bool = False,
        dropout: float = 0.0,
        norm_type: NormType = "layernorm",
        norm_eps: float = 1e-5,
        activation: ActivationType = "silu",
        attention_impl: Literal["manual", "pytorch", "xformers"] = "pytorch",
        force_fp32_attention: bool = True,
        use_cuequivariance: bool = False,
        use_cuequivariance_triangle: bool | None = None,
        use_cuequivariance_attention: bool | None = None,
        use_triangle_attention: bool = False,
        triangle_attention_backend: TriangleAttentionBackend = "deepspeed",
        triangle_attention_num_heads: int = 4,
        triangle_attention_head_dim: int | None = 32,
        triangle_attention_chunk_size: int | None = None,
    ) -> None:
        """Initialize stacked Pairmixer blocks."""
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")
        if use_cuequivariance_triangle is None:
            use_cuequivariance_triangle = use_cuequivariance
        if use_cuequivariance_attention is None:
            use_cuequivariance_attention = use_cuequivariance
        self.layers = nn.ModuleList(
            [
                PairmixerBlock(
                    dim_single=dim_single,
                    dim_pair=dim_pair,
                    num_heads=num_heads,
                    update_single=update_single,
                    dropout=dropout,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                    activation=activation,
                    attention_impl=attention_impl,
                    force_fp32_attention=force_fp32_attention,
                    use_cuequivariance=use_cuequivariance,
                    use_cuequivariance_triangle=use_cuequivariance_triangle,
                    use_cuequivariance_attention=use_cuequivariance_attention,
                    use_triangle_attention=use_triangle_attention,
                    triangle_attention_backend=triangle_attention_backend,
                    triangle_attention_num_heads=triangle_attention_num_heads,
                    triangle_attention_head_dim=triangle_attention_head_dim,
                    triangle_attention_chunk_size=triangle_attention_chunk_size,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        s: Float["b n ds"],
        z: Float["b n n dz"],
        atom_mask: Bool["b n"],
    ) -> tuple[Float["b n ds"], Float["b n n dz"]]:
        """Run all Pairmixer blocks."""
        for layer in self.layers:
            s, z = layer(s, z, atom_mask)
        return s, z
