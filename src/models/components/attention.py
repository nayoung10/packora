# ruff: noqa: F722,F821

import warnings
from typing import Optional, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange
from einops.layers.torch import Rearrange

from src.models.components.norms import (
    NormType,
    build_norm,
    folded_norm_linear_params,
    normalize_without_affine,
)
from src.utils.tensor_typing import Float, Bool

# Check if xformers is available
try:
    from xformers.ops import memory_efficient_attention

    xformers_installed = True
except ImportError:
    xformers_installed = False

_cueq_attention_warned = False


def _warn_cueq_attention(reason: str) -> None:
    """Warn once when cuEquivariance attention acceleration cannot be used."""
    global _cueq_attention_warned
    if _cueq_attention_warned:
        return
    warnings.warn(
        f"cuEquivariance attention_pair_bias unavailable: {reason}. "
        "Using PyTorch fallback.",
        stacklevel=3,
    )
    _cueq_attention_warned = True


class AttentionPairBias(nn.Module):
    """Attention pair bias layer."""

    def __init__(
        self,
        c_s: int,
        c_z: Optional[int],
        num_heads: int,
        inf: float = 1e6,
        initial_norm: bool = True,
        attention_impl: Literal["manual", "pytorch", "xformers"] = "pytorch",
        force_fp32_attention: bool = True,
        norm_type: NormType = "layernorm",
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        use_xsa: bool = False,
        use_cuequivariance: bool = False,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence dimension.
        c_z : int, optional
            The input pairwise dimension. If None, the pairwise bias is not used.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6
        initial_norm: bool, optional
            Whether to apply layer norm to the input, by default True
        attention_impl: Literal["manual", "pytorch", "xformers"], optional
            The attention implementation to use, by default "pytorch".
            If "xformers" is chosen but not installed, it falls back to "pytorch".
        """
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.force_fp32_attention = force_fp32_attention
        self.norm_type = norm_type
        self.use_xsa = use_xsa
        self.qk_norm = qk_norm
        self.use_cuequivariance = use_cuequivariance

        self.initial_norm = initial_norm
        if self.initial_norm:
            self.norm_s = build_norm(c_s, norm_type=norm_type, eps=norm_eps)
        else:
            self.norm_s = nn.Identity()
        if self.qk_norm:
            self.q_norm = build_norm(self.head_dim, norm_type=norm_type, eps=norm_eps)
            self.k_norm = build_norm(self.head_dim, norm_type=norm_type, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)
        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        # init.final_init_(self.proj_o.weight)

        # Attention implementation
        if attention_impl == "manual":
            self.attention_fn = self._manual_attention
        elif attention_impl == "pytorch":
            self.attention_fn = self._pytorch_attention
        elif attention_impl == "xformers":
            if xformers_installed:
                self.attention_fn = self._xformers_attention
            else:
                warnings.warn("xformers is not installed. Using PyTorch.")
                self.attention_fn = self._pytorch_attention
        else:
            raise ValueError(f"Unknown attention implementation: {attention_impl}")

        # (Optional) pairwise features
        self.use_pair_bias = c_z is not None
        if self.use_pair_bias:
            self.proj_z = nn.Sequential(
                build_norm(c_z, norm_type=norm_type, eps=norm_eps),
                nn.Linear(c_z, num_heads, bias=False),
                Rearrange("b ... h -> b h ..."),
            )

    @torch.compiler.disable
    def _cueq_attention(
        self,
        s: Float["b n d"],
        q: Float["b n h dh"],
        k: Float["b n h dh"],
        v: Float["b n h dh"],
        mask: Bool["b n"],
        z: Optional[Float["b n n c_z"]] = None,
        projected_pair_bias: Optional[Float["b h n n"]] = None,
    ) -> Optional[Float["b n d"]]:
        """Run cuEquivariance attention pair bias when available and compatible."""
        if not self.use_pair_bias:
            return None
        if self.norm_type != "layernorm" and projected_pair_bias is None:
            _warn_cueq_attention("LayerNorm pair-bias normalization is required")
            return None
        if z is None and projected_pair_bias is None:
            return None
        try:
            from cuequivariance_torch import attention_pair_bias  # noqa: PLC0415
        except ImportError:
            _warn_cueq_attention("package is not installed")
            return None

        q_heads = rearrange(q, "b n h d -> b h n d")
        k_heads = rearrange(k, "b n h d -> b h n d")
        v_heads = rearrange(v, "b n h d -> b h n d")
        norm_z = self.proj_z[0]
        linear_z = self.proj_z[1]
        if projected_pair_bias is None:
            z_input = z
            is_cached_z_proj = False
            w_proj_z = linear_z.weight
            b_proj_z = linear_z.bias
            w_ln_z = norm_z.weight
            b_ln_z = norm_z.bias
        else:
            z_input = projected_pair_bias
            is_cached_z_proj = True
            w_proj_z = None
            b_proj_z = None
            w_ln_z = None
            b_ln_z = None

        result = attention_pair_bias(
            s=s,
            q=q_heads,
            k=k_heads,
            v=v_heads,
            z=z_input,
            mask=mask,
            num_heads=self.num_heads,
            w_proj_z=w_proj_z,
            w_proj_g=self.proj_g.weight,
            w_proj_o=self.proj_o.weight,
            w_ln_z=w_ln_z,
            b_ln_z=b_ln_z,
            b_proj_z=b_proj_z,
            b_proj_g=self.proj_g.bias,
            b_proj_o=self.proj_o.bias,
            inf=self.inf,
            eps=norm_z.eps,
            return_z_proj=False,
            is_cached_z_proj=is_cached_z_proj,
        )
        if isinstance(result, tuple):
            return result[0]
        return result

    def _manual_attention(
        self,
        q: Float["b n h d"],
        k: Float["b n h d"],
        v: Float["b n h d"],
        attn_mask: Float["b h n n"],
    ):
        attn_scores = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
        attn_scores = attn_scores / (self.head_dim**0.5)
        attn_scores = attn_scores + attn_mask

        attn_weights = attn_scores.softmax(dim=-1)

        o = torch.einsum("bhij,bjhd->bihd", attn_weights, v.float()).to(v.dtype)
        return o

    def normalize_pair_bias(
        self,
        pair_bias: Float["b n n c_z"],
    ) -> Float["b n n c_z"]:
        """Apply pair-bias LayerNorm without affine parameters."""
        norm = self.proj_z[0]
        return normalize_without_affine(norm, pair_bias, self.norm_type)

    def _project_pair_bias(
        self,
        pair_bias: Float["b n n c_z"],
        normalized_pair_bias: Optional[Float["b n n c_z"]] = None,
    ) -> Float["b h n n"]:
        """Project pairwise features into per-head attention bias."""
        if normalized_pair_bias is not None:
            norm = self.proj_z[0]
            linear = self.proj_z[1]
            weight, bias = folded_norm_linear_params(norm, linear, self.norm_type)
            z_bias = F.linear(normalized_pair_bias, weight, bias)
            return rearrange(z_bias, "b ... h -> b h ...")
        return self.proj_z(pair_bias)

    def _pytorch_attention(
        self,
        q: Float["b n h d"],
        k: Float["b n h d"],
        v: Float["b n h d"],
        attn_mask: Float["b h n n"],
    ):
        # Reshape for scaled_dot_product_attention: (b, n, h, d) -> (b, h, n, d)
        q, k, v = map(lambda t: rearrange(t, "b n h d -> b h n d"), (q, k, v))

        if self.force_fp32_attention:
            o = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), attn_mask=attn_mask.float()
            ).to(v.dtype)
        else:
            o = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask.to(dtype=q.dtype),
            )

        # Transpose back to (b, n, h, d_h)
        return o.transpose(1, 2)

    def _xformers_attention(
        self,
        q: Float["b n h d_h"],
        k: Float["b n h d_h"],
        v: Float["b n h d_h"],
        attn_mask: Float["b h n n"],
    ) -> Tensor:
        # Expects input shape of (b, n, h, d_h) which is already satisfied
        o = memory_efficient_attention(q, k, v, attn_bias=attn_mask)
        return o

    def _create_attn_mask(
        self,
        mask: Bool["b n"],
        pair_bias: Optional[Float["b n n c_z"]] = None,
        normalized_pair_bias: Optional[Float["b n n c_z"]] = None,
        projected_pair_bias: Optional[Float["b h n n"]] = None,
    ) -> Float["b h n n"]:

        B, N = mask.shape

        # Attention mask broadcasted to (b, 1, 1, n)
        attn_mask = (1 - mask[:, None, None, :].float()) * -self.inf
        attn_mask = attn_mask.expand(B, self.num_heads, N, N)

        # (Optional) Add pair bias
        if self.use_pair_bias:
            if projected_pair_bias is None:
                assert pair_bias is not None, (
                    "pair_bias must be provided if use_pair_bias is True"
                )
                z_bias = self._project_pair_bias(pair_bias, normalized_pair_bias)
            else:
                z_bias = projected_pair_bias
            if not self.force_fp32_attention and attn_mask.dtype != z_bias.dtype:
                attn_mask = attn_mask.to(dtype=z_bias.dtype)
            attn_mask = attn_mask + z_bias

        return attn_mask.contiguous()  # contiguous for xformers

    def forward(
        self,
        s: Tensor,
        mask: Tensor,
        z: Optional[Tensor] = None,
        normalized_pair_bias: Optional[Tensor] = None,
        projected_pair_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            s: Input sequence tensor (B, N, D).
            mask: Sequence mask tensor (B, N).
            z: Optional pairwise bias tensor (B, N, N, D).

        Returns:
            The output sequence tensor.
        """
        B, N, _ = s.shape

        # Initial normalization and projections (common logic)
        s = self.norm_s(s)
        q = rearrange(self.proj_q(s), "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(self.proj_k(s), "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(self.proj_v(s), "b n (h d) -> b n h d", h=self.num_heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        g = self.proj_g(s).sigmoid()

        # Attention mask
        if self.use_cuequivariance:
            cueq_output = self._cueq_attention(
                s=s,
                q=q,
                k=k,
                v=v,
                mask=mask,
                z=z,
                projected_pair_bias=projected_pair_bias,
            )
            if cueq_output is not None:
                return cueq_output

        attn_mask = self._create_attn_mask(
            mask,
            z,
            normalized_pair_bias,
            projected_pair_bias,
        )

        # Attention computation
        o = self.attention_fn(q, k, v, attn_mask)
        if self.use_xsa:
            v_normalized = F.normalize(v, dim=-1)
            o = o - (o * v_normalized).sum(dim=-1, keepdim=True) * v_normalized

        # Final gating and projection
        o = rearrange(o, "b n h d -> b n (h d)")
        o = self.proj_o(g * o)

        return o
