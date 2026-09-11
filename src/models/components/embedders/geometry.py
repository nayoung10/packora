# ruff: noqa: F722,F821

import math
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.components.activations import ActivationType, build_activation
from src.models.components.lattice_repr import cell_to_y1_features
from src.utils.tensor_typing import Bool, Float


def _softplus_inverse(x: Float["..."]) -> Float["..."]:
    """Invert softplus for positive initial parameter values."""
    x = x.clamp_min(1e-8)
    return x + torch.log(-torch.expm1(-x))


class GeometryPairEmbedder(nn.Module):
    """Embed GEM-style minimum-image geometry into pair representation space."""

    def __init__(
        self,
        dim_pair: int,
        n_fourier_freqs: int = 8,
        n_rbf: int = 16,
        rbf_max: float = 2.0,
        hidden_dim: int = 128,
        pbc_radius: int = 1,
        activation: ActivationType = "silu",
        use_time_gate: bool = True,
    ) -> None:
        """Initialize full GEM-style geometry feature projection."""
        super().__init__()
        if n_fourier_freqs <= 0:
            raise ValueError("n_fourier_freqs must be positive.")
        if n_rbf <= 0:
            raise ValueError("n_rbf must be positive.")
        if rbf_max <= 0.0:
            raise ValueError("rbf_max must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if pbc_radius < 0:
            raise ValueError("pbc_radius must be non-negative.")

        self.dim_pair = dim_pair
        self.use_time_gate = use_time_gate
        offsets = torch.tensor(
            list(product(range(-pbc_radius, pbc_radius + 1), repeat=3)),
            dtype=torch.float32,
        )
        self.register_buffer(
            "pbc_offsets",
            rearrange(offsets, "k c -> 1 1 1 k c"),
            persistent=False,
        )
        self.register_buffer(
            "fourier_freqs",
            torch.arange(1, n_fourier_freqs + 1, dtype=torch.float32),
            persistent=False,
        )
        centers = torch.linspace(0.0, rbf_max, n_rbf, dtype=torch.float32)
        self.register_buffer("rbf_centers", centers, persistent=False)
        if n_rbf == 1:
            rbf_gamma = 1.0
        else:
            delta = rbf_max / float(n_rbf - 1)
            rbf_gamma = 1.0 / max(delta * delta, 1e-6)
        self.register_buffer(
            "rbf_gamma",
            torch.tensor(rbf_gamma, dtype=torch.float32),
            persistent=False,
        )

        feature_dim = 6 * n_fourier_freqs + n_rbf + 6
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            build_activation(activation),
            nn.Linear(hidden_dim, dim_pair, bias=False),
        )
        distance_scale_init = torch.ones((1, 1, 1, dim_pair), dtype=torch.float32)
        self.distance_scale_raw = nn.Parameter(_softplus_inverse(distance_scale_init))
        if self.use_time_gate:
            alpha_init = torch.ones((1, 1, 1, dim_pair), dtype=torch.float32)
            self.gate_alpha_raw = nn.Parameter(_softplus_inverse(alpha_init))
            self.gate_beta = nn.Parameter(torch.zeros_like(alpha_init))

    def _cell_matrix(self, cell: Float["b ..."]) -> Float["b 3 3"]:
        """Normalize flattened or matrix cell tensors to matrix form."""
        if cell.ndim == 3 and cell.shape[-2:] == (3, 3):
            return cell
        if cell.ndim == 2 and cell.shape[-1] == 9:
            return rearrange(cell, "b (i j) -> b i j", i=3, j=3)
        raise ValueError(f"Expected cell shape (B, 3, 3) or (B, 9), got {cell.shape}.")

    def _cart_to_frac(
        self,
        coords: Float["b n 3"],
        cell: Float["b 3 3"],
    ) -> Float["b n 3"]:
        """Convert Cartesian coordinates to fractional coordinates."""
        cell_inv = torch.linalg.inv(cell)
        return torch.einsum("b n c, b c f -> b n f", coords, cell_inv)

    def _minimum_image_features(
        self,
        coords: Float["b n 3"],
        cell: Float["b 3 3"],
    ) -> tuple[Float["b n n 3"], Float["b n n"], Float["b 6"]]:
        """Compute minimum-image fractional deltas, distances, and lattice features."""
        calc_dtype = torch.float32
        coords = coords.to(dtype=calc_dtype)
        cell = cell.to(dtype=calc_dtype)
        frac = self._cart_to_frac(coords, cell)
        delta = rearrange(frac, "b i c -> b i 1 c") - rearrange(
            frac,
            "b j c -> b 1 j c",
        )
        offsets = self.pbc_offsets.to(device=coords.device, dtype=coords.dtype)
        delta_images = rearrange(delta, "b i j c -> b i j 1 c") + offsets

        gram = cell @ rearrange(cell, "b i j -> b j i")
        dist2_images = torch.einsum(
            "b i j k c, b c d, b i j k d -> b i j k",
            delta_images,
            gram,
            delta_images,
        )
        min_idx = dist2_images.argmin(dim=-1, keepdim=True)
        min_delta = torch.gather(
            delta_images,
            dim=3,
            index=repeat(min_idx, "b i j 1 -> b i j 1 c", c=3),
        )
        min_delta = rearrange(min_delta, "b i j 1 c -> b i j c")
        min_dist2 = torch.gather(dist2_images, dim=-1, index=min_idx)
        min_dist = rearrange(min_dist2.clamp_min(1e-12).sqrt(), "b i j 1 -> b i j")

        lengths = torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
        cell_scale = lengths.mean(dim=-1).clamp_min(1e-6)
        min_dist_norm = min_dist / rearrange(cell_scale, "b -> b 1 1")
        lattice_y1 = cell_to_y1_features(cell)
        return min_delta, min_dist_norm, lattice_y1

    def _pair_mask(self, atom_mask: Bool["b n"]) -> Float["b n n 1"]:
        """Build all-real non-self pair mask."""
        real_pair = rearrange(atom_mask, "b i -> b i 1") & rearrange(
            atom_mask,
            "b j -> b 1 j",
        )
        n = atom_mask.shape[1]
        diagonal = torch.eye(n, dtype=torch.bool, device=atom_mask.device)
        real_pair = real_pair & ~rearrange(diagonal, "i j -> 1 i j")
        return rearrange(real_pair.float(), "b i j -> b i j 1")

    def _time_gate(
        self,
        times: Float["b"],
        dtype: torch.dtype,
    ) -> Float["b 1 1 d"]:
        """Compute Crystalite-style learned sigmoid gate over logit(t)."""
        if not self.use_time_gate:
            return torch.ones(
                (times.shape[0], 1, 1, self.dim_pair),
                device=times.device,
                dtype=dtype,
            )
        t = times.to(dtype=torch.float32).clamp(min=1e-5, max=1.0 - 1e-5)
        logit_t = torch.logit(t)
        alpha = F.softplus(self.gate_alpha_raw).to(device=times.device)
        beta = self.gate_beta.to(device=times.device)
        gate = torch.sigmoid(rearrange(logit_t, "b -> b 1 1 1") * alpha + beta)
        return gate.to(dtype=dtype)

    def forward(self, noisy_batch: dict) -> Float["b n n dp"]:
        """Embed minimum-image geometry from physical noisy coordinates and cell."""
        if "x_t_physical" not in noisy_batch or "cell_t_physical" not in noisy_batch:
            raise RuntimeError(
                "GeometryPairEmbedder requires x_t_physical and cell_t_physical."
            )
        coords = noisy_batch["x_t_physical"]
        cell = self._cell_matrix(noisy_batch["cell_t_physical"])
        min_delta, min_dist_norm, lattice_y1 = self._minimum_image_features(
            coords,
            cell,
        )

        freqs = self.fourier_freqs.to(device=coords.device, dtype=min_delta.dtype)
        args = 2.0 * math.pi * rearrange(min_delta, "b i j c -> b i j c 1") * freqs
        fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        fourier = rearrange(fourier, "b i j c f -> b i j (c f)")

        centers = self.rbf_centers.to(device=coords.device, dtype=min_dist_norm.dtype)
        gamma = self.rbf_gamma.to(device=coords.device, dtype=min_dist_norm.dtype)
        rbf = torch.exp(
            -gamma
            * (
                rearrange(min_dist_norm, "b i j -> b i j 1")
                - rearrange(centers, "r -> 1 1 1 r")
            ).square()
        )
        lattice = repeat(
            lattice_y1.to(dtype=min_dist_norm.dtype),
            "b d -> b i j d",
            i=coords.shape[1],
            j=coords.shape[1],
        )
        geom_features = torch.cat([fourier, rbf, lattice], dim=-1)

        edge_bias = self.proj(geom_features.to(dtype=coords.dtype))
        distance_bias = -F.softplus(self.distance_scale_raw).to(
            device=coords.device,
            dtype=edge_bias.dtype,
        ) * rearrange(min_dist_norm, "b i j -> b i j 1").to(dtype=edge_bias.dtype)
        geom = edge_bias + distance_bias
        times = noisy_batch.get("flow_times", noisy_batch["times"])
        geom = geom * self._time_gate(times, geom.dtype)
        return geom * self._pair_mask(noisy_batch["atom_mask"]).to(dtype=geom.dtype)
