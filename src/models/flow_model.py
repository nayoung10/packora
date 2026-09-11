# ruff: noqa: F722,F821

import logging
import math
from functools import partial
from typing import Any, Literal

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.utils._pytree import tree_map

from src.data.components.prior import get_sampler
from src.models.components.interpolants import Interpolant
from src.models.components.interpolants.linear import LinearInterpolant
from src.models.components.lattice_repr import lattice_params_to_cell
from src.models.components.sampling.edm_heun import (
    churn_gamma,
    edm_derivative,
    edm_heun_config_from_dict,
    karras_sigma_schedule,
    sigma_to_flow_time,
)
from src.models.components.scalers import Scaler
from src.models.components.time_samplers import TimeSampler
from src.models.energy import MLIPEnergy
from src.utils.tensor_typing import Bool, Float, Int

logger = logging.getLogger(__name__)

NoisyInputEntry = Literal["before_pairmixer", "after_pairmixer"]


class MaterialFlowModel(nn.Module):
    """CSP model: predicts coordinates and lattice conditioned on atom types."""

    def __init__(
        self,
        input_embedder: nn.Module,
        timestep_embedder: nn.Module,
        transformer: nn.Module,
        heads: nn.Module,
        pairmixer: nn.Module | None = None,
        use_cuequivariance: bool = False,
        noisy_input_entry: NoisyInputEntry = "before_pairmixer",
        edm_preconditioning: dict[str, Any] | None = None,
        self_conditioning_embedder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if noisy_input_entry not in {"before_pairmixer", "after_pairmixer"}:
            raise ValueError(f"Unknown noisy_input_entry: {noisy_input_entry}")
        self.self_conditioning_embedder = self_conditioning_embedder
        self.input_embedder = input_embedder
        self.timestep_embedder = timestep_embedder
        self.transformer = transformer
        self.heads = heads
        self.pairmixer = pairmixer
        self.use_cuequivariance = use_cuequivariance
        self.noisy_input_entry = noisy_input_entry
        self.edm_preconditioning = edm_preconditioning or {}
        self.edm_preconditioning_enabled = bool(
            self.edm_preconditioning.get("enabled", False)
        )
        self.edm_loss_weight_enabled = bool(
            self.edm_preconditioning.get("loss_weight_enabled", True)
        )
        self.edm_preconditioning_eps = float(self.edm_preconditioning.get("eps", 1e-5))
        if not 0.0 < self.edm_preconditioning_eps < 0.5:
            raise ValueError("edm_preconditioning.eps must be in (0, 0.5).")
        self._validate_pairwise_contract()

    def _expand_coeff(self, coeff: Float["b"], target: Tensor) -> Tensor:
        """Broadcast a per-sample EDM coefficient to a target tensor."""
        expanded = coeff
        for _ in range(target.ndim - 1):
            expanded = expanded.unsqueeze(-1)
        if expanded.ndim != target.ndim or expanded.shape[0] != target.shape[0]:
            raise RuntimeError("EDM coefficient broadcast shape is invalid.")
        return expanded

    def _edm_coefficients(self, t: Float["b"]) -> dict[str, Float["b"]]:
        """Compute EDM preconditioning coefficients for linear interpolation."""
        eps = self.edm_preconditioning_eps
        t = t.clamp(min=eps, max=1.0 - eps)
        one_minus_t = 1.0 - t
        denom = one_minus_t.square() + t.square()
        sqrt_denom = denom.sqrt()
        coeffs = {
            "c_in": denom.rsqrt(),
            "c_skip": t / denom,
            "c_out": one_minus_t / sqrt_denom,
            "c_noise": 0.25 * torch.log(one_minus_t / t),
        }
        for name, value in coeffs.items():
            if value.shape != t.shape:
                raise RuntimeError(f"EDM coefficient {name} has invalid shape.")
            torch._assert(
                torch.isfinite(value).all(),
                f"EDM coefficient {name} is not finite.",
            )
        return coeffs

    def edm_loss_weight(self, t: Float["b"], loss_type: str) -> Float["b"]:
        """Compute EDM loss weights matching the endpoint preconditioning."""
        c_out = self._edm_coefficients(t)["c_out"]
        if loss_type == "l1":
            return c_out.abs().reciprocal()
        if loss_type == "l2":
            return c_out.square().reciprocal()
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    def _precondition_noisy_batch(
        self,
        noisy_batch: dict,
    ) -> tuple[dict, dict[str, Float["b"]]]:
        """Scale noisy inputs and timestep for EDM residual prediction."""
        coeffs = self._edm_coefficients(noisy_batch["times"])
        preconditioned_batch = dict(noisy_batch)
        c_in_x = self._expand_coeff(coeffs["c_in"], noisy_batch["x_t"])
        c_in_l = self._expand_coeff(coeffs["c_in"], noisy_batch["l_t"])
        preconditioned_batch["x_t"] = c_in_x * noisy_batch["x_t"]
        preconditioned_batch["l_t"] = c_in_l * noisy_batch["l_t"]
        preconditioned_batch["times"] = coeffs["c_noise"]
        return preconditioned_batch, coeffs

    def _apply_edm_output(
        self,
        noisy_batch: dict,
        preds: dict[str, Tensor],
        coeffs: dict[str, Float["b"]],
    ) -> dict[str, Tensor]:
        """Convert residual network outputs to clean endpoint predictions."""
        c_skip_x = self._expand_coeff(coeffs["c_skip"], noisy_batch["x_t"])
        c_out_x = self._expand_coeff(coeffs["c_out"], preds["coords"])
        c_skip_l = self._expand_coeff(coeffs["c_skip"], noisy_batch["l_t"])
        c_out_l = self._expand_coeff(coeffs["c_out"], preds["lattice"])
        return {
            "coords": c_skip_x * noisy_batch["x_t"] + c_out_x * preds["coords"],
            "lattice": c_skip_l * noisy_batch["l_t"] + c_out_l * preds["lattice"],
        }

    def _validate_pairwise_contract(self) -> None:
        """Ensure pairwise conditioning and transformer pair bias stay in sync."""
        pairwise_embedder = getattr(self.input_embedder, "pairwise_embedder", None)
        pairwise_condition = getattr(
            pairwise_embedder,
            "pairwise_condition_embedder",
            None,
        )
        pairwise_present = pairwise_condition is not None

        uses_pair_bias = False
        layers = getattr(self.transformer, "layers", None)
        if layers is not None and len(layers) > 0:
            attn = getattr(layers[0], "attn", None)
            uses_pair_bias = bool(getattr(attn, "use_pair_bias", False))

        if pairwise_present and not uses_pair_bias:
            raise ValueError(
                "Pairwise conditioning is present, but transformer pair bias is disabled. "
                "Set transformer.dim_pair to the pairwise embedder dimension."
            )
        if uses_pair_bias and not pairwise_present:
            raise ValueError(
                "Transformer pair bias is enabled, but pairwise conditioning is missing."
            )
        if self.pairmixer is not None and not pairwise_present:
            raise ValueError("Pairmixer requires pairwise conditioning features.")

    def build_inference_cache(
        self, noisy_batch: dict[str, Any]
    ) -> dict[str, Float["..."]]:
        """Cache condition-only representations and static attention biases for sampling."""
        if self.noisy_input_entry == "before_pairmixer":
            _, c, _, z = self.input_embedder.base_parts(noisy_batch)
            cache = {"c": c}
            if z is not None:
                cache["z"] = z
            return cache

        _, _, s, z = self.input_embedder.base_parts(noisy_batch)
        if self.pairmixer is not None:
            if z is None:
                raise RuntimeError("Pairmixer requires pairwise representation z.")
            s, z = self.pairmixer(s, z, noisy_batch["atom_mask"])
        cache = {"s": s}
        if z is not None:
            cache["z"] = z
            # After Pairmixer, pair biases are constant unless noisy geometry is added
            if self.input_embedder.pairwise_embedder.geometry_pair_embedder is None:
                projected_pair_biases = self.transformer.precompute_pair_biases(z)
                if projected_pair_biases is not None:
                    cache["projected_pair_biases"] = projected_pair_biases
        return cache

    def _cached_input_representations(
        self,
        model_batch: dict,
        inference_cache: dict[str, Tensor],
    ) -> tuple[Float["b n d"], Float["b n n dp"] | None]:
        """Return model input representations from a sampling cache."""
        if self.noisy_input_entry == "before_pairmixer":
            c = inference_cache["c"]
            s = self.input_embedder.single_embedder.base_embedding(
                model_batch,
                c,
            )
            s = s + self.input_embedder.single_embedder.noisy_embedding(model_batch)
            z = inference_cache.get("z")
            z = self.input_embedder.add_geometry(model_batch, z)
            return s, z

        s = inference_cache["s"] + self.input_embedder.noisy_single_embedding(
            model_batch
        )
        z = inference_cache.get("z")
        z = self.input_embedder.add_geometry(model_batch, z)
        return s, z

    def forward(self, noisy_batch: dict[str, Any]) -> dict[str, Float["..."]]:
        """Run full forward pass: embed, transform, predict coords and lattice."""
        edm_coeffs = None
        model_batch = noisy_batch
        if self.edm_preconditioning_enabled:
            model_batch, edm_coeffs = self._precondition_noisy_batch(noisy_batch)

        # Embed crystal components at the configured noisy-input entry point
        inference_cache = model_batch.get("inference_cache")
        if inference_cache is not None:
            s, z = self._cached_input_representations(model_batch, inference_cache)
        elif self.noisy_input_entry == "before_pairmixer":
            s, z = self.input_embedder(model_batch)
        else:
            s, z = self.input_embedder.base_representations(model_batch)
        mask = model_batch["atom_mask"]
        if self.pairmixer is not None and not (
            inference_cache is not None and self.noisy_input_entry == "after_pairmixer"
        ):
            if z is None:
                raise RuntimeError("Pairmixer requires pairwise representation z.")
            s, z = self.pairmixer(s, z, mask)
        if self.noisy_input_entry == "after_pairmixer" and inference_cache is None:
            s = s + self.input_embedder.noisy_single_embedding(model_batch)
            z = self.input_embedder.add_geometry(model_batch, z)

        # Dynamic self-conditioning enters only the single track
        if self.self_conditioning_embedder is not None:
            s = s + self.self_conditioning_embedder(noisy_batch)

        # Compute global conditioning signal from timestep
        t = model_batch["times"]
        c = self.timestep_embedder(t)

        # Reuse each block's pair bias across denoising calls when cached
        projected_pair_biases = (
            None
            if inference_cache is None
            else inference_cache.get("projected_pair_biases")
        )
        if projected_pair_biases is None:
            s = self.transformer(s, c, mask, z)
        else:
            s = self.transformer(
                s, c, mask, z, projected_pair_biases=projected_pair_biases
            )

        # Predict coordinates and lattice (atom types are conditioning input, not predicted)
        preds = self.heads(s, mask)
        if edm_coeffs is None:
            return preds
        return self._apply_edm_output(noisy_batch, preds, edm_coeffs)


class MaterialFlowMatching(nn.Module):
    """CSP flow matching: interpolate coords and lattice, condition on fixed atom types."""

    def __init__(
        self,
        net: nn.Module,
        interpolant: Interpolant,
        scaler: Scaler,
        prior: dict,
        time_sampler: TimeSampler,
        periodic_pair_distance: nn.Module | None = None,
        center_cart_coords: bool = True,
        self_conditioning_probability: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= self_conditioning_probability <= 1.0:
            raise ValueError("self_conditioning_probability must be in [0, 1].")
        self.self_conditioning_probability = float(self_conditioning_probability)
        self.net = net
        self.interpolant = interpolant
        self.scaler = scaler
        self.lattice_repr = scaler.lattice_repr
        self.time_sampler = time_sampler
        self.periodic_pair_distance = periodic_pair_distance
        self.center_cart_coords = bool(center_cart_coords)
        self._mlip_energy_cache: dict[tuple[str, str, str], MLIPEnergy] = {}
        self.autoguidance_bad_net: nn.Module | None = None
        self.autoguidance_weight = 1.0
        self._setup_prior(prior)
        self._validate_edm_preconditioning()

    def set_autoguidance(self, bad_net: nn.Module | None, weight: float) -> None:
        """Configure an optional bad denoiser for autoguided sampling."""
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("autoguidance weight must be finite and >= 0.")
        self.autoguidance_bad_net = bad_net
        self.autoguidance_weight = weight

    def _autoguidance_requires_bad_forward(self) -> bool:
        """Return whether sampling must evaluate the bad denoiser."""
        return self.autoguidance_bad_net is not None and self.autoguidance_weight != 1.0

    def _validate_edm_preconditioning(self) -> None:
        """Ensure EDM preconditioning is only used with linear interpolation."""
        if not bool(getattr(self.net, "edm_preconditioning_enabled", False)):
            return
        if not isinstance(self.interpolant, LinearInterpolant):
            raise ValueError("EDM preconditioning requires LinearInterpolant.")

    def _setup_prior(self, prior_cfg: dict) -> None:
        """Configure prior samplers from registry."""
        coord_sampler = get_sampler("cart_coords", prior_cfg["coord_sampler"])
        self.coord_sampler = partial(coord_sampler, center=self.center_cart_coords)
        base_sampler = get_sampler("lattice_params", prior_cfg["lattice_sampler"])
        lattice_dim = prior_cfg.get("lattice_dim", 6)
        self.lattice_sampler = partial(base_sampler, lattice_dim=lattice_dim)

    def _sample_atom_count(
        self,
        num_atoms: Int["b"],
        conditioning: dict[str, Tensor],
    ) -> int:
        """Return the padded atom dimension used for generation."""
        atomic_numbers = conditioning["atomic_numbers"]
        if atomic_numbers.ndim != 2:
            raise ValueError("conditioning['atomic_numbers'] must have shape (B, N).")
        n = int(atomic_numbers.shape[1])
        if n < int(num_atoms.max().item()):
            raise ValueError(
                "conditioning['atomic_numbers'] is shorter than num_atoms.max()."
            )
        return n

    def _edm_loss_weight(self, t: Float["b"], loss_type: str) -> Float["b"]:
        """Return per-sample EDM loss weights when EDM preconditioning is enabled."""
        if loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        if not bool(getattr(self.net, "edm_preconditioning_enabled", False)):
            return torch.ones_like(t)
        if not bool(getattr(self.net, "edm_loss_weight_enabled", True)):
            return torch.ones_like(t)
        if not hasattr(self.net, "edm_loss_weight"):
            raise TypeError("EDM preconditioning requires net.edm_loss_weight().")
        return self.net.edm_loss_weight(t, loss_type)

    def _get_lattice(self, batch: dict) -> Float["b d"]:
        """Select lattice tensor from batch based on configured representation."""
        if self.lattice_repr in {"cell", "ltri"}:
            return rearrange(batch["cell"], "b i j -> b (i j)")
        return batch["lattice"]

    def _model_lattice_to_cell(self, lattice: Float["b d"]) -> Float["b 9"]:
        """Convert model-space lattice tensors to physical flattened cells."""
        unscaled = self.scaler.unscale_lattice(lattice)
        if self.lattice_repr == "params":
            return lattice_params_to_cell(unscaled)
        return unscaled

    def _coord_physical_scale(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Float[""]:
        """Return angstroms per model coordinate unit for L1 aux scaling."""
        coord_std = getattr(self.scaler, "coord_std", None)
        if coord_std is not None:
            if isinstance(coord_std, torch.Tensor):
                return coord_std.to(device=device, dtype=dtype)
            return torch.tensor(float(coord_std), device=device, dtype=dtype)
        nm_to_ang = getattr(self.scaler, "NM_TO_ANG", None)
        if nm_to_ang is not None:
            return torch.tensor(float(nm_to_ang), device=device, dtype=dtype)
        return torch.ones((), device=device, dtype=dtype)

    def _build_noisy_batch(
        self,
        x_t: Float["b n 3"],
        l_t: Float["b d"],
        times: Float["b"],
        atom_mask: Bool["b n"],
        indices: Int["b n"],
        conditioning: dict,
        conditioning_context: str,
    ) -> dict:
        """Build model inputs with both model-space and physical geometry."""
        return {
            "x_t": x_t,
            "l_t": l_t,
            "times": times,
            "flow_times": times,
            "atom_mask": atom_mask,
            "indices": indices,
            "conditioning": conditioning,
            "conditioning_context": conditioning_context,
            "x_t_physical": self.scaler.unscale_coords(x_t, atom_mask),
            "cell_t_physical": self._model_lattice_to_cell(l_t),
        }

    def forward(
        self,
        batch: dict,
        conditioning_context: str = "train_loss",
    ) -> dict[str, Tensor]:
        """Scale data, sample prior, interpolate coords and lattice, condition on atom types."""
        x1 = batch["cart_coords"]
        l1 = self._get_lattice(batch)
        atom_mask = batch["atom_mask"]
        B = x1.shape[0]

        # Sample timesteps from configured distribution
        t = self.time_sampler.sample(B, x1.device)

        # Scale data to model units before interpolation
        x1_scaled = self.scaler.scale_coords(x1, atom_mask)
        l1_scaled = self.scaler.scale_lattice(l1)

        # Sample standard Gaussian prior (model units)
        x0 = self.coord_sampler(x1, atom_mask.float())
        l0 = self.lattice_sampler(batch_size=B).to(device=x1.device, dtype=x1.dtype)

        # Interpolate continuous modalities in model units
        x_t = self.interpolant.It(t, x0, x1_scaled)
        l_t = self.interpolant.It(t, l0, l1_scaled)

        # Positional indices computed at dataset level, before padding
        indices = batch["indices"]

        # Build noisy_batch for MaterialFlowModel
        noisy_batch = self._build_noisy_batch(
            x_t=x_t,
            l_t=l_t,
            times=t,
            atom_mask=atom_mask,
            indices=indices,
            conditioning=batch["conditioning"],
            conditioning_context=conditioning_context,
        )

        if (
            self.training
            and conditioning_context == "train_loss"
            and self._uses_self_conditioning(self.net)
        ):
            selected = (
                torch.rand(B, device=x_t.device) < self.self_conditioning_probability
            )
            if selected.any():
                subset = tree_map(
                    lambda value: (
                        value[selected] if isinstance(value, Tensor) else value
                    ),
                    noisy_batch,
                )
                # Keep detached autocast weight copies out of the main pass
                device_type = x_t.device.type
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=device_type,
                        enabled=torch.is_autocast_enabled(device_type),
                        dtype=torch.get_autocast_dtype(device_type),
                        cache_enabled=False,
                    ),
                ):
                    preliminary = self.net(subset)
                guess = {
                    "coords": torch.zeros_like(x_t),
                    "lattice": torch.zeros_like(l_t),
                    "available": selected,
                }
                for key in ("coords", "lattice"):
                    guess[key][selected] = preliminary[key].detach().to(guess[key])
                noisy_batch["self_conditioning"] = guess
        preds = self.net(noisy_batch)

        # Include timesteps for downstream loss
        preds["t"] = t

        return preds

    def compute_loss(
        self,
        batch: dict,
        preds: dict[str, Tensor],
        t: Float["b"],
        loss_type: str = "l1",
        reduction: str = "structure",
    ) -> dict[str, Tensor]:
        """Compute coordinate and lattice training losses in model units."""
        mask = batch["atom_mask"]
        x1 = self.scaler.scale_coords(batch["cart_coords"], mask)
        l1 = self.scaler.scale_lattice(self._get_lattice(batch))

        if loss_type == "l1":
            loss_fn = F.l1_loss
        elif loss_type == "l2":
            loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        edm_weight = self._edm_loss_weight(t, loss_type)
        edm_weight_coords = rearrange(edm_weight, "b -> b 1 1")
        edm_weight_lattice = rearrange(edm_weight, "b -> b 1")

        # Coordinate loss: per-sample (B,) then reduce
        coord_err = loss_fn(preds["coords"], x1, reduction="none")  # (B, N, 3)
        coord_err = coord_err * edm_weight_coords
        coord_err = (coord_err * mask[..., None]).sum(dim=-1)  # (B, N)
        per_sample_coords = coord_err.sum(dim=-1) / (mask.sum(dim=-1) * 3)  # (B,)
        if reduction == "atom":
            loss_coords = coord_err.sum() / (mask.sum() * 3)
        else:
            loss_coords = per_sample_coords.mean()

        # Lattice loss: per-sample (B,) then reduce
        if self.lattice_repr == "cell":
            per_sample_lattice_cell = loss_fn(preds["lattice"], l1, reduction="none")
            per_sample_lattice_cell = (
                per_sample_lattice_cell * edm_weight_lattice
            ).mean(dim=-1)  # (B,)
            loss_lattice_cell = per_sample_lattice_cell.mean()
        elif self.lattice_repr == "ltri":
            per_sample_lattice_ltri = loss_fn(preds["lattice"], l1, reduction="none")
            per_sample_lattice_ltri = (
                per_sample_lattice_ltri * edm_weight_lattice
            ).mean(dim=-1)  # (B,)
            loss_lattice_ltri = per_sample_lattice_ltri.mean()
        else:
            per_sample_lattice_lengths = loss_fn(
                preds["lattice"][:, :3], l1[:, :3], reduction="none"
            )
            per_sample_lattice_lengths = (
                per_sample_lattice_lengths * edm_weight_lattice
            ).mean(dim=-1)  # (B,)
            per_sample_lattice_angles = loss_fn(
                preds["lattice"][:, 3:], l1[:, 3:], reduction="none"
            )
            per_sample_lattice_angles = (
                per_sample_lattice_angles * edm_weight_lattice
            ).mean(dim=-1)  # (B,)
            loss_lattice_lengths = per_sample_lattice_lengths.mean()
            loss_lattice_angles = per_sample_lattice_angles.mean()

        result = {
            "loss_coords": loss_coords,
            "per_sample_coords": per_sample_coords,
        }
        if self.lattice_repr == "cell":
            result["loss_lattice_cell"] = loss_lattice_cell
            result["per_sample_lattice_cell"] = per_sample_lattice_cell
        elif self.lattice_repr == "ltri":
            result["loss_lattice_ltri"] = loss_lattice_ltri
            result["per_sample_lattice_ltri"] = per_sample_lattice_ltri
        else:
            result["loss_lattice_lengths"] = loss_lattice_lengths
            result["loss_lattice_angles"] = loss_lattice_angles
            result["per_sample_lattice_lengths"] = per_sample_lattice_lengths
            result["per_sample_lattice_angles"] = per_sample_lattice_angles
        periodic_pair_distance = getattr(self, "periodic_pair_distance", None)
        if periodic_pair_distance is not None:
            coord_scale = self._coord_physical_scale(
                device=preds["coords"].device,
                dtype=preds["coords"].dtype,
            )
            loss_periodic_pair_distance, per_sample_periodic_pair_distance = (
                periodic_pair_distance(
                    pred_coords=self.scaler.unscale_coords(preds["coords"], mask),
                    pred_cell=self._model_lattice_to_cell(preds["lattice"]),
                    true_coords=batch["cart_coords"],
                    true_cell=batch["cell"],
                    atom_mask=mask,
                    coord_scale=coord_scale,
                )
            )
            result["loss_periodic_pair_distance"] = loss_periodic_pair_distance
            result["per_sample_periodic_pair_distance"] = (
                per_sample_periodic_pair_distance
            )
        return result

    def _sde_euler_maruyama_step(
        self,
        x_t: Tensor,
        v: Tensor,
        x1_pred: Tensor,
        t: Float["b"],
        dt: float,
        noise_scale: float,
    ) -> Tensor:
        """Euler-Maruyama SDE step: drift with score correction plus diffusion noise."""
        score = self.interpolant.score_from_x1_pred(t, x_t, x1_pred)
        beta_t = self.interpolant.diffusion_coeff(t)
        beta_t = self.interpolant._expand_t(noise_scale**2 * beta_t, x_t)

        drift = v + 0.5 * beta_t * score
        noise = torch.randn_like(x_t)
        return x_t + drift * dt + torch.sqrt(beta_t * dt) * noise

    def _constrain_sample_coords(
        self,
        x_t: Float["b n 3"],
        atom_mask: Bool["b n"],
    ) -> Float["b n 3"]:
        """Optionally center valid coordinates and always zero padding."""
        mask = atom_mask[..., None].float()
        if self.center_cart_coords:
            mean = (x_t * mask).sum(1, keepdim=True) / mask.sum(
                1, keepdim=True
            ).clamp_min(1)
            x_t = x_t - mean
        return x_t * mask

    def _sample_heun(
        self,
        x_noise: Float["b n 3"],
        l_noise: Float["b d"],
        atom_mask: Bool["b n"],
        conditioning: dict[str, Tensor],
        conditioning_context: str,
        num_steps: int,
        return_trajectory: bool,
        sampler_args: dict[str, Any] | None,
        inference_cache: dict[str, Tensor] | None = None,
        bad_inference_cache: dict[str, Tensor] | None = None,
    ) -> tuple[Float["b n 3"], Float["b d"], list[dict[str, Tensor]] | None]:
        """Sample with stochastic EDM Heun using flow-matching model predictions."""
        if not isinstance(self.interpolant, LinearInterpolant):
            raise NotImplementedError(
                "Heun sampling currently supports LinearInterpolant only. "
                "TODO: derive sigma-to-time mapping for other interpolants."
            )

        config = edm_heun_config_from_dict(sampler_args)
        device = x_noise.device
        batch_size = int(x_noise.shape[0])
        num_nodes = int(x_noise.shape[1])
        sigmas = karras_sigma_schedule(
            num_steps=num_steps,
            config=config,
            device=device,
            dtype=x_noise.dtype,
        )
        indices = repeat(
            torch.arange(num_nodes, device=device), "n -> b n", b=batch_size
        )

        x_sigma = self._constrain_sample_coords(sigmas[0] * x_noise, atom_mask)
        l_sigma = sigmas[0] * l_noise
        trajectory = [] if return_trajectory else None
        atomic_numbers = conditioning["atomic_numbers"]

        history: dict[str, Any] = {}
        for i in range(num_steps):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]
            gamma = churn_gamma(sigma_curr, num_steps, config)
            sigma_hat = sigma_curr * (1.0 + gamma)

            if gamma > 0.0:
                noise_scale = (sigma_hat.square() - sigma_curr.square()).sqrt()
                x_sigma = x_sigma + noise_scale * config.s_noise * torch.randn_like(
                    x_sigma
                )
                x_sigma = self._constrain_sample_coords(x_sigma, atom_mask)
                l_sigma = l_sigma + noise_scale * config.s_noise * torch.randn_like(
                    l_sigma
                )

            t_hat = sigma_to_flow_time(sigma_hat).expand(batch_size)
            x_flow = x_sigma / (1.0 + sigma_hat)
            l_flow = l_sigma / (1.0 + sigma_hat)
            noisy_batch = self._build_noisy_batch(
                x_t=x_flow,
                l_t=l_flow,
                times=t_hat,
                atom_mask=atom_mask,
                indices=indices,
                conditioning=conditioning,
                conditioning_context=conditioning_context,
            )
            preds = self._sample_model_forward(
                noisy_batch,
                inference_cache,
                bad_inference_cache,
                history,
            )

            d_x = edm_derivative(x_sigma, preds["coords"], sigma_hat)
            d_l = edm_derivative(l_sigma, preds["lattice"], sigma_hat)
            sigma_step = sigma_next - sigma_hat
            x_next = x_sigma + sigma_step * d_x
            x_next = self._constrain_sample_coords(x_next, atom_mask)
            l_next = l_sigma + sigma_step * d_l

            if i < num_steps - 1:
                t_next = sigma_to_flow_time(sigma_next).expand(batch_size)
                x_next_flow = x_next / (1.0 + sigma_next)
                l_next_flow = l_next / (1.0 + sigma_next)
                next_batch = self._build_noisy_batch(
                    x_t=x_next_flow,
                    l_t=l_next_flow,
                    times=t_next,
                    atom_mask=atom_mask,
                    indices=indices,
                    conditioning=conditioning,
                    conditioning_context=conditioning_context,
                )
                next_preds = self._sample_model_forward(
                    next_batch,
                    inference_cache,
                    bad_inference_cache,
                    history,
                )
                d_x_next = edm_derivative(x_next, next_preds["coords"], sigma_next)
                d_l_next = edm_derivative(l_next, next_preds["lattice"], sigma_next)
                x_next = x_sigma + sigma_step * (0.5 * d_x + 0.5 * d_x_next)
                x_next = self._constrain_sample_coords(x_next, atom_mask)
                l_next = l_sigma + sigma_step * (0.5 * d_l + 0.5 * d_l_next)

            x_sigma = x_next
            l_sigma = l_next

            if trajectory is not None:
                denom = 1.0 + sigma_next
                trajectory.append(
                    {
                        "cart_coords": self.scaler.unscale_coords(
                            (x_sigma / denom).clone(), atom_mask
                        ),
                        "atomic_numbers": atomic_numbers.clone(),
                        "lattice": self.scaler.unscale_lattice(
                            (l_sigma / denom).clone()
                        ),
                    }
                )

        return x_sigma, l_sigma, trajectory

    def _repeat_batch_tensor(
        self,
        tensor: Tensor,
        multiplicity: int,
    ) -> Tensor:
        """Repeat a batch-major tensor in source-major order."""
        if multiplicity == 1:
            return tensor
        repeated = repeat(tensor, "b ... -> b m ...", m=multiplicity)
        return rearrange(repeated, "b m ... -> (b m) ...")

    def _fk_is_enabled(self, steering_args: dict[str, Any] | None) -> bool:
        """Return whether Feynman-Kac steering is enabled."""
        return bool(
            steering_args is not None and steering_args.get("fk_steering", False)
        )

    def _fk_log_potential(
        self,
        energy_traj: Float["b h"],
        current_energy: Float["b"],
        potential_mode: str,
    ) -> Float["b"]:
        """Compute the FK log-potential from current and historical energies."""
        if potential_mode == "immediate":
            return -current_energy
        if potential_mode == "difference":
            if int(energy_traj.shape[1]) == 1:
                return torch.zeros_like(current_energy)
            return energy_traj[:, -2] - energy_traj[:, -1]
        if potential_mode == "max":
            return -energy_traj.min(dim=1).values
        if potential_mode == "sum":
            return -energy_traj.mean(dim=1)
        raise ValueError(f"Unsupported potential_mode: {potential_mode}")

    def _fk_resample_indices(
        self,
        log_potential: Float["b"],
        num_particles: int,
        fk_lambda: float,
    ) -> Int["b"]:
        """Sample particle indices within each source/multiplicity group."""
        grouped_log_potential = rearrange(
            log_potential,
            "(g k) -> g k",
            k=num_particles,
        )
        weights = F.softmax(grouped_log_potential * fk_lambda, dim=1)
        sampled = torch.multinomial(weights, num_particles, replacement=True)
        offsets = (
            torch.arange(weights.shape[0], device=weights.device)[:, None]
            * num_particles
        )
        return rearrange(sampled + offsets, "g k -> (g k)")

    def _fk_best_particle_indices(
        self,
        final_energy: Float["b"],
        num_particles: int,
    ) -> Int["g"]:
        """Return the lowest-energy particle index for each source/multiplicity group."""
        grouped_energy = rearrange(final_energy, "(g k) -> g k", k=num_particles)
        best_particle = torch.argmin(grouped_energy, dim=1)
        offsets = (
            torch.arange(grouped_energy.shape[0], device=final_energy.device)
            * num_particles
        )
        return best_particle + offsets

    def _get_mlip_energy(
        self,
        model_name: str,
        task_name: str,
        device_name: str,
    ) -> MLIPEnergy:
        """Return a cached MLIP energy module for FK steering."""
        key = (model_name, task_name, device_name)
        potential_fn = self._mlip_energy_cache.get(key)
        if potential_fn is None:
            potential_fn = MLIPEnergy(
                model_name=model_name,
                task_name=task_name,
                device=device_name,
            )
            potential_fn.eval()
            self._mlip_energy_cache[key] = potential_fn
        return potential_fn

    def _inference_cache_net(self, net: nn.Module) -> nn.Module:
        """Return the original denoiser module used to build sampling caches."""
        return getattr(net, "_orig_mod", net)

    def _build_inference_cache(
        self,
        net: nn.Module,
        x_t: Float["b n 3"],
        l_t: Float["b d"],
        atom_mask: Bool["b n"],
        indices: Int["b n"],
        conditioning: dict[str, Tensor],
        conditioning_context: str,
    ) -> dict[str, Tensor]:
        """Build a per-sampling-call denoiser cache."""
        cache_net = self._inference_cache_net(net)
        if not hasattr(cache_net, "build_inference_cache"):
            raise TypeError("Inference cache requires net.build_inference_cache().")
        times = torch.zeros(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        noisy_batch = self._build_noisy_batch(
            x_t=x_t,
            l_t=l_t,
            times=times,
            atom_mask=atom_mask,
            indices=indices,
            conditioning=conditioning,
            conditioning_context=conditioning_context,
        )
        return cache_net.build_inference_cache(noisy_batch)

    def _uses_self_conditioning(self, net: nn.Module) -> bool:
        """Check whether a denoiser has the optional self-conditioning branch."""
        return (
            getattr(self._inference_cache_net(net), "self_conditioning_embedder", None)
            is not None
        )

    def _net_forward(
        self,
        net: nn.Module,
        noisy_batch: dict[str, Any],
        inference_cache: dict[str, Any] | None = None,
        history: dict[str, Any] | None = None,
        branch: str = "main",
    ) -> dict[str, Float["..."]]:
        """Run a denoiser and retain its own clean estimate for the next call."""
        model_batch = dict(noisy_batch)
        if inference_cache is not None:
            model_batch["inference_cache"] = inference_cache
        use_history = history is not None and self._uses_self_conditioning(net)
        if use_history:
            model_batch["self_conditioning"] = history.get(branch)
        preds = net(model_batch)
        if use_history:
            history[branch] = {
                "coords": preds["coords"].detach(),
                "lattice": preds["lattice"].detach(),
                "available": torch.ones_like(noisy_batch["times"], dtype=torch.bool),
            }
        return preds

    def _validate_spacegroup_guidance_weight(
        self,
        weight: float | None,
    ) -> float | None:
        """Validate an optional sampling-time space-group guidance weight."""
        if weight is None:
            return None
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("space-group guidance weight must be finite and >= 0.")
        if weight not in {0.0, 1.0}:
            raise NotImplementedError(
                "Space-group CFG is currently unsupported; use weight=1 for "
                "ordinary conditional inference."
            )
        return weight

    def _autoguided_predictions(
        self,
        main_preds: dict[str, Tensor],
        bad_preds: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Combine main and bad endpoint predictions with autoguidance."""
        guided = dict(main_preds)
        weight = float(self.autoguidance_weight)
        for key in ("coords", "lattice"):
            if key not in main_preds:
                raise ValueError(f"Main model prediction is missing '{key}'.")
            if key not in bad_preds:
                raise ValueError(
                    f"Autoguidance bad model prediction is missing '{key}'."
                )
            if main_preds[key].shape != bad_preds[key].shape:
                raise ValueError(
                    "Autoguidance bad model prediction shape mismatch for "
                    f"'{key}': expected {tuple(main_preds[key].shape)}, "
                    f"got {tuple(bad_preds[key].shape)}."
                )
            guided[key] = weight * main_preds[key] + (1.0 - weight) * bad_preds[key]
        return guided

    def _sample_model_forward(
        self,
        noisy_batch: dict[str, Tensor | dict[str, Tensor]],
        inference_cache: dict[str, Tensor] | None = None,
        bad_inference_cache: dict[str, Tensor] | None = None,
        history: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """Run the denoising model for one effective sample chunk."""
        main_preds = self._net_forward(self.net, noisy_batch, inference_cache, history)
        if not self._autoguidance_requires_bad_forward():
            return main_preds
        if self.autoguidance_bad_net is None:
            raise RuntimeError("Autoguidance bad net is not configured.")
        bad_preds = self._net_forward(
            self.autoguidance_bad_net,
            noisy_batch,
            bad_inference_cache,
            history,
            "bad",
        )
        return self._autoguided_predictions(main_preds, bad_preds)

    @torch.no_grad()
    def sample(
        self,
        num_atoms: Int["b"],
        multiplicity: int = 1,
        num_steps: int = 100,
        method: str = "ode",
        sde_noise_scale: float = 1.0,
        return_trajectory: bool = False,
        *,
        conditioning: dict[str, Tensor],
        conditioning_context: str = "predict",
        steering_args: dict[str, Any] | None = None,
        sampler_args: dict[str, Any] | None = None,
        spacegroup_guidance_weight: float | None = None,
        time_epsilon: float = 1e-3,
        use_inference_cache: bool = False,
    ) -> dict[str, Tensor]:
        """Predict crystal structures via configured flow sampler."""
        if multiplicity < 1:
            raise ValueError("multiplicity must be >= 1.")
        if time_epsilon <= 0.0 or time_epsilon >= 0.5:
            raise ValueError("time_epsilon must satisfy 0 < value < 0.5.")
        self._validate_spacegroup_guidance_weight(spacegroup_guidance_weight)
        atomic_numbers = conditioning["atomic_numbers"]

        method = str(method).lower()
        if method not in {"ode", "sde", "heun"}:
            raise ValueError(f"Unsupported sampling method: {method}.")
        sampler_args = None if sampler_args is None else dict(sampler_args)
        if (
            method == "sde"
            and sampler_args is not None
            and "noise_scale" in sampler_args
        ):
            sde_noise_scale = float(sampler_args["noise_scale"])

        is_steering = self._fk_is_enabled(steering_args)
        if use_inference_cache and is_steering:
            raise ValueError("Inference cache is not supported with FK steering.")
        num_particles = 1
        fk_lambda = 1.0
        resampling_interval = 1
        fk_start_time = 1.0
        potential_mode = "immediate"
        energy_traj = None
        potential_fn = None

        if is_steering:
            if return_trajectory:
                raise ValueError("return_trajectory is not supported with FK steering.")
            if method != "sde":
                logger.warning(
                    "FK steering requires stochastic sampling; overriding method=%s to sde.",
                    method,
                )
                method = "sde"
            if steering_args is None:
                raise ValueError(
                    "steering_args must be provided when FK steering is enabled."
                )
            if steering_args.get("energy_fn", "mlip") != "mlip":
                raise ValueError(
                    "Packora FK steering currently supports energy_fn='mlip' only."
                )
            num_particles = int(steering_args.get("num_particles", 16))
            if num_particles < 1:
                raise ValueError("num_particles must be >= 1.")
            fk_lambda = float(steering_args.get("fk_lambda", 2.0))
            resampling_interval = int(steering_args.get("fk_resampling_interval", 5))
            if resampling_interval < 1:
                raise ValueError("fk_resampling_interval must be >= 1.")
            fk_start_time = float(steering_args.get("fk_start_time", 0.80))
            potential_mode = str(steering_args.get("potential_mode", "immediate"))
            multiplicity = multiplicity * num_particles

        B = int(num_atoms.shape[0])
        M = multiplicity
        BM = B * M
        N = self._sample_atom_count(num_atoms, conditioning)
        device = num_atoms.device

        if is_steering:
            device_name = "cuda" if device.type == "cuda" else "cpu"
            potential_fn = self._get_mlip_energy(
                model_name=str(steering_args.get("mlip_model", "uma-s-1p2")),
                task_name=str(steering_args.get("mlip_task_name", "omc")),
                device_name=device_name,
            )
            energy_traj = torch.empty((BM, 0), device=device)

        # Expand the source batch only when drawing multiple samples per input
        if M > 1:
            num_atoms = self._repeat_batch_tensor(num_atoms, M)
            conditioning = {
                key: self._repeat_batch_tensor(value, M)
                for key, value in conditioning.items()
            }
            atomic_numbers = conditioning["atomic_numbers"]

        model_conditioning = conditioning

        # Build atom_mask from num_atoms
        atom_mask = torch.arange(N, device=device)[None] < num_atoms[:, None]
        indices = repeat(torch.arange(N, device=device), "n -> b n", b=BM)

        # Sample standard Gaussian prior in model units
        x_0 = torch.zeros(BM, N, 3, device=device)
        x_t = self.coord_sampler(x_0, atom_mask.float())
        l_t = self.lattice_sampler(batch_size=BM).to(device=device)
        inference_cache = None
        bad_inference_cache = None
        if use_inference_cache:
            inference_cache = self._build_inference_cache(
                self.net,
                x_t=x_t,
                l_t=l_t,
                atom_mask=atom_mask,
                indices=indices,
                conditioning=model_conditioning,
                conditioning_context=conditioning_context,
            )
            if self._autoguidance_requires_bad_forward():
                if self.autoguidance_bad_net is None:
                    raise RuntimeError("Autoguidance bad net is not configured.")
                bad_inference_cache = self._build_inference_cache(
                    self.autoguidance_bad_net,
                    x_t=x_t,
                    l_t=l_t,
                    atom_mask=atom_mask,
                    indices=indices,
                    conditioning=model_conditioning,
                    conditioning_context=conditioning_context,
                )

        trajectory = [] if return_trajectory else None

        if method == "heun":
            x_t, l_t, trajectory = self._sample_heun(
                x_noise=x_t,
                l_noise=l_t,
                atom_mask=atom_mask,
                conditioning=model_conditioning,
                conditioning_context=conditioning_context,
                num_steps=num_steps,
                return_trajectory=return_trajectory,
                sampler_args=sampler_args,
                inference_cache=inference_cache,
                bad_inference_cache=bad_inference_cache,
            )
        else:
            # Timesteps: avoid singularities at 0 and 1
            timesteps = torch.linspace(
                time_epsilon,
                1.0 - time_epsilon,
                num_steps + 1,
                device=device,
            )

            history: dict[str, Any] = {}
            for i in range(num_steps):
                t_curr = timesteps[i]
                dt = timesteps[i + 1] - t_curr
                t_batch = t_curr.expand(BM)

                noisy_batch = self._build_noisy_batch(
                    x_t=x_t,
                    l_t=l_t,
                    times=t_batch,
                    atom_mask=atom_mask,
                    indices=indices,
                    conditioning=model_conditioning,
                    conditioning_context=conditioning_context,
                )
                preds = self._sample_model_forward(
                    noisy_batch,
                    inference_cache,
                    bad_inference_cache,
                    history,
                )

                if (
                    is_steering
                    and potential_fn is not None
                    and energy_traj is not None
                    and float(t_curr.item()) >= fk_start_time
                    and i % resampling_interval == 0
                ):
                    pred_coords_raw = self.scaler.unscale_coords(
                        preds["coords"], atom_mask
                    )
                    pred_lattice_raw = self.scaler.unscale_lattice(preds["lattice"])
                    current_energy = potential_fn(
                        pred_coords_raw,
                        pred_lattice_raw,
                        atomic_numbers,
                        atom_mask,
                    ).to(device=device)
                    energy_traj = torch.cat(
                        [energy_traj, current_energy[:, None]],
                        dim=1,
                    )
                    log_potential = self._fk_log_potential(
                        energy_traj,
                        current_energy,
                        potential_mode,
                    )
                    selected = self._fk_resample_indices(
                        log_potential,
                        num_particles,
                        fk_lambda,
                    )
                    history = tree_map(lambda value: value[selected], history)
                    x_t = x_t[selected]
                    l_t = l_t[selected]
                    preds["coords"] = preds["coords"][selected]
                    preds["lattice"] = preds["lattice"][selected]
                    energy_traj = energy_traj[selected]
                    atom_mask = atom_mask[selected]
                    conditioning = {
                        key: value[selected] for key, value in conditioning.items()
                    }
                    model_conditioning = conditioning
                    atomic_numbers = conditioning["atomic_numbers"]

                # Continuous step: coords (model units)
                v_x = self.interpolant.velocity_from_x1_pred(
                    t_batch,
                    x_t,
                    preds["coords"],
                )
                if method == "sde" and i < num_steps - 1:
                    x_t = self._sde_euler_maruyama_step(
                        x_t,
                        v_x,
                        preds["coords"],
                        t_batch,
                        dt,
                        sde_noise_scale,
                    )
                else:
                    x_t = x_t + dt * v_x

                x_t = self._constrain_sample_coords(x_t, atom_mask)

                # Continuous step: lattice (model units)
                v_l = self.interpolant.velocity_from_x1_pred(
                    t_batch,
                    l_t,
                    preds["lattice"],
                )
                if method == "sde" and i < num_steps - 1:
                    l_t = self._sde_euler_maruyama_step(
                        l_t,
                        v_l,
                        preds["lattice"],
                        t_batch,
                        dt,
                        sde_noise_scale,
                    )
                else:
                    l_t = l_t + dt * v_l

                # Store trajectory in raw units
                if return_trajectory and trajectory is not None:
                    trajectory.append(
                        {
                            "cart_coords": self.scaler.unscale_coords(
                                x_t.clone(),
                                atom_mask,
                            ),
                            "atomic_numbers": atomic_numbers.clone(),
                            "lattice": self.scaler.unscale_lattice(
                                l_t.clone(),
                            ),
                        }
                    )

        # Unscale from model units to raw units
        x_t = self.scaler.unscale_coords(x_t, atom_mask)
        l_t = self.scaler.unscale_lattice(l_t)

        final_energy = None
        if is_steering and potential_fn is not None:
            final_energy = potential_fn(
                x_t,
                l_t,
                atomic_numbers,
                atom_mask,
            ).to(device=device)
            selected = self._fk_best_particle_indices(final_energy, num_particles)
            x_t = x_t[selected]
            l_t = l_t[selected]
            atomic_numbers = atomic_numbers[selected]
            atom_mask = atom_mask[selected]
            final_energy = final_energy[selected]

        # Zero out padding positions
        x_t = x_t * atom_mask[..., None].float()
        atomic_numbers = atomic_numbers * atom_mask.long()

        result = {
            "cart_coords": x_t,
            "atomic_numbers": atomic_numbers,
            "lattice": l_t,
            "atom_mask": atom_mask,
        }
        if final_energy is not None:
            result["final_energy"] = final_energy
        if return_trajectory:
            result["trajectory"] = trajectory
        return result
