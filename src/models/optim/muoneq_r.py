"""MuonEq-R optimizer for matrix parameters with AdamW fallback."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer

from src.utils.tensor_typing import Float

POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def polar_express_orthogonalize(
    update: Float["m n"],
    steps: int = 5,
) -> Float["m n"]:
    """Orthogonalize a matrix update with Polar Express iterations."""
    if steps < 0 or steps > len(POLAR_EXPRESS_COEFFS):
        raise ValueError(f"steps must be in [0, {len(POLAR_EXPRESS_COEFFS)}].")

    x = update.bfloat16()
    x = x / (x.float().norm().to(dtype=x.dtype) * 1.02 + 1e-6)
    if x.shape[-2] > x.shape[-1]:
        for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
            gram = x.mT @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.to(dtype=update.dtype)


class MuonEqRWithAdamW(Optimizer):
    """Use MuonEq-R for 2D parameters and AdamW for remaining parameters."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        beta2: float = 0.95,
        adamw_betas: tuple[float, float] = (0.95, 0.95),
        adamw_eps: float = 1e-8,
        row_eps: float = 1e-7,
        variance_eps: float = 1e-10,
    ) -> None:
        """Initialize split MuonEq-R and AdamW optimizer state."""
        params_list = list(params)
        if not params_list:
            raise ValueError("optimizer got an empty parameter list")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1).")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "beta2": beta2,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
            "row_eps": row_eps,
            "variance_eps": variance_eps,
        }
        super().__init__(params_list, defaults)

        for param in params_list:
            self.state[param]["use_muoneq"] = param.ndim == 2

    @staticmethod
    def _adjust_lr_for_muon(lr: float, param_shape: torch.Size) -> float:
        """Scale Muon-style matrix learning rates by parameter shape."""
        rows, cols = param_shape[:2]
        return lr * 0.2 * math.sqrt(max(rows, cols))

    @staticmethod
    def _variance_reduction(
        update: Float["m n"],
        second_moment: Float["m one"] | Float["one n"],
        beta2: float,
        variance_eps: float,
    ) -> Float["m n"]:
        """Apply MuonEq-R variance reduction to an orthogonalized update."""
        red_dim = -1 if update.shape[-2] >= update.shape[-1] else -2
        red_dim_size = update.shape[red_dim]
        v_mean = update.float().square().mean(dim=red_dim, keepdim=True)
        v_norm = (v_mean.sum() * red_dim_size).sqrt()
        second_moment.lerp_(v_mean.to(dtype=second_moment.dtype), 1 - beta2)
        step_size = second_moment.clamp_min(variance_eps).rsqrt()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum().sqrt()
        final_scale = step_size * (v_norm / v_norm_new.clamp_min(variance_eps))
        return update * final_scale.to(dtype=update.dtype)

    @torch.no_grad()
    def _step_muoneq_param(
        self,
        param: Float["m n"],
        group: dict[str, Any],
    ) -> None:
        """Update a 2D parameter with MuonEq-R."""
        grad = param.grad
        if grad is None:
            return
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        momentum_buffer = state["momentum_buffer"]
        momentum_buffer.lerp_(grad, 1 - group["momentum"])
        update = grad.lerp(momentum_buffer, group["momentum"]) if group["nesterov"] else momentum_buffer

        update = update / update.float().norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(group["row_eps"]).to(dtype=update.dtype)
        update = polar_express_orthogonalize(update, steps=group["ns_steps"])

        if "second_momentum_buffer" not in state:
            red_dim = -1 if param.shape[-2] >= param.shape[-1] else -2
            second_shape = (param.shape[-2], 1) if red_dim == -1 else (1, param.shape[-1])
            state["second_momentum_buffer"] = torch.zeros(
                second_shape,
                dtype=torch.float32,
                device=param.device,
            )
        update = self._variance_reduction(
            update,
            state["second_momentum_buffer"],
            group["beta2"],
            group["variance_eps"],
        ).to(dtype=param.dtype)

        lr = self._adjust_lr_for_muon(group["lr"], param.shape)
        weight_decay = group["weight_decay"]
        decay_mask = (update * param) >= 0
        param.sub_(lr * update + lr * weight_decay * param * decay_mask)

    @torch.no_grad()
    def _step_adamw_param(
        self,
        param: Float["..."],
        group: dict[str, Any],
    ) -> None:
        """Update a non-matrix parameter with AdamW."""
        grad = param.grad
        if grad is None:
            return
        beta1, beta2 = group["adamw_betas"]
        state = self.state[param]
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
        state["step"] += 1

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2)

        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add(group["adamw_eps"])

        param.mul_(1 - group["lr"] * group["weight_decay"])
        param.add_(exp_avg / denom, alpha=-group["lr"] / bias_correction1)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Run one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if self.state[param]["use_muoneq"]:
                    self._step_muoneq_param(param, group)
                else:
                    self._step_adamw_param(param, group)
        return loss
