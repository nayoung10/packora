"""Muon optimizer helpers."""

from __future__ import annotations

# ruff: noqa: F722,F821

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim import _functional as optim_functional

from src.utils.tensor_typing import Float

DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
DEFAULT_MUON_EXCLUDE_NAMES = ("input_embedder", "timestep_embedder", "heads")


def _zeropower_batched(
    updates: Float["p m n"],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Float["p m n"]:
    """Orthogonalize same-shape matrix updates with Newton-Schulz iterations."""
    if ns_steps >= 100:
        raise ValueError("ns_steps must be less than 100.")
    if len(ns_coefficients) != 3:
        raise ValueError("ns_coefficients must contain exactly three values.")

    a, b, c = ns_coefficients
    x = updates.bfloat16()
    is_transposed = x.shape[-2] > x.shape[-1]
    if is_transposed:
        x = x.mT

    # Bound spectral norm before iterating
    scale = x.float().norm(dim=(-2, -1), keepdim=True).clamp_min(eps).to(dtype=x.dtype)
    x = x / scale
    for _ in range(ns_steps):
        gram = x @ x.mT
        gram_update = b * gram + c * (gram @ gram)
        x = a * x + gram_update @ x

    if is_transposed:
        x = x.mT
    return x


def _adjust_lr(
    lr: float,
    adjust_lr_fn: str | None,
    param_shape: torch.Size,
) -> float:
    """Return the Muon learning rate adjusted for matrix shape."""
    rows, cols = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return lr * math.sqrt(max(1.0, rows / cols))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(rows, cols))
    if adjust_lr_fn == "none":
        return lr
    raise ValueError(f"Unsupported adjust_lr_fn: {adjust_lr_fn}")


def _as_named_parameters(params: Iterable[Any]) -> list[tuple[str | None, Any]]:
    """Normalize bare or named parameters into name-parameter pairs."""
    named_params: list[tuple[str | None, Any]] = []
    for item in params:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            named_params.append((item[0], item[1]))
        else:
            named_params.append((None, item))
    return named_params


def _uses_muon(
    name: str | None,
    param: Any,
    exclude_names: Sequence[str],
) -> bool:
    """Return whether a parameter should use Muon updates."""
    if param.ndim != 2 or torch.is_complex(param):
        return False
    if name is None:
        return True
    return not any(excluded in name for excluded in exclude_names)


class MuonWithAdamW(Optimizer):
    """Use Muon for hidden 2D weights and AdamW for all remaining parameters."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = "match_rms_adamw",
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        muon_exclude_names: Sequence[str] = DEFAULT_MUON_EXCLUDE_NAMES,
    ) -> None:
        """Initialize mixed Muon and AdamW parameter groups."""
        if lr < 0.0:
            raise ValueError(f"Learning rate must be nonnegative, got {lr}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be nonnegative, got {weight_decay}.")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum must be nonnegative, got {momentum}.")
        if len(adamw_betas) != 2:
            raise ValueError("adamw_betas must contain exactly two values.")
        beta1, beta2 = adamw_betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("adamw_betas must be in [0, 1).")

        named_params = _as_named_parameters(params)
        if not named_params:
            raise ValueError("optimizer got an empty parameter list")

        muon_entries: list[tuple[str | None, Any]] = []
        adamw_entries: list[tuple[str | None, Any]] = []
        for name, param in named_params:
            if _uses_muon(name, param, muon_exclude_names):
                muon_entries.append((name, param))
            else:
                adamw_entries.append((name, param))

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": tuple(ns_coefficients),
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "adamw_betas": tuple(adamw_betas),
            "adamw_eps": adamw_eps,
        }
        param_groups = self._build_param_groups(muon_entries, adamw_entries)
        super().__init__(param_groups, defaults)
        self._muon_buckets = self._build_muon_buckets()

    @staticmethod
    def _build_param_groups(
        muon_entries: list[tuple[str | None, Any]],
        adamw_entries: list[tuple[str | None, Any]],
    ) -> list[dict[str, Any]]:
        """Build optimizer parameter groups from split entries."""
        param_groups: list[dict[str, Any]] = []
        if muon_entries:
            names, params = zip(*muon_entries)
            param_groups.append(
                {
                    "params": list(params),
                    "param_names": list(names),
                    "use_muon": True,
                }
            )
        if adamw_entries:
            names, params = zip(*adamw_entries)
            param_groups.append(
                {
                    "params": list(params),
                    "param_names": list(names),
                    "use_muon": False,
                }
            )
        return param_groups

    def _build_muon_buckets(self) -> list[tuple[dict[str, Any], list[list[Any]]]]:
        """Group Muon parameters by shape for batched updates."""
        bucket_groups: list[tuple[dict[str, Any], list[list[Any]]]] = []
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            buckets: dict[tuple[int, ...], list[Any]] = defaultdict(list)
            for param in group["params"]:
                buckets[tuple(param.shape)].append(param)
            bucket_groups.append((group, list(buckets.values())))
        return bucket_groups

    @torch.no_grad()
    def _step_muon_group(self, group: dict[str, Any], buckets: list[list[Any]]) -> None:
        """Run one Muon update for a shape-bucketed parameter group."""
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        momentum = float(group["momentum"])

        for bucket in buckets:
            active_params = []
            grads = []
            momentum_buffers = []
            for param in bucket:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        param.grad,
                        memory_format=torch.preserve_format,
                    )
                active_params.append(param)
                grads.append(param.grad)
                momentum_buffers.append(state["momentum_buffer"])

            if not active_params:
                continue

            torch._foreach_lerp_(momentum_buffers, grads, 1 - momentum)
            if group["nesterov"]:
                updates = torch._foreach_lerp(grads, momentum_buffers, momentum)
            else:
                updates = momentum_buffers

            update_stack = torch.stack(tuple(updates))
            update_stack = _zeropower_batched(
                update_stack,
                group["ns_coefficients"],
                group["ns_steps"],
                group["eps"],
            )
            adjusted_lr = _adjust_lr(
                lr,
                group["adjust_lr_fn"],
                active_params[0].shape,
            )
            if weight_decay != 0.0:
                torch._foreach_mul_(active_params, 1 - lr * weight_decay)
            torch._foreach_add_(
                active_params,
                list(update_stack.unbind(0)),
                alpha=-adjusted_lr,
            )

    @torch.no_grad()
    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        """Run one AdamW update for fallback parameters."""
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_steps = []
        has_complex = False

        for param in group["params"]:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients.")
            has_complex = has_complex or torch.is_complex(param)
            state = self.state[param]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(
                    param,
                    memory_format=torch.preserve_format,
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param,
                    memory_format=torch.preserve_format,
                )
            params_with_grad.append(param)
            grads.append(param.grad)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])

        if not params_with_grad:
            return

        beta1, beta2 = group["adamw_betas"]
        optim_functional.adamw(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            [],
            state_steps,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            has_complex=has_complex,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["adamw_eps"],
            maximize=False,
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Run one mixed Muon and AdamW optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group, buckets in self._muon_buckets:
            self._step_muon_group(group, buckets)
        for group in self.param_groups:
            if not group["use_muon"]:
                self._step_adamw_group(group)
        return loss
