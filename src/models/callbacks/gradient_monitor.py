"""Callback for logging per-component gradient norms during training."""

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.grads import grad_norm


# Model sub-modules to monitor individually.
_COMPONENTS = (
    "input_embedder",
    "timestep_embedder",
    "pairmixer",
    "transformer",
    "heads",
)


class GradientMonitor(Callback):
    """Log gradient norms per model component for debugging."""

    def __init__(self, log_every_n_steps: int = 50) -> None:
        """Initialize with logging frequency."""
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: object,
    ) -> None:
        """Compute and log gradient norms after backward pass."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Total gradient norm across all parameters
        norms = grad_norm(pl_module, norm_type=2.0)
        pl_module.log("debug/grad_norm_total", norms["grad_2.0_norm_total"])

        # Per-component gradient norms
        net = pl_module.flow_matching.net
        for name in _COMPONENTS:
            module = getattr(net, name, None)
            if module is None:
                continue
            comp_norms = grad_norm(module, norm_type=2.0)
            pl_module.log(
                f"debug/grad_norm_{name}",
                comp_norms["grad_2.0_norm_total"],
            )
