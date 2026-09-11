from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.finetune import initialize_stage2_weights, prepare_finetune_config
from src.models.callbacks.ema import EMA
from src.prediction.checkpoint import load_checkpoint_weights


class _EmaInitializableLinear(torch.nn.Linear):
    """Linear test model that captures explicit EMA initialization weights."""

    def __init__(self) -> None:
        """Initialize a small linear model and empty EMA state."""
        super().__init__(2, 1)
        self.ema_initial_weights: dict[str, Any] | None = None

    @property
    def device(self) -> torch.device:
        """Return the parameter device expected by the EMA callback."""
        return self.weight.device

    def set_ema_initial_weights(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        """Capture the requested EMA initialization state."""
        self.ema_initial_weights = state_dict


def _write_source_run(
    root: Path,
    learning_rate: float = 2.0e-4,
    warmup_steps: int = 500,
) -> tuple[Path, Path]:
    """Write a minimal checkpoint path and associated Hydra model config."""
    checkpoint_path = root / "checkpoints" / "model.ckpt"
    config_path = root / ".hydra" / "config.yaml"
    checkpoint_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    checkpoint_path.touch()
    run_cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "example.Model",
                "net": {"width": 384},
                "optimizer": {"_target_": "example.Optimizer", "lr": learning_rate},
                "scheduler": {
                    "_target_": "example.Scheduler",
                    "target_lr": learning_rate,
                    "warmup_no_steps": warmup_steps,
                },
            }
        }
    )
    OmegaConf.save(run_cfg, config_path)
    return checkpoint_path, config_path


def _finetune_cfg(
    init_ckpt_path: Path | None = None,
    ckpt_path: Path | None = None,
    run_config_path: Path | None = None,
) -> DictConfig:
    """Build a minimal finetune config for checkpoint-resolution tests."""
    return OmegaConf.create(
        {
            "init_ckpt_path": (
                str(init_ckpt_path) if init_ckpt_path is not None else None
            ),
            "ckpt_path": str(ckpt_path) if ckpt_path is not None else None,
            "run_config_path": (
                str(run_config_path) if run_config_path is not None else None
            ),
            "resolved_run_config_path": None,
            "finetune": {
                "learning_rate": 1.0e-4,
                "warmup_steps": 10000,
                "init_weight_source": {"raw": "raw", "ema": "ema"},
                "ema_warm_start": False,
            },
        }
    )


def test_prepare_finetune_config_infers_source_and_applies_settings(
    tmp_path: Path,
) -> None:
    """New-stage setup preserves model shape and applies visible stage-2 settings."""
    checkpoint_path, config_path = _write_source_run(tmp_path / "source")
    cfg = _finetune_cfg(init_ckpt_path=checkpoint_path)

    selected_path, is_resume = prepare_finetune_config(cfg)

    assert selected_path == checkpoint_path.resolve()
    assert is_resume is False
    assert cfg.resolved_run_config_path == str(config_path.resolve())
    assert cfg.model.net.width == 384
    assert cfg.model.optimizer.lr == pytest.approx(1.0e-4)
    assert cfg.model.scheduler.target_lr == pytest.approx(1.0e-4)
    assert cfg.model.scheduler.warmup_no_steps == 10000
    assert cfg.model.ema_warm_start is False
    assert cfg.model.center_cart_coords is True


def test_prepare_finetune_config_accepts_explicit_run_config(tmp_path: Path) -> None:
    """Copied checkpoints can use an explicitly supplied source run config."""
    _, config_path = _write_source_run(tmp_path / "source")
    copied_checkpoint = tmp_path / "copied.ckpt"
    copied_checkpoint.touch()
    cfg = _finetune_cfg(
        init_ckpt_path=copied_checkpoint,
        run_config_path=config_path,
    )

    selected_path, is_resume = prepare_finetune_config(cfg)

    assert selected_path == copied_checkpoint.resolve()
    assert is_resume is False
    assert cfg.resolved_run_config_path == str(config_path.resolve())


def test_prepare_finetune_config_preserves_saved_settings_on_resume(
    tmp_path: Path,
) -> None:
    """Full resume uses its saved model config instead of new-stage overrides."""
    checkpoint_path, _ = _write_source_run(tmp_path / "stage2")
    cfg = _finetune_cfg(ckpt_path=checkpoint_path)

    _, is_resume = prepare_finetune_config(cfg)

    assert is_resume is True
    assert cfg.model.optimizer.lr == pytest.approx(2.0e-4)
    assert cfg.model.scheduler.target_lr == pytest.approx(2.0e-4)
    assert cfg.model.scheduler.warmup_no_steps == 500


def test_prepare_finetune_config_requires_checkpoint() -> None:
    """Finetuning fails clearly when neither checkpoint mode is configured."""
    with pytest.raises(ValueError, match="init_ckpt_path"):
        prepare_finetune_config(_finetune_cfg())


def test_load_checkpoint_weights_selects_raw_weights(tmp_path: Path) -> None:
    """Raw initialization ignores EMA and unrelated Trainer checkpoint state."""
    source = torch.nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.bias.fill_(3.0)
    ema_weights = {
        "weight": torch.full_like(source.weight, 7.0),
        "bias": torch.full_like(source.bias, 8.0),
    }
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "ema": {"cur_step": 9, "ema_weights": ema_weights},
            "optimizer_states": [{"ignored": True}],
            "lr_schedulers": [{"ignored": True}],
            "epoch": 4,
            "global_step": 99,
        },
        checkpoint_path,
    )
    target = torch.nn.Linear(2, 1)

    load_checkpoint_weights(target, checkpoint_path, eval_with_ema=False)

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


@pytest.mark.parametrize(
    ("raw_source", "ema_source", "expected_raw", "expected_ema"),
    [
        ("raw", "raw", 2.0, 2.0),
        ("raw", "ema", 2.0, 7.0),
        ("ema", "raw", 7.0, 2.0),
        ("ema", "ema", 7.0, 7.0),
    ],
)
def test_initialize_stage2_weights_selects_raw_and_ema_independently(
    tmp_path: Path,
    raw_source: str,
    ema_source: str,
    expected_raw: float,
    expected_ema: float,
) -> None:
    """Stage-2 raw and EMA initializations independently select checkpoint state."""
    source = torch.nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.bias.fill_(2.0)
    ema_weights = {
        "weight": torch.full_like(source.weight, 7.0),
        "bias": torch.full_like(source.bias, 7.0),
    }
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "ema": {"cur_step": 9, "ema_weights": ema_weights},
        },
        checkpoint_path,
    )
    target = _EmaInitializableLinear()

    initialize_stage2_weights(
        target,
        checkpoint_path,
        raw_source=raw_source,
        ema_source=ema_source,
    )

    assert torch.equal(target.weight, torch.full_like(target.weight, expected_raw))
    assert torch.equal(target.bias, torch.full_like(target.bias, expected_raw))
    assert target.ema_initial_weights is not None
    assert torch.equal(
        target.ema_initial_weights["weight"],
        torch.full_like(target.weight, expected_ema),
    )
    assert torch.equal(
        target.ema_initial_weights["bias"],
        torch.full_like(target.bias, expected_ema),
    )


def test_initialize_stage2_weights_requires_requested_ema(tmp_path: Path) -> None:
    """EMA initialization fails clearly when the checkpoint has no EMA state."""
    source = torch.nn.Linear(2, 1)
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": source.state_dict()}, checkpoint_path)

    with pytest.raises(KeyError, match="no EMA state"):
        initialize_stage2_weights(
            _EmaInitializableLinear(),
            checkpoint_path,
            raw_source="raw",
            ema_source="ema",
        )


def test_ema_uses_explicit_initial_weights_without_warm_start() -> None:
    """EMA callback honors explicit initial weights and fixed decay settings."""
    model = _EmaInitializableLinear()
    initial_weights = {
        "weight": torch.full_like(model.weight, 7.0),
        "bias": torch.full_like(model.bias, 8.0),
    }
    callback = EMA(
        decay=0.9999,
        warm_start=False,
        initial_weights=initial_weights,
    )

    callback.on_train_start(trainer=None, pl_module=model)

    ema_weights = callback.state_dict()["ema_weights"]
    assert callback.warm_start is False
    assert torch.equal(ema_weights["weight"], initial_weights["weight"])
    assert torch.equal(ema_weights["bias"], initial_weights["bias"])
