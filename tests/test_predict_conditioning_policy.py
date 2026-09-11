import pytest
from omegaconf import OmegaConf

from src.prediction.config import (
    center_cart_coords_from_run_config,
    resolve_conditioning_policy,
    resolve_max_num_atoms,
    resolve_model_overrides,
    resolve_source_spec,
    validate_matching_centering_modes,
)
from src.prediction.sampling import (
    effective_spacegroup_cfg_weight,
    sample_chunk_size,
    samples_per_datapoint,
    spacegroup_cfg_weight,
    time_epsilon,
)


def test_resolve_conditioning_policy_applies_direct_prediction_modes() -> None:
    """Apply explicit prediction conditioning modes to the resolved context."""
    run_cfg = OmegaConf.create(
        {
            "conditioning": {
                "context_by_role": {"predict": "predict"},
                "policy": {
                    "dropout_probs": {
                        "template": 0.5,
                        "stereochemistry": 0.5,
                        "spacegroup": 0.9,
                    },
                    "contexts": {
                        "train_loss": {
                            "template": "stochastic",
                            "stereochemistry": "stochastic",
                            "spacegroup": "stochastic",
                        },
                        "predict": {
                            "template": "on",
                            "stereochemistry": "off",
                            "spacegroup": "off",
                        },
                    },
                },
            },
        }
    )
    cfg = OmegaConf.create(
        {
            "sampling": {"conditioning_context": "predict"},
            "conditioning_policy": {
                "template": "off",
                "stereochemistry": "on",
                "spacegroup": False,
            },
        }
    )

    policy = resolve_conditioning_policy(cfg, run_cfg)

    assert policy.contexts.predict.template == "off"
    assert policy.contexts.predict.stereochemistry == "on"
    assert policy.contexts.predict.spacegroup == "off"
    assert policy.contexts.train_loss.template == "stochastic"
    assert policy.dropout_probs.spacegroup == 0.9


def test_resolve_conditioning_policy_rejects_nested_overrides() -> None:
    """Reject the old raw conditioning-policy merge shape."""
    run_cfg = OmegaConf.create(
        {
            "conditioning": {
                "context_by_role": {"predict": "predict"},
                "policy": {
                    "dropout_probs": {
                        "template": 0.5,
                        "stereochemistry": 0.5,
                        "spacegroup": 0.9,
                    },
                    "contexts": {
                        "predict": {
                            "template": "off",
                            "stereochemistry": "off",
                            "spacegroup": "off",
                        },
                    },
                },
            },
        }
    )
    cfg = OmegaConf.create(
        {
            "sampling": {"conditioning_context": "predict"},
            "conditioning_policy": {
                "contexts": {
                    "predict": {
                        "template": "on",
                        "stereochemistry": "off",
                        "spacegroup": "off",
                    }
                }
            },
        }
    )

    with pytest.raises(ValueError, match="conditioning_policy only supports"):
        resolve_conditioning_policy(cfg, run_cfg)


def test_resolve_model_overrides_translates_compile_toggle() -> None:
    """Translate the explicit compile toggle to model constructor fields."""
    enabled_cfg = OmegaConf.create({"model_overrides": {"compile": True}})
    disabled_cfg = OmegaConf.create({"model_overrides": {"compile": False}})

    assert resolve_model_overrides(enabled_cfg).compile_target == "net"
    assert resolve_model_overrides(disabled_cfg).compile_target is None


def test_resolve_model_overrides_rejects_raw_model_fields() -> None:
    """Reject internal model override fields at the prediction config boundary."""
    cfg = OmegaConf.create({"model_overrides": {"compile_target": "net"}})

    with pytest.raises(ValueError, match="model_overrides only supports"):
        resolve_model_overrides(cfg)


def test_resolve_source_spec_uses_benchmark_dataset() -> None:
    """Benchmark prediction resolves to the shared benchmark dataset."""
    cfg = OmegaConf.create(
        {
            "paths": {"data_dir": "/example/data"},
            "source": {
                "benchmark": "oxtal",
                "data_dir": None,
                "dataset_name": None,
                "split": "test",
            },
            "data": {"max_num_atoms": None},
        }
    )
    run_cfg = OmegaConf.create(
        {
            "data": {
                "data_dir": "/ignored/data",
                "dataset_name": "csd",
                "max_num_atoms": 300,
            }
        }
    )

    source = resolve_source_spec(cfg, run_cfg)

    assert source.data_dir == "/example/data"
    assert source.dataset_name == "csd_benchmarks"
    assert source.split == "oxtal"
    assert source.benchmark == "oxtal"
    assert resolve_max_num_atoms(cfg, run_cfg) is None


def test_prediction_sampling_helpers_resolve_new_batching_names() -> None:
    """Resolve prediction batching controls from the new config names."""
    cfg = OmegaConf.create(
        {
            "sampling": {
                "samples_per_datapoint": 5,
                "sample_chunk_size": None,
                "time_epsilon": 0.002,
            }
        }
    )

    assert samples_per_datapoint(cfg) == 5
    assert sample_chunk_size(cfg) == 5
    assert time_epsilon(cfg) == 0.002

    cfg.sampling.sample_chunk_size = 2

    assert sample_chunk_size(cfg) == 2


@pytest.mark.parametrize(
    ("policy_mode", "configured_weight", "expected_weight"),
    [
        ("off", 1.0, 0.0),
        ("off", 2.0, 0.0),
        ("on", None, 1.0),
        ("on", 1.0, 1.0),
    ],
)
def test_effective_spacegroup_cfg_weight_truth_table(
    policy_mode: str,
    configured_weight: float | None,
    expected_weight: float,
) -> None:
    """Resolve the policy master switch and sampling guidance weight together."""
    run_cfg = OmegaConf.create(
        {
            "conditioning": {
                "context_by_role": {"predict": "predict"},
                "policy": {
                    "dropout_probs": {
                        "template": 0.5,
                        "stereochemistry": 0.5,
                        "spacegroup": 0.9,
                    },
                    "contexts": {
                        "predict": {
                            "template": "on",
                            "stereochemistry": "off",
                            "spacegroup": "off",
                        }
                    },
                },
            }
        }
    )
    cfg = OmegaConf.create(
        {
            "sampling": {
                "conditioning_context": "predict",
                "spacegroup_cfg": {"weight": configured_weight},
            },
            "conditioning_policy": {"spacegroup": policy_mode},
        }
    )

    assert spacegroup_cfg_weight(cfg) == configured_weight
    assert effective_spacegroup_cfg_weight(cfg, run_cfg) == expected_weight


@pytest.mark.parametrize("configured_weight", [0.0, 2.0])
def test_effective_spacegroup_cfg_weight_rejects_non_unit_conditioning(
    configured_weight: float,
) -> None:
    """Reject non-unit weights when ordinary space-group conditioning is on."""
    run_cfg = OmegaConf.create(
        {
            "conditioning": {
                "context_by_role": {"predict": "predict"},
                "policy": {
                    "dropout_probs": {
                        "template": 0.5,
                        "stereochemistry": 0.5,
                        "spacegroup": 0.9,
                    },
                    "contexts": {
                        "predict": {
                            "template": "on",
                            "stereochemistry": "off",
                            "spacegroup": "off",
                        }
                    },
                },
            }
        }
    )
    cfg = OmegaConf.create(
        {
            "sampling": {
                "conditioning_context": "predict",
                "spacegroup_cfg": {"weight": configured_weight},
            },
            "conditioning_policy": {"spacegroup": "on"},
        }
    )

    with pytest.raises(NotImplementedError, match="CFG is currently unsupported"):
        effective_spacegroup_cfg_weight(cfg, run_cfg)


def test_checkpoint_centering_mode_defaults_to_legacy_true() -> None:
    """Read the saved centering mode while defaulting old runs to centered."""
    legacy = OmegaConf.create({"model": {}})
    uncentered = OmegaConf.create({"model": {"center_cart_coords": False}})

    assert center_cart_coords_from_run_config(legacy) is True
    assert center_cart_coords_from_run_config(uncentered) is False


def test_autoguidance_rejects_mixed_centering_modes() -> None:
    """Reject guidance checkpoints trained in incompatible coordinate gauges."""
    centered = OmegaConf.create({"model": {"center_cart_coords": True}})
    uncentered = OmegaConf.create({"model": {"center_cart_coords": False}})

    with pytest.raises(ValueError, match="same model.center_cart_coords"):
        validate_matching_centering_modes(centered, uncentered)
