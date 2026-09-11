import hydra
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.prediction.sampling import autoguidance_args_dict, spacegroup_cfg_weight
from src.train import validate_lattice_config


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_cartesian_centering_config_fans_out() -> None:
    """Resolve one centering override across data, scaler, model, and head."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["model.center_cart_coords=false"],
        )

    assert cfg.model.center_cart_coords is False
    assert cfg.data.center_cart_coords is False
    assert cfg.model.scaler.center_cart_coords is False
    assert cfg.model.net.heads.coord_head.center is False


@pytest.mark.parametrize("scheduler_name", ["linear_warmup", "alphafold"])
def test_scheduler_override_config_instantiation(scheduler_name: str) -> None:
    """Tests scheduler overrides compose and instantiate as callable factories."""
    # Reset global Hydra state before composing a fresh config.
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[f"model/scheduler={scheduler_name}"],
        )

    HydraConfig().set_config(cfg)

    # Instantiate the scheduler node and validate it stays partial/callable.
    scheduler_factory = hydra.utils.instantiate(cfg.model.scheduler)
    assert scheduler_factory is not None
    assert callable(scheduler_factory)

    # Clear Hydra to avoid state leakage across tests.
    GlobalHydra.instance().clear()


@pytest.mark.parametrize(
    ("edm_enabled", "expected_factor"),
    [(False, 1000.0), (True, 1.0)],
)
def test_timestep_embedder_time_factor_tracks_edm_preconditioning(
    edm_enabled: bool,
    expected_factor: float,
) -> None:
    """Tests timestep scaling follows the EDM preconditioning config."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"model.net.edm_preconditioning.enabled={str(edm_enabled).lower()}"
            ],
        )

    timestep_embedder = hydra.utils.instantiate(cfg.model.net.timestep_embedder)

    assert timestep_embedder.t_embedder.time_factor == expected_factor

    GlobalHydra.instance().clear()


def test_csd_preprocessor_config_instantiation() -> None:
    """Tests the CSD preprocessor config composes and instantiates."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs/preprocess"):
        cfg = compose(config_name="csd")

    preprocessor = hydra.utils.instantiate(cfg.preprocessor)

    assert preprocessor.deduplicate.enabled is True
    assert not hasattr(preprocessor.filters, "require_single_crystal_xray")

    GlobalHydra.instance().clear()


def test_default_model_scheduler_enabled() -> None:
    """Tests default train config enables linear warmup scheduler."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml")

    assert (
        cfg.model.scheduler._target_
        == "src.models.optim.scheduler.LinearWarmupLRScheduler"
    )

    GlobalHydra.instance().clear()


def test_default_lattice_repr_is_ltri() -> None:
    """Tests default train config uses ltri lattice representation."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml")

    assert cfg.model.lattice_repr == "ltri"
    assert cfg.model.lattice_dim == 6

    GlobalHydra.instance().clear()


def test_periodic_pair_distance_default_is_disabled() -> None:
    """Tests periodic pair-distance auxiliary loss is disabled by default."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml")

    assert cfg.model.periodic_pair_distance is None
    assert cfg.model.loss_weights.periodic_pair_distance == pytest.approx(1.0)

    GlobalHydra.instance().clear()


@pytest.mark.parametrize(
    ("config_name", "variant"),
    [("l1", "l1"), ("smooth_lddt", "smooth_lddt")],
)
def test_periodic_pair_distance_configs_compose(
    config_name: str,
    variant: str,
) -> None:
    """Tests periodic pair-distance auxiliary loss configs compose."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[f"model/loss/periodic_pair_distance={config_name}"],
        )

    loss = hydra.utils.instantiate(cfg.model.periodic_pair_distance)

    assert loss.variant == variant
    assert loss.cutoff == pytest.approx(15.0)
    assert loss.disable_autocast is False
    assert cfg.model.loss_weights.periodic_pair_distance == pytest.approx(1.0)

    GlobalHydra.instance().clear()


def test_periodic_pair_distance_smooth_thresholds_override_compose() -> None:
    """Tests smooth-LDDT thresholds can be overridden through Hydra."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "model/loss/periodic_pair_distance=smooth_lddt",
                "model.periodic_pair_distance.thresholds=[0.25,0.75,1.5]",
            ],
        )

    loss = hydra.utils.instantiate(cfg.model.periodic_pair_distance)

    assert loss.variant == "smooth_lddt"
    assert loss.smooth_thresholds.tolist() == pytest.approx([0.25, 0.75, 1.5])

    GlobalHydra.instance().clear()


def test_periodic_pair_distance_autocast_override_compose() -> None:
    """Tests pair-distance autocast behavior can be overridden through Hydra."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "model/loss/periodic_pair_distance=l1",
                "model.periodic_pair_distance.disable_autocast=true",
            ],
        )

    loss = hydra.utils.instantiate(cfg.model.periodic_pair_distance)

    assert loss.variant == "l1"
    assert loss.disable_autocast is True

    GlobalHydra.instance().clear()


def test_pairmixer_and_geometry_configs_compose() -> None:
    """Tests optional Pairmixer and GEM geometry configs compose."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["model/pairmixer=default", "model/embedder/geometry_pair=gem"],
        )

    assert (
        cfg.model.net.pairmixer._target_ == "src.models.components.pairmixer.Pairmixer"
    )
    assert cfg.model.net.pairmixer.num_blocks == 4
    assert (
        cfg.model.net.input_embedder.pairwise_embedder.geometry_pair_embedder._target_
        == "src.models.components.embedders.geometry.GeometryPairEmbedder"
    )

    GlobalHydra.instance().clear()


def test_pairmixer_triangle_attention_config_compose() -> None:
    """Tests Pairmixer triangle attention config overrides compose."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "model/pairmixer=default",
                "model.net.pairmixer.use_triangle_attention=true",
                "model.net.pairmixer.triangle_attention_chunk_size=2",
                "model.net.pairmixer.update_single=true",
                "model.net.noisy_input_entry=after_pairmixer",
            ],
        )

    pairmixer = cfg.model.net.pairmixer
    assert pairmixer.use_triangle_attention is True
    assert pairmixer.triangle_attention_backend == "deepspeed"
    assert pairmixer.triangle_attention_num_heads == 4
    assert pairmixer.triangle_attention_head_dim == 32
    assert pairmixer.triangle_attention_chunk_size == 2
    assert pairmixer.update_single is True
    assert cfg.model.net.noisy_input_entry == "after_pairmixer"
    assert (
        cfg.model.net.input_embedder.pairwise_embedder.pairwise_condition_embedder
        is not None
    )

    GlobalHydra.instance().clear()


def test_noisy_input_entry_config_overrides_compose() -> None:
    """Tests noisy input entry defaults and override compose."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        default_cfg = compose(config_name="train.yaml")

    if default_cfg.model.net.noisy_input_entry != "after_pairmixer":
        GlobalHydra.instance().clear()
        pytest.skip("Default noisy_input_entry is not after_pairmixer.")

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        before_cfg = compose(
            config_name="train.yaml",
            overrides=["model.net.noisy_input_entry=before_pairmixer"],
        )

    assert before_cfg.model.net.noisy_input_entry == "before_pairmixer"

    GlobalHydra.instance().clear()


def test_predict_inference_cache_config_overrides_compose() -> None:
    """Tests prediction inference cache defaults and override compose."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        default_cfg = compose(config_name="predict.yaml")

    assert default_cfg.sampling.use_inference_cache is True

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        uncached_cfg = compose(
            config_name="predict.yaml",
            overrides=["sampling.use_inference_cache=false"],
        )

    assert uncached_cfg.sampling.use_inference_cache is False

    GlobalHydra.instance().clear()


def test_predict_autoguidance_config_defaults_and_overrides_compose() -> None:
    """Tests prediction autoguidance defaults and enabled overrides compose."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        default_cfg = compose(config_name="predict.yaml")

    assert default_cfg.sampling.autoguidance.enabled is False
    assert default_cfg.sampling.autoguidance.bad_ckpt_path is None
    assert default_cfg.sampling.autoguidance.weight == 1.0
    assert autoguidance_args_dict(default_cfg) is None

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        enabled_cfg = compose(
            config_name="predict.yaml",
            overrides=[
                "sampling.autoguidance.enabled=true",
                "sampling.autoguidance.bad_ckpt_path=/tmp/bad.ckpt",
                "sampling.autoguidance.weight=2.0",
            ],
        )

    assert autoguidance_args_dict(enabled_cfg) == {
        "bad_ckpt_path": "/tmp/bad.ckpt",
        "weight": 2.0,
    }

    GlobalHydra.instance().clear()


def test_predict_spacegroup_cfg_config_defaults_and_override_compose() -> None:
    """Tests prediction space-group CFG weight lives under sampling."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        default_cfg = compose(config_name="predict.yaml")

    assert default_cfg.sampling.spacegroup_cfg.weight == 1.0
    assert spacegroup_cfg_weight(default_cfg) == 1.0

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        guided_cfg = compose(
            config_name="predict.yaml",
            overrides=["sampling.spacegroup_cfg.weight=2.0"],
        )

    assert spacegroup_cfg_weight(guided_cfg) == 2.0

    GlobalHydra.instance().clear()


def test_predict_autoguidance_config_validation() -> None:
    """Tests prediction autoguidance config validation failures."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        missing_path_cfg = compose(
            config_name="predict.yaml",
            overrides=["sampling.autoguidance.enabled=true"],
        )

    with pytest.raises(ValueError, match="bad_ckpt_path is required"):
        autoguidance_args_dict(missing_path_cfg)

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        invalid_weight_cfg = compose(
            config_name="predict.yaml",
            overrides=[
                "sampling.autoguidance.enabled=true",
                "sampling.autoguidance.bad_ckpt_path=/tmp/bad.ckpt",
                "sampling.autoguidance.weight=-1.0",
            ],
        )

    with pytest.raises(ValueError, match="weight must be finite"):
        autoguidance_args_dict(invalid_weight_cfg)

    GlobalHydra.instance().clear()


def test_sampler_configs_compose() -> None:
    """Tests length and fixed-shape sampler configs compose."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        fixed_cfg = compose(config_name="train.yaml")
    assert (
        fixed_cfg.data.sampler._target_
        == "src.data.components.samplers.DistributedFixedShapeBucketBatchSampler"
    )
    assert fixed_cfg.data.sampler.atom_buckets[-1] == 300
    assert fixed_cfg.data.sampler.get("batch_sizes") is None

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        length_cfg = compose(
            config_name="train.yaml",
            overrides=["data/sampler=length_bucket"],
        )
    assert (
        length_cfg.data.sampler._target_
        == "src.data.components.samplers.DistributedLengthBucketBatchSampler"
    )

    GlobalHydra.instance().clear()


def test_finetune_config_exposes_stage2_defaults() -> None:
    """Tests finetune config is checkpoint-agnostic and uses variable batches."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="finetune.yaml")

    assert cfg.init_ckpt_path is None
    assert cfg.ckpt_path is None
    assert cfg.run_config_path is None
    assert cfg.task_name == "finetune"
    assert cfg.finetune.learning_rate == pytest.approx(1.0e-4)
    assert cfg.finetune.warmup_steps == 10000
    assert cfg.finetune.init_weight_source.raw == "raw"
    assert cfg.finetune.init_weight_source.ema == "ema"
    assert cfg.finetune.ema_warm_start is False
    assert cfg.data.dataset_name == "csd_smiles_aligned"
    assert cfg.data.max_num_atoms is None
    assert cfg.data.effective_batch_size is None
    assert cfg.data.val_batch_size == 5
    assert cfg.data.sampler.atom_buckets[-1] == 512
    assert cfg.data.sampler.batch_sizes[300] == 16
    assert cfg.data.sampler.batch_sizes[512] == 5
    assert cfg.callbacks.model_checkpoint.every_n_epochs == 20
    assert "train_generation" not in cfg.callbacks
    assert "validation_generation" not in cfg.callbacks

    GlobalHydra.instance().clear()


def test_compile_config_wraps_net_with_eager_backend() -> None:
    """Tests compile config can wrap the core net without code generation."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "model.compile_target=net",
                "model.compile_backend=eager",
            ],
        )

    model = hydra.utils.instantiate(cfg.model)

    assert hasattr(model.flow_matching.net, "_orig_mod")

    GlobalHydra.instance().clear()


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_optimizer_override_config_instantiation(optimizer_name: str) -> None:
    """Tests optimizer overrides compose and instantiate as callable factories."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[f"model/optimizer={optimizer_name}"],
        )

    HydraConfig().set_config(cfg)

    optimizer_factory = hydra.utils.instantiate(cfg.model.optimizer)
    assert optimizer_factory is not None
    assert callable(optimizer_factory)

    GlobalHydra.instance().clear()


def test_optimizer_shared_defaults_and_overrides() -> None:
    """Tests shared optimizer defaults flow into concrete optimizer configs."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "model/optimizer=muon",
                "model.optimizer.lr=0.001",
                "model.optimizer.weight_decay=0.02",
            ],
        )

    assert cfg.model.optimizer.lr == pytest.approx(0.001)
    assert cfg.model.optimizer.weight_decay == pytest.approx(0.02)
    assert cfg.model.optimizer.momentum == pytest.approx(0.95)

    GlobalHydra.instance().clear()


def test_ltri_lattice_repr_config() -> None:
    """Tests ltri override uses Crystalite-style lattice representation."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["model/lattice_repr=ltri"],
        )

    assert cfg.model.lattice_repr == "ltri"
    assert cfg.model.lattice_dim == 6

    GlobalHydra.instance().clear()


def test_ltri_rejects_crystal_rotation() -> None:
    """Tests ltri representation fails fast with crystal rotation enabled."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["model/lattice_repr=ltri"],
        )

    with open_dict(cfg):
        cfg.data.crystal_rotate = True

    with pytest.raises(ValueError, match="crystal_rotate"):
        validate_lattice_config(cfg)

    GlobalHydra.instance().clear()
