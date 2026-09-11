# ruff: noqa: F722,F821

import torch
import torch.nn as nn
import pytest
from einops import repeat

from src.models.components import pairmixer as pairmixer_module
from src.models.components.embedders import (
    InputEmbedder,
    PairwiseEmbedder,
    SingleEmbedder,
)
from src.models.components.embedders.geometry import GeometryPairEmbedder
from src.models.components.pairmixer import (
    Pairmixer,
    TriangleSelfAttention,
    _pairmixer_dropout_mask,
    _pairmixer_oriented_dropout_mask,
)
from src.models.flow_model import MaterialFlowModel
from src.utils.tensor_typing import Float


class _ConditionSingle(nn.Module):
    """Return a stored single conditioning tensor."""

    def forward(self, noisy_batch: dict) -> Float["b n d"]:
        """Return condition singles from the batch."""
        return noisy_batch["conditioning"]["c"]


class _CoordEmbedder(nn.Module):
    """Return Cartesian coordinates as coordinate embeddings."""

    def forward(self, x: Float["b n 3"]) -> Float["b n 3"]:
        """Return coordinate tensor unchanged."""
        return x


class _LatticeEmbedder(nn.Module):
    """Broadcast lattice vectors to atom positions."""

    def forward(self, lattice: Float["b d"], num_atoms_dim: int) -> Float["b n d"]:
        """Broadcast lattice tensor across atom positions."""
        return repeat(lattice, "b d -> b n d", n=num_atoms_dim)


class _PairwiseCondition(nn.Module):
    """Return a stored pair conditioning tensor."""

    dim_pair = 2

    def forward(
        self,
        s: Float["b n d"],
        atom_mask: Float["b n"],
        conditioning: dict,
        conditioning_masks: dict,
    ) -> Float["b n n dz"]:
        """Return condition pair features from the batch."""
        return conditioning["z"]


class _GeometryPair(nn.Module):
    """Return a stored noisy geometry pair tensor."""

    def forward(self, noisy_batch: dict) -> Float["b n n dz"]:
        """Return geometry pair features from the batch."""
        return noisy_batch["geometry"]


class _TimeEmbedder(nn.Module):
    """Broadcast timesteps to the single feature dimension."""

    def forward(self, t: Float["b"]) -> Float["b d"]:
        """Return a simple timestep conditioning vector."""
        return repeat(t, "b -> b d", d=3)


class _CaptureAttn(nn.Module):
    """Expose the transformer pair-bias flag for model validation."""

    def __init__(self, use_pair_bias: bool) -> None:
        """Initialize the validation flag."""
        super().__init__()
        self.use_pair_bias = use_pair_bias


class _CaptureLayer(nn.Module):
    """Expose a minimal attention module for model validation."""

    def __init__(self, use_pair_bias: bool) -> None:
        """Initialize a minimal transformer layer stub."""
        super().__init__()
        self.attn = _CaptureAttn(use_pair_bias)


class _CaptureTransformer(nn.Module):
    """Capture the tensor inputs passed to the transformer."""

    def __init__(self, use_pair_bias: bool = True) -> None:
        """Initialize the capture transformer."""
        super().__init__()
        self.layers = nn.ModuleList([_CaptureLayer(use_pair_bias)])
        self.s: Float["b n d"] | None = None
        self.z: Float["b n n dz"] | None = None

    def forward(
        self,
        x: Float["b n d"],
        c: Float["b d"],
        mask: Float["b n"],
        z: Float["b n n dz"] | None = None,
    ) -> Float["b n d"]:
        """Store transformer inputs and return singles unchanged."""
        self.s = x.detach().clone()
        self.z = None if z is None else z.detach().clone()
        return x


class _Heads(nn.Module):
    """Return simple nonzero-shaped prediction tensors."""

    def forward(self, s: Float["b n d"], mask: Float["b n"]) -> dict[str, Float]:
        """Build placeholder coordinate and lattice predictions."""
        return {"coords": s[..., :3], "lattice": s.mean(dim=1)}


class _PairmixerRecorder(nn.Module):
    """Record Pairmixer inputs and add deterministic offsets."""

    def __init__(self) -> None:
        """Initialize captured Pairmixer tensors."""
        super().__init__()
        self.calls = 0
        self.s_in: Float["b n d"] | None = None
        self.z_in: Float["b n n dz"] | None = None

    def forward(
        self,
        s: Float["b n d"],
        z: Float["b n n dz"],
        atom_mask: Float["b n"],
    ) -> tuple[Float["b n d"], Float["b n n dz"]]:
        """Store Pairmixer inputs and return updated tensors."""
        self.calls += 1
        self.s_in = s.detach().clone()
        self.z_in = z.detach().clone()
        return s + 10.0, z + 20.0


def _conditioning(c: Float["b n d"]) -> dict[str, torch.Tensor]:
    """Build minimal conditioning fields needed by InputEmbedder."""
    batch_size = c.shape[0]
    return {
        "c": c,
        "template_mask": torch.ones(batch_size),
        "stereochemistry_mask": torch.ones(batch_size),
        "spacegroup_mask": torch.ones(batch_size),
    }


def _batch(
    c: Float["b n d"],
    z: Float["b n n dz"] | None = None,
    geometry: Float["b n n dz"] | None = None,
) -> dict:
    """Build a noisy model batch for embedder routing tests."""
    batch = {
        "x_t": torch.randn(c.shape[0], c.shape[1], c.shape[2]),
        "l_t": torch.randn(c.shape[0], c.shape[2]),
        "times": torch.rand(c.shape[0]),
        "indices": torch.arange(c.shape[1]).expand(c.shape[0], c.shape[1]),
        "atom_mask": torch.ones(c.shape[0], c.shape[1], dtype=torch.bool),
        "conditioning": _conditioning(c),
    }
    if z is not None:
        batch["conditioning"]["z"] = z
    if geometry is not None:
        batch["geometry"] = geometry
    return batch


def _input_embedder(
    use_pairwise: bool = False, use_geometry: bool = False
) -> InputEmbedder:
    """Build an input embedder with optional pair and geometry features."""
    return InputEmbedder(
        single_embedder=SingleEmbedder(
            single_condition_embedder=_ConditionSingle(),
            coord_embedder=_CoordEmbedder(),
            lattice_embedder=_LatticeEmbedder(),
        ),
        pairwise_embedder=(
            PairwiseEmbedder(
                pairwise_condition_embedder=_PairwiseCondition(),
                geometry_pair_embedder=_GeometryPair() if use_geometry else None,
            )
            if use_pairwise
            else None
        ),
    )


def _model(
    input_embedder: InputEmbedder,
    transformer: _CaptureTransformer,
    pairmixer: nn.Module | None = None,
    noisy_input_entry: str = "before_pairmixer",
) -> MaterialFlowModel:
    """Build a minimal MaterialFlowModel for routing tests."""
    return MaterialFlowModel(
        input_embedder=input_embedder,
        timestep_embedder=_TimeEmbedder(),
        transformer=transformer,
        heads=_Heads(),
        pairmixer=pairmixer,
        noisy_input_entry=noisy_input_entry,
    )


def test_input_embedder_matches_current_noisy_single_path() -> None:
    """InputEmbedder adds condition, coordinate, and lattice embeddings."""
    c = torch.randn(2, 4, 3)
    x_t = torch.randn(2, 4, 3)
    l_t = torch.randn(2, 3)
    input_embedder = InputEmbedder(
        single_embedder=SingleEmbedder(
            single_condition_embedder=_ConditionSingle(),
            coord_embedder=_CoordEmbedder(),
            lattice_embedder=_LatticeEmbedder(),
        ),
        pairwise_embedder=None,
    )

    s, z = input_embedder(
        {
            "x_t": x_t,
            "l_t": l_t,
            "indices": torch.arange(4).expand(2, 4),
            "atom_mask": torch.ones(2, 4, dtype=torch.bool),
            "conditioning": _conditioning(c),
        }
    )

    expected = c + x_t + repeat(l_t, "b d -> b n d", n=4)
    assert z is None
    assert torch.allclose(s, expected)


def test_after_entry_matches_before_entry_without_pairmixer() -> None:
    """After-entry and before-entry routes match when Pairmixer is absent."""
    c = torch.randn(2, 4, 3)
    z = torch.randn(2, 4, 4, 2)
    geometry = torch.randn(2, 4, 4, 2)
    batch = _batch(c, z, geometry)
    input_embedder = _input_embedder(use_pairwise=True, use_geometry=True)
    before_transformer = _CaptureTransformer()
    after_transformer = _CaptureTransformer()
    before_model = _model(
        input_embedder,
        before_transformer,
        noisy_input_entry="before_pairmixer",
    )
    after_model = _model(
        input_embedder,
        after_transformer,
        noisy_input_entry="after_pairmixer",
    )

    before_model(batch)
    after_model(batch)

    assert before_transformer.s is not None
    assert before_transformer.z is not None
    assert after_transformer.s is not None
    assert after_transformer.z is not None
    assert torch.allclose(after_transformer.s, before_transformer.s)
    assert torch.allclose(after_transformer.z, before_transformer.z)


def test_before_entry_matches_legacy_pairmixer_inputs() -> None:
    """Before-entry route preserves the legacy transformer input path."""
    torch.manual_seed(0)
    c = torch.randn(2, 4, 8)
    z = torch.randn(2, 4, 4, 4)
    geometry = torch.randn(2, 4, 4, 4)
    batch = _batch(c, z, geometry)
    input_embedder = InputEmbedder(
        single_embedder=SingleEmbedder(
            single_condition_embedder=_ConditionSingle(),
            coord_embedder=nn.Linear(8, 8, bias=False),
            lattice_embedder=_LatticeEmbedder(),
        ),
        pairwise_embedder=PairwiseEmbedder(
            pairwise_condition_embedder=_PairwiseCondition(),
            geometry_pair_embedder=_GeometryPair(),
        ),
    )
    input_embedder.pairwise_embedder.pairwise_condition_embedder.dim_pair = 4
    pairmixer = Pairmixer(
        dim_single=8,
        dim_pair=4,
        num_heads=2,
        num_blocks=1,
        update_single=True,
        dropout=0.0,
    )
    transformer = _CaptureTransformer()
    model = _model(input_embedder, transformer, pairmixer=pairmixer)
    model.eval()

    conditioned_batch = dict(batch)
    conditioned_batch["conditioning_masks"] = input_embedder._conditioning_masks(
        batch["conditioning"]
    )
    _, legacy_s = input_embedder.single_embedder(conditioned_batch)
    legacy_z = input_embedder.pairwise_embedder(conditioned_batch, c)
    expected_s, expected_z = pairmixer(
        legacy_s,
        legacy_z,
        conditioned_batch["atom_mask"],
    )

    model(batch)

    assert transformer.s is not None
    assert transformer.z is not None
    assert torch.allclose(transformer.s, expected_s)
    assert torch.allclose(transformer.z, expected_z)


def test_after_entry_injects_noisy_terms_after_pairmixer() -> None:
    """After-entry route keeps Pairmixer inputs time-independent."""
    c = torch.randn(2, 4, 3)
    z = torch.randn(2, 4, 4, 2)
    geometry = torch.randn(2, 4, 4, 2)
    batch = _batch(c, z, geometry)
    input_embedder = _input_embedder(use_pairwise=True, use_geometry=True)
    pairmixer = _PairmixerRecorder()
    transformer = _CaptureTransformer()
    model = _model(
        input_embedder,
        transformer,
        pairmixer=pairmixer,
        noisy_input_entry="after_pairmixer",
    )

    model(batch)

    expected_base_s = c
    expected_noisy_s = batch["x_t"] + repeat(batch["l_t"], "b d -> b n d", n=4)
    assert pairmixer.s_in is not None
    assert pairmixer.z_in is not None
    assert transformer.s is not None
    assert transformer.z is not None
    assert torch.allclose(pairmixer.s_in, expected_base_s)
    assert torch.allclose(pairmixer.z_in, z)
    assert torch.allclose(transformer.s, expected_base_s + 10.0 + expected_noisy_s)
    assert torch.allclose(transformer.z, z + 20.0 + geometry)


def test_before_entry_inference_cache_matches_uncached() -> None:
    """Before-entry cache reuses condition embeddings but still runs Pairmixer."""
    c = torch.randn(2, 4, 3)
    z = torch.randn(2, 4, 4, 2)
    geometry = torch.randn(2, 4, 4, 2)
    batch = _batch(c, z, geometry)
    input_embedder = _input_embedder(use_pairwise=True, use_geometry=True)
    pairmixer = _PairmixerRecorder()
    uncached_transformer = _CaptureTransformer()
    model = _model(
        input_embedder,
        uncached_transformer,
        pairmixer=pairmixer,
        noisy_input_entry="before_pairmixer",
    )

    uncached = model(batch)
    cache = model.build_inference_cache(batch)
    assert pairmixer.calls == 1

    cached_transformer = _CaptureTransformer()
    model.transformer = cached_transformer
    cached_batch = dict(batch)
    cached_batch["inference_cache"] = cache
    cached = model(cached_batch)

    assert pairmixer.calls == 2
    assert uncached_transformer.s is not None
    assert uncached_transformer.z is not None
    assert cached_transformer.s is not None
    assert cached_transformer.z is not None
    assert torch.allclose(cached["coords"], uncached["coords"])
    assert torch.allclose(cached["lattice"], uncached["lattice"])
    assert torch.allclose(cached_transformer.s, uncached_transformer.s)
    assert torch.allclose(cached_transformer.z, uncached_transformer.z)


def test_after_entry_inference_cache_matches_uncached_and_skips_pairmixer() -> None:
    """After-entry cache reuses the Pairmixer output across denoising calls."""
    c = torch.randn(2, 4, 3)
    z = torch.randn(2, 4, 4, 2)
    geometry = torch.randn(2, 4, 4, 2)
    batch = _batch(c, z, geometry)
    input_embedder = _input_embedder(use_pairwise=True, use_geometry=True)
    pairmixer = _PairmixerRecorder()
    uncached_transformer = _CaptureTransformer()
    model = _model(
        input_embedder,
        uncached_transformer,
        pairmixer=pairmixer,
        noisy_input_entry="after_pairmixer",
    )

    uncached = model(batch)
    cache = model.build_inference_cache(batch)
    assert pairmixer.calls == 2

    cached_transformer = _CaptureTransformer()
    model.transformer = cached_transformer
    cached_batch = dict(batch)
    cached_batch["inference_cache"] = cache
    cached = model(cached_batch)

    assert pairmixer.calls == 2
    assert uncached_transformer.s is not None
    assert uncached_transformer.z is not None
    assert cached_transformer.s is not None
    assert cached_transformer.z is not None
    assert torch.allclose(cached["coords"], uncached["coords"])
    assert torch.allclose(cached["lattice"], uncached["lattice"])
    assert torch.allclose(cached_transformer.s, uncached_transformer.s)
    assert torch.allclose(cached_transformer.z, uncached_transformer.z)


def test_geometry_pair_embedder_minimum_image_and_masks() -> None:
    """GeometryPairEmbedder computes wrapped distances and masks invalid pairs."""
    embedder = GeometryPairEmbedder(
        dim_pair=4,
        n_fourier_freqs=2,
        n_rbf=4,
        hidden_dim=8,
        pbc_radius=1,
    )
    coords = torch.tensor([[[0.9, 0.0, 0.0], [0.1, 0.0, 0.0], [0.4, 0.0, 0.0]]])
    cell = torch.eye(3).unsqueeze(0)
    min_delta, min_dist_norm, _ = embedder._minimum_image_features(coords, cell)

    assert torch.allclose(min_dist_norm[0, 0, 1], torch.tensor(0.2), atol=1e-6)
    assert torch.allclose(min_delta[0, 0, 1].abs(), torch.tensor([0.2, 0.0, 0.0]))

    out = embedder(
        {
            "x_t_physical": coords,
            "cell_t_physical": cell,
            "times": torch.tensor([0.7]),
            "flow_times": torch.tensor([0.7]),
            "atom_mask": torch.tensor([[True, True, False]]),
        }
    )

    assert out.shape == (1, 3, 3, 4)
    assert torch.allclose(
        out[:, torch.arange(3), torch.arange(3)], torch.zeros(1, 3, 4)
    )
    assert torch.allclose(out[:, 2], torch.zeros(1, 3, 4))
    assert torch.allclose(out[:, :, 2], torch.zeros(1, 3, 4))


def test_geometry_pair_embedder_backward_has_finite_parameter_grads() -> None:
    """GeometryPairEmbedder supports backpropagation through learnable parameters."""
    embedder = GeometryPairEmbedder(
        dim_pair=4,
        n_fourier_freqs=2,
        n_rbf=4,
        hidden_dim=8,
        pbc_radius=1,
    )
    coords = torch.randn(1, 3, 3, requires_grad=True)
    cell = torch.eye(3).unsqueeze(0).requires_grad_()
    out = embedder(
        {
            "x_t_physical": coords,
            "cell_t_physical": cell,
            "times": torch.tensor([0.7]),
            "flow_times": torch.tensor([0.7]),
            "atom_mask": torch.ones(1, 3, dtype=torch.bool),
        }
    )

    out.square().sum().backward()

    grads = [parameter.grad for parameter in embedder.parameters()]
    assert all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads if grad is not None)
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()


def test_pairmixer_forward_backward_shapes() -> None:
    """Pairmixer preserves representation shapes and supports backward."""
    pairmixer = Pairmixer(
        dim_single=8,
        dim_pair=4,
        num_heads=2,
        num_blocks=1,
        update_single=True,
        dropout=0.0,
    )
    s = torch.randn(2, 3, 8, requires_grad=True)
    z = torch.randn(2, 3, 3, 4, requires_grad=True)
    atom_mask = torch.tensor([[True, True, True], [True, True, False]])

    s_out, z_out = pairmixer(s, z, atom_mask)
    loss = s_out.square().mean() + z_out.square().mean()
    loss.backward()

    assert s_out.shape == s.shape
    assert z_out.shape == z.shape
    assert s.grad is not None
    assert z.grad is not None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in pairmixer.parameters()
    )


def test_pairmixer_triangle_attention_forward_backward_shapes() -> None:
    """Pairmixer triangle attention preserves shapes and supports backward."""
    pairmixer = Pairmixer(
        dim_single=8,
        dim_pair=4,
        num_heads=2,
        num_blocks=1,
        update_single=True,
        dropout=0.0,
        use_triangle_attention=True,
        triangle_attention_backend="pytorch",
        triangle_attention_num_heads=2,
        triangle_attention_head_dim=2,
        triangle_attention_chunk_size=2,
    )
    s = torch.randn(2, 5, 8, requires_grad=True)
    z = torch.randn(2, 5, 5, 4, requires_grad=True)
    atom_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )

    with torch.no_grad():
        layer = pairmixer.layers[0]
        layer.tri_att_start.proj_o.weight.normal_(std=0.02)
        layer.tri_att_end.proj_o.weight.normal_(std=0.02)

    s_out, z_out = pairmixer(s, z, atom_mask)
    loss = s_out.square().mean() + z_out.square().mean()
    loss.backward()

    layer = pairmixer.layers[0]
    assert s_out.shape == s.shape
    assert z_out.shape == z.shape
    assert s.grad is not None
    assert z.grad is not None
    assert layer.tri_att_start.proj_o.weight.grad is not None
    assert torch.isfinite(layer.tri_att_start.proj_o.weight.grad).all()
    assert layer.tri_att_end.proj_o.weight.grad is not None
    assert torch.isfinite(layer.tri_att_end.proj_o.weight.grad).all()


def test_triangle_self_attention_chunking_matches_unchunked_pytorch() -> None:
    """Chunked triangle attention matches the unchunked PyTorch path."""
    torch.manual_seed(0)
    full = TriangleSelfAttention(
        dim_pair=4,
        num_heads=2,
        head_dim=3,
        backend="pytorch",
        chunk_size=None,
    )
    with torch.no_grad():
        full.proj_o.weight.normal_(std=0.02)
    chunked = TriangleSelfAttention(
        dim_pair=4,
        num_heads=2,
        head_dim=3,
        backend="pytorch",
        chunk_size=2,
    )
    chunked.load_state_dict(full.state_dict())
    x = torch.randn(2, 5, 5, 4)
    atom_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    pair_mask = repeat(atom_mask, "b i -> b i j", j=5) & repeat(
        atom_mask, "b j -> b i j", i=5
    )

    full_out = full(x, pair_mask)
    chunked_out = chunked(x, pair_mask)

    assert torch.allclose(chunked_out, full_out, atol=1e-5, rtol=1e-5)


def test_triangle_self_attention_deepspeed_fallback_warns(monkeypatch) -> None:
    """DeepSpeed backend warns once and falls back when unavailable."""
    monkeypatch.setattr(
        pairmixer_module,
        "_deepspeed_triangle_attention_warned",
        False,
    )

    def _fake_deepspeed_attention(*args, **kwargs):
        pairmixer_module._warn_deepspeed_triangle_attention("forced test fallback")
        return None

    monkeypatch.setattr(
        pairmixer_module,
        "_deepspeed_triangle_attention",
        _fake_deepspeed_attention,
    )
    attention = TriangleSelfAttention(
        dim_pair=4,
        num_heads=2,
        head_dim=2,
        backend="deepspeed",
    )
    x = torch.randn(1, 4, 4, 4)
    mask = torch.ones(1, 4, 4, dtype=torch.bool)

    with pytest.warns(UserWarning, match="DeepSpeed4Science EvoformerAttention"):
        out = attention(x, mask)

    assert out.shape == x.shape
    assert not attention.used_deepspeed


def test_pairmixer_dropout_mask_is_structured() -> None:
    """Pairmixer dropout samples rowwise masks broadcast over pair channels."""
    z = torch.ones(3, 5, 4, 7)

    torch.manual_seed(0)
    mask = _pairmixer_dropout_mask(dropout=0.5, z=z, training=True)

    assert mask.shape == (3, 5, 1, 1)
    assert mask.dtype == z.dtype
    assert torch.all((mask == 0.0) | (mask == 2.0))


def test_pairmixer_dropout_mask_can_be_columnwise() -> None:
    """Pairmixer dropout can sample columnwise masks for ending attention."""
    z = torch.ones(3, 5, 4, 7)

    torch.manual_seed(0)
    mask = _pairmixer_oriented_dropout_mask(
        dropout=0.5,
        z=z,
        training=True,
        columnwise=True,
    )

    assert mask.shape == (3, 1, 4, 1)
    assert mask.dtype == z.dtype
    assert torch.all((mask == 0.0) | (mask == 2.0))


def test_pairmixer_dropout_mask_is_disabled_in_eval() -> None:
    """Pairmixer dropout returns an all-ones mask when disabled."""
    z = torch.ones(3, 5, 4, 7)

    train_mask = _pairmixer_dropout_mask(dropout=0.0, z=z, training=True)
    eval_mask = _pairmixer_dropout_mask(dropout=0.5, z=z, training=False)

    assert train_mask.shape == (3, 5, 1, 1)
    assert eval_mask.shape == (3, 5, 1, 1)
    assert torch.equal(train_mask, torch.ones_like(train_mask))
    assert torch.equal(eval_mask, torch.ones_like(eval_mask))
