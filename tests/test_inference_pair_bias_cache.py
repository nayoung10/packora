"""Check pair-bias reuse, dynamic pair features, and training gradients."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from src.models.components.transformers import DiT
from tests.test_input_geometry_pairmixer import (
    _batch,
    _input_embedder,
    _model,
    _PairmixerRecorder,
)


@pytest.mark.parametrize("entry", ["before_pairmixer", "after_pairmixer"])
@pytest.mark.parametrize("geometry", [False, True])
@pytest.mark.parametrize("shared_norm", [False, True])
def test_sampling_cache_matches_changing_noisy_inputs(
    entry: str, geometry: bool, shared_norm: bool
) -> None:
    """Reuse biases only for static pairs and preserve padded batched predictions."""
    torch.manual_seed(42)
    transformer = DiT(
        dim=3, depth=2, heads=1, dim_pair=2, share_pair_bias_norm=shared_norm
    )
    # Nonzero modulation makes attention errors visible in final outputs
    for parameter in transformer.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    model = _model(
        _input_embedder(use_pairwise=True, use_geometry=geometry),
        transformer,
        pairmixer=_PairmixerRecorder(),
        noisy_input_entry=entry,
    ).eval()
    batch = _batch(
        torch.randn(2, 4, 3), torch.randn(2, 4, 4, 2), torch.randn(2, 4, 4, 2)
    )
    batch["atom_mask"][1, -1] = False
    with torch.inference_mode():
        cache = model.build_inference_cache(batch)
        should_cache = entry == "after_pairmixer" and not geometry and shared_norm
        assert ("projected_pair_biases" in cache) == should_cache
        for _ in range(2):
            batch["x_t"] = torch.randn_like(batch["x_t"])
            batch["l_t"] = torch.randn_like(batch["l_t"])
            batch["times"] = torch.rand(2)
            batch["geometry"] = torch.randn_like(batch["geometry"])
            reference = model(batch)
            with patch.object(
                transformer,
                "_project_pair_biases",
                wraps=transformer._project_pair_biases,
            ) as project:
                cached = model({**batch, "inference_cache": cache})
                assert project.call_count == int(shared_norm and not should_cache)
            for key in reference:
                torch.testing.assert_close(cached[key], reference[key], rtol=0, atol=0)


@pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm"])
def test_shared_projection_preserves_block_affines_and_gradients(
    norm_type: str,
) -> None:
    """Shared projection agrees with individual block norms in training."""
    torch.manual_seed(17)
    shared = DiT(
        dim=8,
        depth=2,
        heads=2,
        dim_pair=4,
        share_pair_bias_norm=True,
        norm_type=norm_type,
    )
    for parameter in shared.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    individual = deepcopy(shared)
    individual.share_pair_bias_norm = False
    x, c, z = torch.randn(2, 3, 8), torch.randn(2, 8), torch.randn(2, 3, 3, 4)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    result = shared(x, c, mask, z)
    reference = individual(x, c, mask, z)
    torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)
    result.square().sum().backward()
    reference.square().sum().backward()
    for actual, expected in zip(
        shared.parameters(), individual.parameters(), strict=True
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-6)
