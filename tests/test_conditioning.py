# ruff: noqa: F722,F821

from typing import Any

import pytest
import torch
from einops import rearrange

from src.data.components.conditioning import (
    ATOM_CHIRALITY_MODEL_NULL_ID,
    BOND_STEREOCHEMISTRY_MODEL_NULL_ID,
    BOND_TYPE_MODEL_NO_BOND_ID,
    SPACEGROUP_MODEL_NULL_ID,
    ConditioningPolicy,
    atom_chirality_model_ids,
    bond_stereochemistry_model_ids,
    bond_type_model_ids,
    prepare_model_conditioning,
    spacegroup_model_ids,
)
from src.models.components.embedders.condition import ConditionEmbedder
from src.models.components.embedders.condition.pairwise import PairwiseConditionEmbedder
from src.models.components.embedders.condition.single import SingleConditionEmbedder
from src.utils.tensor_typing import Bool, Float


def _conditioning_masks(
    template: float,
    stereochemistry: float,
    spacegroup: float,
) -> dict[str, torch.Tensor]:
    """Build a one-sample conditioning mask dictionary."""
    return {
        "template": torch.tensor([template]),
        "stereochemistry": torch.tensor([stereochemistry]),
        "spacegroup": torch.tensor([spacegroup]),
    }


def _conditioning_presence(
    template: list[bool],
    stereochemistry: list[bool],
    spacegroup: list[bool],
) -> dict[str, torch.Tensor]:
    """Build conditioning presence tensors for policy tests."""
    return {
        "template_present": torch.tensor(template),
        "stereochemistry_present": torch.tensor(stereochemistry),
        "spacegroup_present": torch.tensor(spacegroup),
    }


def _single_noisy_batch(template_coords: torch.Tensor) -> dict:
    """Build a minimal noisy batch for single conditioning tests."""
    return {
        "atom_mask": torch.tensor([[True, True, False]]),
        "conditioning_masks": _conditioning_masks(0.0, 0.0, 0.0),
        "conditioning": {
            "atomic_numbers": torch.tensor([[6, 8, 0]]),
            "template_coords": template_coords,
            "formal_charges": torch.tensor([[0, -1, 0]]),
            "atom_chirality": torch.tensor(
                [[ATOM_CHIRALITY_MODEL_NULL_ID, ATOM_CHIRALITY_MODEL_NULL_ID, 0]]
            ),
            "spacegroup_number": torch.tensor([SPACEGROUP_MODEL_NULL_ID]),
        },
    }


def _zero_pairwise_branches(embedder: PairwiseConditionEmbedder) -> None:
    """Zero pairwise branches that are irrelevant to an isolated assertion."""
    with torch.no_grad():
        embedder.embed_template_d.weight.zero_()
        embedder.embed_template_dist.weight.zero_()
        embedder.embed_bond.weight.zero_()
        embedder.embed_stereochemistry.weight.zero_()
        embedder.proj_q.weight.zero_()
        embedder.proj_k.weight.zero_()
        for layer in embedder.mlp:
            if hasattr(layer, "weight"):
                layer.weight.zero_()
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.zero_()


def _pair_conditioning() -> dict[str, torch.Tensor]:
    """Build a minimal prepared pair conditioning dictionary."""
    return {
        "membership": torch.tensor([[0, 0, 1, -1]]),
        "template_coords": torch.zeros(1, 4, 3),
        "bond_adj": torch.tensor(
            [
                [
                    [False, True, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                ]
            ]
        ),
        "bond_type_adj": torch.tensor(
            [
                [
                    [BOND_TYPE_MODEL_NO_BOND_ID, 3, 0, 0],
                    [BOND_TYPE_MODEL_NO_BOND_ID, BOND_TYPE_MODEL_NO_BOND_ID, 0, 0],
                    [0, 0, BOND_TYPE_MODEL_NO_BOND_ID, 0],
                    [0, 0, 0, 0],
                ]
            ]
        ),
        "bond_stereochemistry_adj": torch.tensor(
            [
                [
                    [0, BOND_STEREOCHEMISTRY_MODEL_NULL_ID, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ]
        ),
    }


def _reference_pairwise_forward(
    embedder: PairwiseConditionEmbedder,
    s: Float["b n d"],
    atom_mask: Bool["b n"],
    conditioning: dict[str, Any],
    conditioning_masks: dict[str, Float["b"]],
) -> Float["b n n dp"]:
    """Compute the original pairwise forward for equivalence tests."""
    membership = conditioning["membership"]
    same_group = rearrange(membership, "b i -> b i 1") == rearrange(
        membership, "b j -> b 1 j"
    )
    real_pair = rearrange(atom_mask, "b i -> b i 1") & rearrange(
        atom_mask, "b j -> b 1 j"
    )
    valid_mol_pair = same_group & real_pair
    pair_gate = rearrange(valid_mol_pair.to(s.dtype), "b i j -> b i j 1")

    t = conditioning["template_coords"]
    d = rearrange(t, "b i c -> b i 1 c") - rearrange(t, "b j c -> b 1 j c")
    d_norm = 1.0 / (1.0 + d.square().sum(-1, keepdim=True))
    template = embedder.embed_template_d(d) + embedder.embed_template_dist(d_norm)
    template_gate = rearrange(conditioning_masks["template"], "b -> b 1 1 1")
    template = template * pair_gate * template_gate

    bond_ids = bond_type_model_ids(conditioning["bond_type_adj"], valid_mol_pair)
    bond = embedder.embed_bond(bond_ids) * pair_gate

    bond_adj = conditioning.get("bond_adj", conditioning["bond_type_adj"] > 0)
    valid_bond_pair = bond_adj & valid_mol_pair
    stereochemistry_ids = bond_stereochemistry_model_ids(
        conditioning["bond_stereochemistry_adj"],
        valid_bond_pair,
        conditioning_masks["stereochemistry"],
    )
    stereochemistry_gate = rearrange(
        valid_bond_pair.to(s.dtype),
        "b i j -> b i j 1",
    )
    stereochemistry = (
        embedder.embed_stereochemistry(stereochemistry_ids) * stereochemistry_gate
    )
    z = template + bond + stereochemistry

    q = embedder.proj_q(s)
    k = embedder.proj_k(s)
    z = z + rearrange(q, "b i d -> b i 1 d") + rearrange(k, "b j d -> b 1 j d")
    return z + embedder.mlp(z)


def _reference_single_forward(
    embedder: SingleConditionEmbedder,
    noisy_batch: dict[str, Any],
) -> Float["b n d"]:
    """Compute the original single forward with inline model-ID remapping."""
    conditioning = noisy_batch["conditioning"]
    conditioning_masks = noisy_batch["conditioning_masks"]
    atom_types = conditioning["atomic_numbers"]
    atom_mask = noisy_batch["atom_mask"]

    c = embedder.embed_atom(atom_types)
    c = c + embedder.embed_period(embedder.period_lookup[atom_types])
    c = c + embedder.embed_group(embedder.group_lookup[atom_types])
    c = c + embedder.embed_block(embedder.block_lookup[atom_types])

    covalent_radius = rearrange(
        embedder.covalent_radius_lookup[atom_types], "b n -> b n 1"
    )
    vdw_radius = rearrange(embedder.vdw_radius_lookup[atom_types], "b n -> b n 1")
    electronegativity = rearrange(
        embedder.electronegativity_lookup[atom_types], "b n -> b n 1"
    )
    c = c + embedder.embed_covalent_radius(covalent_radius)
    c = c + embedder.embed_vdw_radius(vdw_radius)
    c = c + embedder.embed_electronegativity(electronegativity)

    charges_scalar = rearrange(conditioning["formal_charges"].float(), "b n -> b n 1")
    c = c + embedder.embed_charges(charges_scalar)

    template_gate = rearrange(conditioning_masks["template"], "b -> b 1 1")
    template = embedder.embed_template(conditioning["template_coords"])
    c = c + template * template_gate

    chirality_ids = atom_chirality_model_ids(
        conditioning["atom_chirality"],
        atom_mask,
        conditioning_masks["stereochemistry"],
    )
    c = c + embedder.embed_chirality(chirality_ids)

    spacegroup_ids = spacegroup_model_ids(
        conditioning["spacegroup_number"],
        conditioning_masks["spacegroup"],
    )
    spacegroup = rearrange(embedder.embed_spacegroup(spacegroup_ids), "b d -> b 1 d")
    c = c + spacegroup * rearrange(atom_mask.to(c.dtype), "b n -> b n 1")
    return c


def test_raw_to_model_id_conversion() -> None:
    """Converts raw CSD labels into padding/null-aware model IDs."""
    atom_raw = torch.tensor([[0, 1, 4, 0]])
    atom_mask = torch.tensor([[True, True, True, False]])
    stereo_on = torch.tensor([1.0])
    stereo_off = torch.tensor([0.0])
    assert atom_chirality_model_ids(atom_raw, atom_mask, stereo_on).tolist() == [
        [1, 2, 5, 0]
    ]
    assert atom_chirality_model_ids(atom_raw, atom_mask, stereo_off).tolist() == [
        [ATOM_CHIRALITY_MODEL_NULL_ID] * 3 + [0]
    ]

    valid_pair = torch.tensor([[[True, True], [False, True]]])
    bond_raw = torch.tensor([[[0, 2], [0, 1]]])
    assert bond_type_model_ids(bond_raw, valid_pair).tolist() == [
        [[BOND_TYPE_MODEL_NO_BOND_ID, 3], [0, 2]]
    ]

    valid_bond = torch.tensor([[[False, True], [False, False]]])
    stereo_raw = torch.tensor([[[0, 2], [0, 0]]])
    assert bond_stereochemistry_model_ids(
        stereo_raw,
        valid_bond,
        stereo_on,
    ).tolist() == [[[0, 3], [0, 0]]]
    assert bond_stereochemistry_model_ids(
        stereo_raw,
        valid_bond,
        stereo_off,
    ).tolist() == [[[0, BOND_STEREOCHEMISTRY_MODEL_NULL_ID], [0, 0]]]

    spacegroup_raw = torch.tensor([0, 1, 230, 999])
    assert spacegroup_model_ids(
        spacegroup_raw,
        torch.ones(4),
    ).tolist() == [0, 1, 230, 0]
    assert (
        spacegroup_model_ids(
            spacegroup_raw,
            torch.zeros(4),
        ).tolist()
        == [SPACEGROUP_MODEL_NULL_ID] * 4
    )


def test_conditioning_policy_masks_on_off_and_stochastic() -> None:
    """Resolves deterministic and stochastic masks from a context policy."""
    policy = ConditioningPolicy.from_config(
        {
            "dropout_probs": {
                "template": 0.0,
                "stereochemistry": 1.0,
                "spacegroup": 1.0,
            },
            "contexts": {
                "unit": {
                    "template": "on",
                    "stereochemistry": "off",
                    "spacegroup": "stochastic",
                },
                "unit_keep": {
                    "template": "stochastic",
                    "stereochemistry": "stochastic",
                    "spacegroup": "stochastic",
                },
            },
        }
    )
    masks = policy.masks_from_presence(
        _conditioning_presence(
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ),
        context="unit",
    )
    assert masks["template"].tolist() == [1.0, 1.0, 1.0]
    assert masks["stereochemistry"].tolist() == [0.0, 0.0, 0.0]
    assert masks["spacegroup"].tolist() == [0.0, 0.0, 0.0]

    keep_masks = policy.masks_from_presence(
        _conditioning_presence(
            [True, True],
            [True, True],
            [True, True],
        ),
        context="unit_keep",
    )
    assert keep_masks["template"].tolist() == [1.0, 1.0]
    assert keep_masks["stereochemistry"].tolist() == [0.0, 0.0]
    assert keep_masks["spacegroup"].tolist() == [0.0, 0.0]

    with pytest.raises(ValueError, match="Unknown conditioning context"):
        policy.masks_from_presence(
            _conditioning_presence([True], [True], [True]),
            context="missing",
        )


def test_conditioning_policy_presence_flags_gate_masks() -> None:
    """Presence flags suppress optional conditioning masks per sample."""
    policy = ConditioningPolicy.from_config(
        {
            "contexts": {
                "unit": {
                    "template": "on",
                    "stereochemistry": "on",
                    "spacegroup": "on",
                },
            },
        }
    )
    masks = policy.masks_from_presence(
        _conditioning_presence(
            [True, False],
            [False, True],
            [True, False],
        ),
        context="unit",
    )

    assert masks["template"].tolist() == [1.0, 0.0]
    assert masks["stereochemistry"].tolist() == [0.0, 1.0]
    assert masks["spacegroup"].tolist() == [1.0, 0.0]


def test_single_template_drop_zeroes_projected_contribution() -> None:
    """Dropping template conditioning removes its projected single-token contribution."""
    embedder = SingleConditionEmbedder(dim=8)
    with torch.no_grad():
        embedder.embed_template.weight.fill_(2.0)

    nonzero_template = torch.tensor(
        [[[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [0.0, 0.0, 0.0]]]
    )
    dropped = embedder(_single_noisy_batch(nonzero_template))
    zeroed = embedder(_single_noisy_batch(torch.zeros_like(nonzero_template)))
    assert torch.allclose(dropped, zeroed)


def test_pairwise_bond_type_masks_invalid_pairs_and_keeps_no_bond() -> None:
    """Bond type features are learned for valid no-bond pairs and zero elsewhere."""
    embedder = PairwiseConditionEmbedder(dim_single=3, dim_pair=3)
    _zero_pairwise_branches(embedder)
    with torch.no_grad():
        embedder.embed_bond.weight[BOND_TYPE_MODEL_NO_BOND_ID] = torch.tensor(
            [1.0, 2.0, 3.0]
        )

    output = embedder(
        s=torch.zeros(1, 4, 3),
        atom_mask=torch.tensor([[True, True, True, False]]),
        conditioning=_pair_conditioning(),
        conditioning_masks=_conditioning_masks(0.0, 0.0, 0.0),
    )
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(output[0, 1, 0], expected)
    assert torch.allclose(output[0, 0, 2], torch.zeros(3))
    assert torch.allclose(output[0, 0, 3], torch.zeros(3))


def test_pairwise_stereo_null_only_on_valid_bonds_when_dropped() -> None:
    """Dropped stereochemistry contributes NULL only on valid bonded pairs."""
    embedder = PairwiseConditionEmbedder(dim_single=3, dim_pair=3)
    _zero_pairwise_branches(embedder)
    with torch.no_grad():
        embedder.embed_stereochemistry.weight[BOND_STEREOCHEMISTRY_MODEL_NULL_ID] = (
            torch.tensor([4.0, 5.0, 6.0])
        )

    output = embedder(
        s=torch.zeros(1, 4, 3),
        atom_mask=torch.tensor([[True, True, True, False]]),
        conditioning=_pair_conditioning(),
        conditioning_masks=_conditioning_masks(0.0, 0.0, 0.0),
    )
    assert torch.allclose(output[0, 0, 1], torch.tensor([4.0, 5.0, 6.0]))
    assert torch.allclose(output[0, 1, 0], torch.zeros(3))
    assert torch.allclose(output[0, 0, 2], torch.zeros(3))


def test_pairwise_forward_matches_reference_with_random_weights() -> None:
    """Optimized pairwise forward matches the original random-weight computation."""
    torch.manual_seed(7)
    embedder = PairwiseConditionEmbedder(dim_single=4, dim_pair=4)
    assert torch.count_nonzero(embedder.embed_bond.weight[1:]).item() > 0
    assert torch.count_nonzero(embedder.embed_stereochemistry.weight[1:]).item() > 0
    assert torch.all(embedder.embed_bond.weight[0] == 0)
    assert torch.all(embedder.embed_stereochemistry.weight[0] == 0)

    s = torch.randn(2, 5, 4)
    atom_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    conditioning = {
        "membership": torch.tensor(
            [
                [0, 0, 1, -1, -1],
                [0, 1, 1, 1, -1],
            ]
        ),
        "template_coords": torch.randn(2, 5, 3),
        "bond_adj": torch.tensor(
            [
                [
                    [False, True, False, False, False],
                    [True, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ],
                [
                    [False, False, False, False, False],
                    [False, False, True, True, False],
                    [False, True, False, False, False],
                    [False, True, False, False, False],
                    [False, False, False, False, False],
                ],
            ]
        ),
        "bond_type_adj": torch.tensor(
            [
                [
                    [0, 2, 0, 1, 0],
                    [2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 2, 0],
                    [0, 1, 0, 0, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            ]
        ),
        "bond_stereochemistry_adj": torch.tensor(
            [
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 2, 1, 0],
                    [0, 2, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            ]
        ),
    }
    conditioning_masks = {
        "template": torch.tensor([1.0, 0.0]),
        "stereochemistry": torch.tensor([1.0, 0.0]),
        "spacegroup": torch.tensor([1.0, 1.0]),
    }
    prepared_input = {
        **conditioning,
        "atomic_numbers": torch.ones(2, 5, dtype=torch.long),
        "template_present": torch.tensor([True, False]),
        "formal_charges": torch.zeros(2, 5, dtype=torch.long),
        "atom_chirality": torch.zeros(2, 5, dtype=torch.long),
        "stereochemistry_present": torch.tensor([True, False]),
        "spacegroup_number": torch.tensor([1, 1]),
        "spacegroup_present": torch.tensor([True, True]),
    }
    policy = ConditioningPolicy.from_config(
        {
            "contexts": {
                "unit": {
                    "template": "on",
                    "stereochemistry": "on",
                    "spacegroup": "on",
                },
            },
        }
    )
    prepared = prepare_model_conditioning(
        prepared_input,
        atom_mask,
        policy,
        context="unit",
    )

    expected = _reference_pairwise_forward(
        embedder,
        s,
        atom_mask,
        conditioning,
        conditioning_masks,
    )
    actual = embedder(
        s=s,
        atom_mask=atom_mask,
        conditioning=prepared,
        conditioning_masks={
            "template": prepared["template_mask"],
            "stereochemistry": prepared["stereochemistry_mask"],
            "spacegroup": prepared["spacegroup_mask"],
        },
    )
    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_condition_embedder_matches_reference_with_prepared_batch() -> None:
    """ConditionEmbedder output matches the old inline remapping path."""
    torch.manual_seed(13)
    single_embedder = SingleConditionEmbedder(dim=4)
    pairwise_embedder = PairwiseConditionEmbedder(dim_single=4, dim_pair=4)
    condition_embedder = ConditionEmbedder(
        single_embedder=single_embedder,
        pairwise_embedder=pairwise_embedder,
    )

    atom_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )
    conditioning = {
        "atomic_numbers": torch.tensor(
            [
                [6, 8, 1, 0, 0],
                [6, 7, 8, 1, 0],
            ]
        ),
        "template_coords": torch.randn(2, 5, 3),
        "template_present": torch.tensor([True, False]),
        "formal_charges": torch.tensor(
            [
                [0, -1, 0, 0, 0],
                [0, 1, -1, 0, 0],
            ]
        ),
        "atom_chirality": torch.tensor(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 2, 0, 0],
            ]
        ),
        "stereochemistry_present": torch.tensor([True, False]),
        "membership": torch.tensor(
            [
                [0, 0, 1, -1, -1],
                [0, 1, 1, 1, -1],
            ]
        ),
        "bond_adj": torch.tensor(
            [
                [
                    [False, True, False, False, False],
                    [True, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ],
                [
                    [False, False, False, False, False],
                    [False, False, True, True, False],
                    [False, True, False, False, False],
                    [False, True, False, False, False],
                    [False, False, False, False, False],
                ],
            ]
        ),
        "bond_type_adj": torch.tensor(
            [
                [
                    [0, 2, 0, 1, 0],
                    [2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 2, 0],
                    [0, 1, 0, 0, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            ]
        ),
        "bond_stereochemistry_adj": torch.tensor(
            [
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 2, 1, 0],
                    [0, 2, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            ]
        ),
        "spacegroup_number": torch.tensor([1, 999]),
        "spacegroup_present": torch.tensor([True, True]),
    }
    conditioning_masks = {
        "template": torch.tensor([1.0, 0.0]),
        "stereochemistry": torch.tensor([1.0, 0.0]),
        "spacegroup": torch.tensor([1.0, 1.0]),
    }
    raw_batch = {
        "atom_mask": atom_mask,
        "conditioning": conditioning,
        "conditioning_masks": conditioning_masks,
    }
    policy = ConditioningPolicy.from_config(
        {
            "contexts": {
                "unit": {
                    "template": "on",
                    "stereochemistry": "on",
                    "spacegroup": "on",
                },
            },
        }
    )
    prepared_conditioning = prepare_model_conditioning(
        conditioning,
        atom_mask,
        policy,
        context="unit",
    )

    expected_c = _reference_single_forward(single_embedder, raw_batch)
    expected_z = _reference_pairwise_forward(
        pairwise_embedder,
        expected_c,
        atom_mask,
        conditioning,
        conditioning_masks,
    )
    actual_c, actual_z = condition_embedder(
        {
            "atom_mask": atom_mask,
            "conditioning": prepared_conditioning,
        }
    )
    assert actual_z is not None
    assert torch.allclose(actual_c, expected_c, atol=0.0, rtol=0.0)
    assert torch.allclose(actual_z, expected_z, atol=0.0, rtol=0.0)
