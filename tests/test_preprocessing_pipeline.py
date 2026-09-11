import numpy as np
import pytest
import torch
from types import SimpleNamespace

from src.data.components.collate import collate_fn
from src.data.components.conditioning import (
    ATOM_CHIRALITY_MODEL_NULL_ID,
    BOND_STEREOCHEMISTRY_MODEL_NULL_ID,
    BOND_TYPE_MODEL_NO_BOND_ID,
    SPACEGROUP_MODEL_NULL_ID,
)
from src.data.preprocess.csd import (
    CSDFilterConfig,
    CSDPreprocessor,
    _conditioning_spacegroup,
)
from src.data.preprocess.utils.config import (
    DeduplicateConfig,
    StructureMatcherConfig,
)
from src.data.preprocess.utils.deduplication import deduplicate_same_family
from src.data.types import Material, MoleculeConditioning


def _material(
    refcode: str | None,
    r_factor: object = 1.0,
    offset: float = 0.0,
) -> Material:
    """Build a minimal material for preprocessing pipeline tests."""
    info = {"r_factor": r_factor}
    if refcode is not None:
        info["csd_refcode"] = refcode
    conditioning = MoleculeConditioning(
        atomic_numbers=np.array([6, 6], dtype=np.int32),
        bond_indices=np.array([[0, 1], [1, 0]], dtype=np.int32),
        bond_types=np.array([2, 2], dtype=np.int32),
        formal_charges=np.zeros(2, dtype=np.int32),
        template_coords=np.array(
            [[0.0 + offset, 0.0, 0.0], [1.0 + offset, 1.0, 1.0]],
            dtype=np.float64,
        ),
        template_present=True,
        membership=np.zeros(2, dtype=np.int32),
        spacegroup_number=1,
        spacegroup_present=True,
        atom_chirality=np.zeros(2, dtype=np.int32),
        bond_stereochemistry=np.zeros(2, dtype=np.int32),
        stereochemistry_present=True,
        num_molecules=1,
    )
    return Material(
        cart_coords=np.array(
            [[0.0 + offset, 0.0, 0.0], [1.0 + offset, 1.0, 1.0]],
            dtype=np.float64,
        ),
        frac_coords=np.array(
            [[0.0 + offset / 5.0, 0.0, 0.0], [0.2 + offset / 5.0, 0.2, 0.2]],
            dtype=np.float64,
        ),
        lattice_parameters=np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0]),
        cell=np.eye(3, dtype=np.float64) * 5.0,
        conditioning=conditioning,
        info=info,
    )


def test_deduplicate_same_family_keeps_lowest_r_factor() -> None:
    """Same-family deduplication keeps the matching structure with lowest r_factor."""
    pytest.importorskip("pymatgen")
    materials = [
        _material("ABABUB01", r_factor=5.0),
        _material("ABABUB02", r_factor=2.0),
        _material("OTHER01", r_factor=9.0, offset=2.0),
    ]

    deduplicated = deduplicate_same_family(
        materials,
        DeduplicateConfig(
            enabled=True,
            structure_matcher=StructureMatcherConfig(),
        ),
    )

    assert [m.info["csd_refcode"] for m in deduplicated] == ["ABABUB02", "OTHER01"]
    duplicate_info = deduplicated[0].info["deduplication"]
    assert duplicate_info["was_deduplicated"] is True
    assert duplicate_info["matched_refcodes"] == ["ABABUB01", "ABABUB02"]
    assert duplicate_info["kept_refcode"] == "ABABUB02"
    assert duplicate_info["dropped_refcodes"] == ["ABABUB01"]
    assert duplicate_info["selection"] == "lowest_r_factor"
    assert duplicate_info["r_factors"] == {"ABABUB01": 5.0, "ABABUB02": 2.0}

    singleton_info = deduplicated[1].info["deduplication"]
    assert singleton_info["was_deduplicated"] is False
    assert singleton_info["matched_refcodes"] == ["OTHER01"]
    assert singleton_info["dropped_refcodes"] == []
    assert singleton_info["selection"] == "singleton"


@pytest.mark.parametrize("r_factor", [None, "not-a-number", float("nan")])
def test_deduplicate_same_family_requires_numeric_r_factor(r_factor: object) -> None:
    """Same-family deduplication fails clearly without finite numeric r_factor."""
    pytest.importorskip("pymatgen")

    with pytest.raises(ValueError, match="r_factor"):
        deduplicate_same_family(
            [_material("ABABUB01", r_factor=r_factor)],
            DeduplicateConfig(enabled=True),
        )


def test_molecule_conditioning_tensors_include_presence_flags() -> None:
    """Molecule conditioning exposes required optional-field presence flags."""
    conditioning = _material("ABABUB01").conditioning_tensors()

    assert bool(conditioning.template_present)
    assert bool(conditioning.stereochemistry_present)
    assert bool(conditioning.spacegroup_present)


def test_conditioning_spacegroup_marks_missing_as_absent() -> None:
    """Missing CSD space groups are represented with an absent-conditioning flag."""
    assert _conditioning_spacegroup(None) == (0, False)
    assert _conditioning_spacegroup(14) == (14, True)


def test_csd_filter_allows_neutron_single_crystal_diffraction() -> None:
    """CSD filtering no longer excludes entries solely by radiation source."""
    preprocessor = CSDPreprocessor(
        filters=CSDFilterConfig(
            date_cutoff=None,
            max_r_factor=None,
            allow_powder=False,
            allow_polymeric=False,
            require_3d_coordinates=True,
            require_ambient_pressure=True,
            require_known_spacegroup=True,
            require_organic_or_organometallic=True,
        )
    )
    entry = SimpleNamespace(
        is_powder_study=False,
        is_polymeric=False,
        has_3d_structure=True,
        pressure=None,
        is_organic=True,
        is_organometallic=False,
        radiation_source="Neutron",
    )

    checks = list(preprocessor._entry_criteria(entry))

    assert "xray" not in [reason for reason, _ in checks]
    assert all(passes for _, passes in checks)


def test_known_spacegroup_filter_uses_source_crystal_value() -> None:
    """Accept a source space group independently of reduced-crystal metadata."""
    preprocessor = CSDPreprocessor(
        filters=CSDFilterConfig(max_atoms=512, require_known_spacegroup=True)
    )
    molecules = SimpleNamespace(atoms=[object()] * 10)

    checks = list(preprocessor._crystal_molecule_criteria(14, molecules))

    assert checks == [("known_spacegroup", True), ("max_atoms", True)]


def test_collate_stacks_conditioning_presence_flags() -> None:
    """Collation stacks optional-field presence flags as global bool tensors."""
    materials = [_material("ABABUB01"), _material("OTHER01", offset=2.0)]
    batch = []
    for material in materials:
        num_atoms = material.conditioning.atomic_numbers.shape[0]
        batch.append(
            {
                "cart_coords": torch.tensor(material.cart_coords),
                "frac_coords": torch.tensor(material.frac_coords),
                "lattice": torch.tensor(material.lattice_parameters),
                "cell": torch.tensor(material.cell),
                "num_atoms": torch.tensor(num_atoms),
                "conditioning": material.conditioning_tensors(),
            }
        )

    collated = collate_fn(batch)
    conditioning = collated["conditioning"]

    assert "atomic_numbers" not in collated
    assert conditioning["atomic_numbers"].tolist() == [[6, 6], [6, 6]]
    assert conditioning["template_present"].tolist() == [True, True]
    assert conditioning["stereochemistry_present"].tolist() == [True, True]
    assert conditioning["spacegroup_present"].tolist() == [True, True]


def test_collate_pads_to_fixed_atom_bucket() -> None:
    """Collation pads atom-level tensors to the configured atom bucket."""
    material = _material("ABABUB01")
    batch = [
        {
            "cart_coords": torch.tensor(material.cart_coords),
            "frac_coords": torch.tensor(material.frac_coords),
            "lattice": torch.tensor(material.lattice_parameters),
            "cell": torch.tensor(material.cell),
            "num_atoms": torch.tensor(2),
            "conditioning": material.conditioning_tensors(),
        }
    ]

    collated = collate_fn(batch, pad_to_atom_buckets=[4, 8])

    assert collated["cart_coords"].shape == (1, 4, 3)
    assert collated["atom_mask"].tolist() == [[True, True, False, False]]
    assert collated["conditioning"]["atomic_numbers"].tolist() == [[6, 6, 0, 0]]
    assert collated["conditioning"]["bond_type_adj"].shape == (1, 4, 4)


def test_collate_rejects_fixed_bucket_overflow() -> None:
    """Collation fails when no atom bucket can hold the batch."""
    material = _material("ABABUB01")
    batch = [
        {
            "cart_coords": torch.tensor(material.cart_coords),
            "frac_coords": torch.tensor(material.frac_coords),
            "lattice": torch.tensor(material.lattice_parameters),
            "cell": torch.tensor(material.cell),
            "num_atoms": torch.tensor(2),
            "conditioning": material.conditioning_tensors(),
        }
    ]

    with pytest.raises(ValueError, match="above largest bucket"):
        collate_fn(batch, pad_to_atom_buckets=[1])


def test_collate_prepares_model_conditioning_on_cpu() -> None:
    """Context-aware collation emits model-ready conditioning ids and masks."""
    material = _material("ABABUB01")
    batch = [
        {
            "cart_coords": torch.tensor(material.cart_coords),
            "frac_coords": torch.tensor(material.frac_coords),
            "lattice": torch.tensor(material.lattice_parameters),
            "cell": torch.tensor(material.cell),
            "num_atoms": torch.tensor(2),
            "conditioning": material.conditioning_tensors(),
        }
    ]
    policy = {
        "contexts": {
            "unit": {
                "template": "off",
                "stereochemistry": "off",
                "spacegroup": "off",
            },
        },
    }

    collated = collate_fn(
        batch,
        drop_fields=["conditioning.bond_adj"],
        conditioning_policy=policy,
        conditioning_context="unit",
    )
    conditioning = collated["conditioning"]

    assert "bond_adj" not in conditioning
    assert "template_present" not in conditioning
    assert conditioning["template_mask"].tolist() == [0.0]
    assert conditioning["stereochemistry_mask"].tolist() == [0.0]
    assert conditioning["spacegroup_mask"].tolist() == [0.0]
    assert conditioning["atom_chirality"].tolist() == [
        [ATOM_CHIRALITY_MODEL_NULL_ID, ATOM_CHIRALITY_MODEL_NULL_ID]
    ]
    assert conditioning["bond_type_adj"].tolist() == [
        [[BOND_TYPE_MODEL_NO_BOND_ID, 3], [3, BOND_TYPE_MODEL_NO_BOND_ID]]
    ]
    assert conditioning["bond_stereochemistry_adj"].tolist() == [
        [
            [0, BOND_STEREOCHEMISTRY_MODEL_NULL_ID],
            [BOND_STEREOCHEMISTRY_MODEL_NULL_ID, 0],
        ]
    ]
    assert conditioning["spacegroup_number"].tolist() == [SPACEGROUP_MODEL_NULL_ID]
