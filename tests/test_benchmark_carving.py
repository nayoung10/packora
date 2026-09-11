import numpy as np

from src.data.preprocess.utils.benchmark import (
    BenchmarkCsvConfig,
    BenchmarkRefcode,
    ComponentFilterConfig,
    ComponentRecord,
    build_component_index,
    build_family_index,
    component_is_eligible,
    filter_materials_by_components,
    filter_materials_by_family,
    load_benchmark_refcodes,
    material_component_records,
)
from src.data.types import Material, MoleculeConditioning


def _material(
    refcode: str,
    component_smiles: list[str | None],
    component_atomic_numbers: list[list[int]],
) -> Material:
    """Build a minimal material with component SMILES metadata."""
    atomic_numbers = np.array(
        [number for component in component_atomic_numbers for number in component],
        dtype=np.int32,
    )
    membership = np.array(
        [
            component_index
            for component_index, component in enumerate(component_atomic_numbers)
            for _ in component
        ],
        dtype=np.int32,
    )
    num_atoms = int(atomic_numbers.shape[0])
    conditioning = MoleculeConditioning(
        atomic_numbers=atomic_numbers,
        bond_indices=np.zeros((2, 0), dtype=np.int32),
        bond_types=np.zeros(0, dtype=np.int32),
        formal_charges=np.zeros(num_atoms, dtype=np.int32),
        template_coords=np.zeros((num_atoms, 3), dtype=np.float64),
        template_present=False,
        membership=membership,
        spacegroup_number=1,
        spacegroup_present=True,
        atom_chirality=np.zeros(num_atoms, dtype=np.int32),
        bond_stereochemistry=np.zeros(0, dtype=np.int32),
        stereochemistry_present=True,
        num_molecules=len(component_smiles),
    )
    return Material(
        cart_coords=np.zeros((num_atoms, 3), dtype=np.float64),
        frac_coords=np.zeros((num_atoms, 3), dtype=np.float64),
        lattice_parameters=np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0]),
        cell=np.eye(3, dtype=np.float64) * 5.0,
        conditioning=conditioning,
        info={
            "dataset_name": "csd",
            "csd_refcode": refcode,
            "csd_component_smiles": component_smiles,
        },
    )


def test_load_benchmark_refcodes_handles_bom_and_semicolon_groups(tmp_path) -> None:
    """Benchmark CSV loading handles BOMs and semicolon-separated refcodes."""
    path = tmp_path / "truth.csv"
    path.write_text("\ufeffCSD_ID,name\nABCDEF;GHIJKL,row\nABCDEF,row2\n")

    records = load_benchmark_refcodes(
        [BenchmarkCsvConfig(name="oxtal", path=path, refcode_column="CSD_ID")]
    )

    assert records == [
        BenchmarkRefcode(refcode="ABCDEF", family="ABCDEF", source="oxtal"),
        BenchmarkRefcode(refcode="GHIJKL", family="GHIJKL", source="oxtal"),
    ]


def test_component_eligibility_uses_exact_solvent_and_heavy_atom_rules() -> None:
    """Component eligibility keeps only non-solvent components above threshold."""
    config = ComponentFilterConfig(min_component_heavy_atoms=8, ignore_solvents=True)

    assert component_is_eligible("CCCCCCCC", 8, {"O"}, config) == (True, None)
    assert component_is_eligible("O", 1, {"O"}, config) == (False, "solvent")
    assert component_is_eligible("[Cl-]", 1, {"O"}, config) == (
        False,
        "min_component_heavy_atoms",
    )
    assert component_is_eligible(None, 0, set(), config) == (False, "missing_smiles")


def test_material_component_records_use_persisted_csd_smiles() -> None:
    """Candidate component records are built from persisted exact CSD SMILES."""
    material = _material(
        "ABCDEF01",
        ["O", "CCCCCCCC", "[Cl-]", None],
        [[8], [6, 6, 6, 6, 6, 6, 6, 6], [17], [6, 6, 6, 6, 6, 6, 6, 6]],
    )

    records, skips = material_component_records(
        material,
        {"O"},
        ComponentFilterConfig(min_component_heavy_atoms=8, ignore_solvents=True),
    )

    assert [record.smiles for record in records] == ["CCCCCCCC"]
    assert [(skip.component_index, skip.reason) for skip in skips] == [
        (0, "solvent"),
        (2, "min_component_heavy_atoms"),
        (3, "missing_smiles"),
    ]


def test_filter_materials_by_family_reports_primary_reason() -> None:
    """Family filtering removes materials with benchmark-family overlap."""
    material = _material("ABCDEF01", ["CCCCCCCC"], [[6, 6, 6, 6, 6, 6, 6, 6]])
    index = build_family_index(
        [BenchmarkRefcode(refcode="ABCDEF", family="ABCDEF", source="oxtal")]
    )

    kept, exclusions = filter_materials_by_family([material], index)

    assert kept == []
    assert exclusions[0].refcode == "ABCDEF01"
    assert exclusions[0].reason == "benchmark_refcode_family"
    assert exclusions[0].benchmark_refcodes == ["ABCDEF"]


def test_filter_materials_by_components_reports_exact_smiles_overlap() -> None:
    """Component filtering removes materials sharing eligible exact CSD SMILES."""
    material = _material("ZZZZZZ01", ["CCCCCCCC"], [[6, 6, 6, 6, 6, 6, 6, 6]])
    component_index = build_component_index(
        [
            ComponentRecord(
                refcode="ABCDEF",
                family="ABCDEF",
                source="oxtal",
                component_index=0,
                smiles="CCCCCCCC",
                heavy_atoms=8,
            )
        ]
    )

    kept, exclusions, skips = filter_materials_by_components(
        [material],
        component_index,
        set(),
        ComponentFilterConfig(min_component_heavy_atoms=8, ignore_solvents=True),
    )

    assert kept == []
    assert skips == []
    assert exclusions[0].refcode == "ZZZZZZ01"
    assert exclusions[0].reason == "benchmark_component"
    assert exclusions[0].matches[0].smiles == "CCCCCCCC"
    assert exclusions[0].matches[0].benchmark_refcodes == ["ABCDEF"]
