import numpy as np
import pytest
from rdkit import Chem

from src.data.preprocess.csd import CSDPreprocessor
from src.data.preprocess.utils.rdkit import (
    RDKitConformerError,
    generate_rdkit_conformer_from_sdf,
)


def _sdf_from_smiles(smiles: str) -> tuple[str, list[int]]:
    """Build an SDF block and atomic numbers from a sanitized RDKit molecule."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    atomic_numbers = [int(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    return Chem.MolToMolBlock(mol), atomic_numbers


def test_single_atom_conformer_skips_embedding_and_uff() -> None:
    """Single-atom conformers are placed at the origin without UFF."""
    result = generate_rdkit_conformer_from_sdf("", [53], 2, 125, False, 1000)

    assert np.allclose(result.coords, np.zeros((1, 3)))
    assert result.metadata["component_index"] == 2
    assert result.metadata["seed"] == 125
    assert result.metadata["uff_status"] == "skipped_single_atom"


def test_rdkit_conformer_skips_uff_when_disabled() -> None:
    """RDKit conformer generation leaves UFF disabled when requested."""
    sdf_block, atomic_numbers = _sdf_from_smiles("CO")

    result = generate_rdkit_conformer_from_sdf(
        sdf_block,
        atomic_numbers,
        0,
        123,
        False,
        1000,
    )

    assert result.coords.shape == (len(atomic_numbers), 3)
    assert result.metadata["relax_with_uff"] is False
    assert result.metadata["uff_status"] == "skipped_disabled"


def test_csd_preprocessor_disables_uff_by_default() -> None:
    """CSD preprocessing leaves RDKit UFF relaxation opt-in."""
    preprocessor = CSDPreprocessor()

    assert preprocessor.rdkit_template.random_seed == 123
    assert preprocessor.rdkit_template.relax_with_uff is False
    assert preprocessor.rdkit_template.uff_max_iters == 1000


def test_rdkit_conformer_validates_atomic_number_order() -> None:
    """RDKit conformer generation rejects atom-order mismatches."""
    sdf_block, atomic_numbers = _sdf_from_smiles("CO")

    with pytest.raises(RDKitConformerError, match="rdkit_atom_order_mismatch"):
        generate_rdkit_conformer_from_sdf(
            sdf_block,
            list(reversed(atomic_numbers)),
            0,
            123,
            False,
            1000,
        )
