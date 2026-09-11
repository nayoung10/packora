from types import SimpleNamespace

import torch
import pytest

from src.data.components.indexing import (
    INDEX_BUILDERS,
    get_index_builder,
    sequential,
    atomic_number_random,
    atomic_number_xyz,
    xyz,
)


def _conditioning(atomic_numbers: list[int]) -> SimpleNamespace:
    """Create a minimal conditioning object for indexing tests."""
    return SimpleNamespace(atomic_numbers=torch.tensor(atomic_numbers))


def _make_sample(n: int = 5) -> dict:
    """Create a sample dict with distinct coordinates and mixed atomic numbers."""
    return {
        "conditioning": _conditioning([6, 8, 6, 1, 8]),
        "cart_coords": torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
                [3.0, 2.0, 2.0],
            ]
        ),
    }


def _assert_valid_permutation(indices: torch.Tensor, n: int) -> None:
    """Assert indices form a valid permutation of [0, n)."""
    assert indices.shape == (n,), f"Expected shape ({n},), got {indices.shape}"
    assert indices.dtype == torch.long, f"Expected long dtype, got {indices.dtype}"
    assert set(indices.tolist()) == set(range(n)), (
        f"Expected permutation of {set(range(n))}, got {set(indices.tolist())}"
    )


# ---------------------------------------------------------------------------
# sequential
# ---------------------------------------------------------------------------


class TestSequential:
    def test_returns_identity(self) -> None:
        """Sequential should return 0..N-1."""
        sample = _make_sample()
        indices = sequential(sample)
        _assert_valid_permutation(indices, 5)
        assert indices.tolist() == [0, 1, 2, 3, 4]

    def test_single_atom(self) -> None:
        """Single atom returns tensor([0])."""
        sample = {"conditioning": _conditioning([6]), "cart_coords": torch.zeros(1, 3)}
        indices = sequential(sample)
        assert indices.tolist() == [0]


# ---------------------------------------------------------------------------
# atomic_number_random
# ---------------------------------------------------------------------------


class TestAtomicNumberRandom:
    def test_valid_permutation(self) -> None:
        """Returns a valid permutation."""
        sample = _make_sample()
        indices = atomic_number_random(sample)
        _assert_valid_permutation(indices, 5)

    def test_stochastic(self) -> None:
        """Two calls may produce different results (random tiebreaking)."""
        sample = _make_sample()
        results = {tuple(atomic_number_random(sample).tolist()) for _ in range(50)}
        # With 2 atoms of Z=6 and 2 of Z=8, random tiebreaking should produce variation
        assert len(results) > 1, "Expected stochastic variation across calls"

    def test_ascending_descending(self) -> None:
        """Descending should reverse the element ordering."""
        sample = {
            "conditioning": _conditioning([1, 6, 8]),
            "cart_coords": torch.zeros(3, 3),
        }
        asc = atomic_number_random(sample, descending=False)
        desc = atomic_number_random(sample, descending=True)
        # H (1) should get lowest index ascending, highest descending
        assert asc[0] < asc[2], "H should rank before O in ascending"
        assert desc[0] > desc[2], "H should rank after O in descending"


# ---------------------------------------------------------------------------
# atomic_number_xyz
# ---------------------------------------------------------------------------


class TestAtomicNumberXyz:
    def test_valid_permutation(self) -> None:
        """Returns a valid permutation."""
        sample = _make_sample()
        indices = atomic_number_xyz(sample)
        _assert_valid_permutation(indices, 5)

    def test_deterministic(self) -> None:
        """Same input produces same output."""
        sample = _make_sample()
        a = atomic_number_xyz(sample)
        b = atomic_number_xyz(sample)
        assert torch.equal(a, b)

    def test_element_primary_sort(self) -> None:
        """Atoms with lower atomic number get lower indices."""
        sample = _make_sample()
        indices = atomic_number_xyz(sample)
        # H (idx 3, Z=1) should get the lowest index
        assert indices[3] == 0, "H (Z=1) should get index 0"

    def test_descending(self) -> None:
        """Descending reverses element ordering."""
        sample = _make_sample()
        asc = atomic_number_xyz(sample, descending=False)
        desc = atomic_number_xyz(sample, descending=True)
        # H has the lowest Z — index 0 ascending, index N-1 descending
        assert asc[3] == 0
        assert desc[3] == 4


# ---------------------------------------------------------------------------
# xyz
# ---------------------------------------------------------------------------


class TestXyz:
    def test_valid_permutation(self) -> None:
        """Returns a valid permutation."""
        sample = _make_sample()
        indices = xyz(sample)
        _assert_valid_permutation(indices, 5)

    def test_deterministic(self) -> None:
        """Same input produces same output."""
        sample = _make_sample()
        a = xyz(sample)
        b = xyz(sample)
        assert torch.equal(a, b)

    def test_x_dominates_ordering(self) -> None:
        """Atom with smallest x gets index 0 in ascending mode."""
        sample = _make_sample()
        indices = xyz(sample)
        # cart_coords x-values: [2.0, 1.0, 0.5, 0.5, 3.0]
        # Sorted by x: indices 2,3 (x=0.5), 1 (x=1.0), 0 (x=2.0), 4 (x=3.0)
        # Between atoms 2 and 3 (same x=0.5): y=1.0 vs y=0.5 → atom 3 first
        assert indices[3] == 0, "Atom 3 (x=0.5, y=0.5) should get index 0"
        assert indices[2] == 1, "Atom 2 (x=0.5, y=1.0) should get index 1"
        assert indices[1] == 2, "Atom 1 (x=1.0) should get index 2"
        assert indices[0] == 3, "Atom 0 (x=2.0) should get index 3"
        assert indices[4] == 4, "Atom 4 (x=3.0) should get index 4"

    def test_descending(self) -> None:
        """Descending reverses the ordering."""
        sample = _make_sample()
        desc = xyz(sample, descending=True)
        # The atom with the highest x (atom 4, x=3.0) should get index 0 descending
        assert desc[4] == 0

    def test_single_atom(self) -> None:
        """Single atom returns tensor([0])."""
        sample = {
            "conditioning": _conditioning([6]),
            "cart_coords": torch.tensor([[1.0, 2.0, 3.0]]),
        }
        indices = xyz(sample)
        assert indices.tolist() == [0]

    def test_identical_coordinates(self) -> None:
        """Handles identical coordinates gracefully (stable sort)."""
        sample = {
            "conditioning": _conditioning([6, 8, 1]),
            "cart_coords": torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
        }
        indices = xyz(sample)
        _assert_valid_permutation(indices, 3)

    def test_ignores_atomic_number(self) -> None:
        """Ordering depends only on coordinates, not atomic numbers."""
        coords = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        sample_a = {"conditioning": _conditioning([1, 8]), "cart_coords": coords}
        sample_b = {"conditioning": _conditioning([8, 1]), "cart_coords": coords}
        assert torch.equal(xyz(sample_a), xyz(sample_b))


# ---------------------------------------------------------------------------
# get_index_builder factory
# ---------------------------------------------------------------------------


class TestGetIndexBuilder:
    def test_all_registered(self) -> None:
        """All 4 schemes are registered."""
        expected = {"sequential", "atomic_number_random", "atomic_number_xyz", "xyz"}
        assert set(INDEX_BUILDERS.keys()) == expected

    def test_factory_returns_callable(self) -> None:
        """Factory returns a callable for each registered scheme."""
        for name in INDEX_BUILDERS:
            cfg = {"index_type": name}
            builder = get_index_builder(cfg)
            assert callable(builder)

    def test_factory_binds_kwargs(self) -> None:
        """Extra config keys are bound via partial."""
        cfg = {"index_type": "xyz", "descending": True}
        builder = get_index_builder(cfg)
        sample = _make_sample()
        indices = builder(sample)
        _assert_valid_permutation(indices, 5)
        # Atom 4 (x=3.0, highest) should get index 0 with descending=True
        assert indices[4] == 0

    def test_unknown_scheme_raises(self) -> None:
        """Unknown scheme raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_index_builder({"index_type": "nonexistent"})
