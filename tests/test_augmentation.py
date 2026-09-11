"""Verify crystal augmentation preserves physical equivalence."""

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from src.data.components.transforms import augment_crystal, augment_template


def _make_structure(
    cart_coords: torch.Tensor, cell: torch.Tensor, atomic_numbers: torch.Tensor
) -> Structure:
    """Build a pymatgen Structure from tensors."""
    lattice = Lattice(cell.numpy())
    frac = lattice.get_fractional_coords(cart_coords.numpy())
    species = atomic_numbers.tolist()
    return Structure(lattice, species, frac)


def test_rotation_preserves_structure() -> None:
    """Rotated crystal must match the original under StructureMatcher."""
    cell = torch.eye(3) * 5.0
    frac_coords = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    cart_coords = frac_coords @ cell
    atomic_numbers = torch.tensor([11, 17])

    matcher = StructureMatcher(stol=0.3, ltol=0.2, angle_tol=5.0)
    orig = _make_structure(cart_coords, cell, atomic_numbers)

    torch.manual_seed(42)
    for _ in range(20):
        aug_cart, aug_frac, aug_cell = augment_crystal(
            cart_coords,
            frac_coords,
            cell,
            rotate=True,
            translate=False,
        )
        aug = _make_structure(aug_cart, aug_cell, atomic_numbers)
        assert matcher.fit(orig, aug), "Rotated structure does not match original"


def test_translation_preserves_structure() -> None:
    """Translated + wrapped crystal must match the original."""
    cell = torch.diag(torch.tensor([4.0, 5.0, 6.0]))
    frac_coords = torch.tensor([[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]])
    cart_coords = frac_coords @ cell
    atomic_numbers = torch.tensor([26, 8])

    matcher = StructureMatcher(stol=0.3, ltol=0.2, angle_tol=5.0)
    orig = _make_structure(cart_coords, cell, atomic_numbers)

    torch.manual_seed(0)
    for _ in range(20):
        aug_cart, aug_frac, aug_cell = augment_crystal(
            cart_coords,
            frac_coords,
            cell,
            rotate=False,
            translate=True,
        )
        aug = _make_structure(aug_cart, aug_cell, atomic_numbers)
        assert matcher.fit(orig, aug), "Translated structure does not match original"


def test_rotation_and_translation_preserves_structure() -> None:
    """Combined rotation + translation must preserve equivalence."""
    cell = torch.tensor([[5.0, 0.0, 0.0], [1.0, 4.5, 0.0], [0.5, 0.3, 6.0]])
    frac_coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.0],
            [0.75, 0.75, 0.75],
        ]
    )
    cart_coords = frac_coords @ cell
    atomic_numbers = torch.tensor([14, 14, 8, 8])

    matcher = StructureMatcher(stol=0.3, ltol=0.2, angle_tol=5.0)
    orig = _make_structure(cart_coords, cell, atomic_numbers)

    torch.manual_seed(7)
    for _ in range(20):
        aug_cart, aug_frac, aug_cell = augment_crystal(
            cart_coords,
            frac_coords,
            cell,
            rotate=True,
            translate=True,
        )
        aug = _make_structure(aug_cart, aug_cell, atomic_numbers)
        assert matcher.fit(orig, aug), "Augmented structure does not match original"


def _make_molecular_crystal() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Build a 2-molecule crystal (2x H2O) for molecular augmentation tests."""
    cell = torch.eye(3) * 5.0
    # Two water molecules with distinct positions
    frac_coords = torch.tensor(
        [
            [0.1, 0.1, 0.1],  # mol 0: O
            [0.15, 0.1, 0.1],  # mol 0: H
            [0.1, 0.15, 0.1],  # mol 0: H
            [0.8, 0.8, 0.8],  # mol 1: O
            [0.85, 0.8, 0.8],  # mol 1: H
            [0.8, 0.85, 0.8],  # mol 1: H
        ]
    )
    cart_coords = frac_coords @ cell
    atomic_numbers = torch.tensor([8, 1, 1, 8, 1, 1])
    membership = torch.tensor([0, 0, 0, 1, 1, 1])
    return cart_coords, frac_coords, cell, atomic_numbers, membership


def test_molecular_translation_preserves_structure() -> None:
    """Molecule-aware translation must preserve crystal equivalence."""
    cart_coords, frac_coords, cell, atomic_numbers, membership = (
        _make_molecular_crystal()
    )
    matcher = StructureMatcher(stol=0.3, ltol=0.2, angle_tol=5.0)
    orig = _make_structure(cart_coords, cell, atomic_numbers)

    torch.manual_seed(42)
    for _ in range(20):
        aug_cart, aug_frac, aug_cell = augment_crystal(
            cart_coords,
            frac_coords,
            cell,
            rotate=False,
            translate=True,
            membership=membership,
        )
        aug = _make_structure(aug_cart, aug_cell, atomic_numbers)
        assert matcher.fit(orig, aug), "Molecular translation broke crystal equivalence"


def test_molecular_translation_preserves_intramolecular_distances() -> None:
    """Pairwise distances within each molecule must be exactly preserved."""
    cart_coords, frac_coords, cell, _, membership = _make_molecular_crystal()

    torch.manual_seed(7)
    for _ in range(20):
        aug_cart, _, _ = augment_crystal(
            cart_coords,
            frac_coords,
            cell,
            rotate=False,
            translate=True,
            membership=membership,
        )
        for mol_id in membership.unique():
            mask = membership == mol_id
            orig_dists = torch.cdist(cart_coords[mask], cart_coords[mask])
            aug_dists = torch.cdist(aug_cart[mask], aug_cart[mask])
            assert torch.allclose(orig_dists, aug_dists, atol=1e-5), (
                f"Intramolecular distances changed for molecule {mol_id.item()}"
            )


def test_molecular_translation_centroids_in_cell() -> None:
    """After molecular translation, every molecule centroid must lie in [0, 1)."""
    cart_coords, frac_coords, cell, _, membership = _make_molecular_crystal()

    torch.manual_seed(99)
    for _ in range(20):
        _, aug_frac, _ = augment_crystal(
            cart_coords,
            frac_coords,
            cell,
            rotate=False,
            translate=True,
            membership=membership,
        )
        for mol_id in membership.unique():
            centroid = aug_frac[membership == mol_id].mean(dim=0)
            assert (centroid >= 0).all() and (centroid < 1).all(), (
                f"Centroid {centroid.tolist()} outside [0, 1) for molecule {mol_id.item()}"
            )


def test_molecular_translation_backward_compat() -> None:
    """With membership=None, per-atom wrapping into [0, 1) must apply."""
    cell = torch.eye(3) * 5.0
    frac_coords = torch.tensor([[0.1, 0.9, 0.5], [0.3, 0.4, 0.7]])
    cart_coords = frac_coords @ cell

    torch.manual_seed(0)
    for _ in range(20):
        _, aug_frac, _ = augment_crystal(
            cart_coords,
            frac_coords,
            cell,
            rotate=False,
            translate=True,
            membership=None,
        )
        assert (aug_frac >= 0).all() and (aug_frac < 1).all(), (
            "Per-atom wrapping failed with membership=None"
        )


def test_augmentation_tensor_invariants() -> None:
    """Verify tensor-level invariants after augmentation."""
    cell = torch.diag(torch.tensor([5.0, 5.0, 5.0]))
    frac_coords = torch.tensor([[0.1, 0.9, 0.5], [0.3, 0.4, 0.7]])
    cart_coords = frac_coords @ cell

    torch.manual_seed(99)
    aug_cart, aug_frac, aug_cell = augment_crystal(
        cart_coords,
        frac_coords,
        cell,
        rotate=True,
        translate=True,
    )

    # frac_coords in [0, 1)
    assert (aug_frac >= 0).all() and (aug_frac < 1).all()

    # cart = frac @ cell consistency
    assert torch.allclose(aug_cart, aug_frac @ aug_cell, atol=1e-5)

    # Lattice parameters (lengths and angles) preserved after rotation
    def lattice_params(
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute lattice lengths and cosines of angles from cell matrix."""
        lengths = c.norm(dim=1)
        cos_alpha = (c[1] @ c[2]) / (lengths[1] * lengths[2])
        cos_beta = (c[0] @ c[2]) / (lengths[0] * lengths[2])
        cos_gamma = (c[0] @ c[1]) / (lengths[0] * lengths[1])
        return lengths, torch.stack([cos_alpha, cos_beta, cos_gamma])

    orig_len, orig_ang = lattice_params(cell)
    aug_len, aug_ang = lattice_params(aug_cell)
    assert torch.allclose(orig_len, aug_len, atol=1e-5)
    assert torch.allclose(orig_ang, aug_ang, atol=1e-5)


def test_template_new_augmentations_disabled_preserve_behavior() -> None:
    """Disabled template torsion and jitter must not change existing outputs."""
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 3.0, 3.0],
            [4.0, 3.0, 3.0],
        ]
    )
    membership = torch.tensor([0, 0, 0, 1, 1])
    bond_indices = torch.tensor([[0, 1], [1, 0]])
    bond_is_rotatable = torch.tensor([True, True])

    torch.manual_seed(123)
    baseline = augment_template(
        coords,
        membership,
        rotate=True,
        translate=True,
    )
    torch.manual_seed(123)
    augmented = augment_template(
        coords,
        membership,
        bond_indices=bond_indices,
        bond_is_rotatable=bond_is_rotatable,
        rotate=True,
        translate=True,
        torsion_perturb=False,
        jitter=False,
    )

    assert torch.allclose(augmented, baseline)


def test_template_jitter_is_seeded_and_changes_coordinates() -> None:
    """Template jitter must be deterministic under torch manual seeds."""
    coords = torch.zeros(5, 3)
    membership = torch.zeros(5, dtype=torch.long)

    torch.manual_seed(7)
    first = augment_template(
        coords,
        membership,
        center=False,
        jitter=True,
        jitter_sigma=0.05,
    )
    torch.manual_seed(7)
    second = augment_template(
        coords,
        membership,
        center=False,
        jitter=True,
        jitter_sigma=0.05,
    )

    assert torch.allclose(first, second)
    assert not torch.allclose(first, coords)


def test_template_local_augmentations_keep_final_centering() -> None:
    """Torsion and jitter run before centering so final molecule centers are zero."""
    coords = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [10.0, 1.0, 0.0],
            [11.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [13.0, 1.0, 0.0],
        ]
    )
    membership = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    bond_indices = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6],
        ]
    )
    bond_is_rotatable = torch.tensor(
        [
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
        ]
    )

    torch.manual_seed(11)
    augmented = augment_template(
        coords,
        membership,
        bond_indices=bond_indices,
        bond_is_rotatable=bond_is_rotatable,
        rotate=True,
        translate=False,
        torsion_perturb=True,
        jitter=True,
        jitter_sigma=0.05,
    )

    for mol_id in membership.unique(sorted=True):
        centroid = augmented[membership == mol_id].mean(dim=0)
        assert torch.allclose(centroid, torch.zeros(3), atol=1e-6)


def test_template_torsion_preserves_bond_lengths() -> None:
    """Torsion updates must preserve covalent bond lengths on an acyclic chain."""
    coords = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    membership = torch.zeros(4, dtype=torch.long)
    bond_indices = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    bond_is_rotatable = torch.tensor([False, False, True, True, False, False])

    torch.manual_seed(5)
    augmented = augment_template(
        coords,
        membership,
        bond_indices=bond_indices,
        bond_is_rotatable=bond_is_rotatable,
        center=False,
        torsion_perturb=True,
    )

    undirected_edges = [(0, 1), (1, 2), (2, 3)]
    for start, end in undirected_edges:
        original_length = (coords[start] - coords[end]).norm()
        augmented_length = (augmented[start] - augmented[end]).norm()
        assert torch.allclose(augmented_length, original_length, atol=1e-5)
    assert not torch.allclose(augmented, coords)


def test_template_torsion_skips_non_bridge_rotatable_bonds() -> None:
    """Rotatable bonds that do not split the graph must be skipped."""
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ]
    )
    membership = torch.zeros(3, dtype=torch.long)
    bond_indices = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    bond_is_rotatable = torch.tensor([True, True, False, False, False, False])

    torch.manual_seed(17)
    augmented = augment_template(
        coords,
        membership,
        bond_indices=bond_indices,
        bond_is_rotatable=bond_is_rotatable,
        center=False,
        torsion_perturb=True,
    )

    assert torch.allclose(augmented, coords)
