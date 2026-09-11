# ruff: noqa: F722,F821
# Rotation sampling adapted from PyTorch3D (BSD License, Copyright (c) Meta Platforms, Inc.)

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from einops import rearrange
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.types import Device

from src.utils.tensor_typing import Bool, Float, Int


def _copysign(a: Float["..."], b: Float["..."]) -> Float["..."]:
    """Return tensor with magnitude of a and sign of b."""
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _random_quaternions(
    n: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Device] = None,
) -> Float["n 4"]:
    """Sample n unit quaternions uniformly at random."""
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def _quaternion_to_matrix(quaternions: Float["... 4"]) -> Float["... 3 3"]:
    """Convert quaternions (real-part-first) to rotation matrices (..., 3, 3)."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def _random_rotations(
    n: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Device] = None,
) -> Float["n 3 3"]:
    """Sample n random SO(3) rotation matrices (n, 3, 3)."""
    return _quaternion_to_matrix(_random_quaternions(n, dtype=dtype, device=device))


def _as_bool_list(flags: Bool["e"] | Sequence[bool] | None) -> list[bool] | None:
    """Convert optional rotatable-bond flags to Python bools."""
    if flags is None:
        return None
    if isinstance(flags, torch.Tensor):
        return [bool(flag) for flag in flags.detach().cpu().tolist()]
    return [bool(flag) for flag in flags]


def _connected_side(
    adjacency: dict[int, set[int]],
    start: int,
    blocked_edge: tuple[int, int],
) -> set[int]:
    """Return atoms reachable from start with one undirected edge removed."""
    blocked = frozenset(blocked_edge)
    seen = {start}
    queue = [start]
    while queue:
        atom = queue.pop()
        for neighbor in adjacency[atom]:
            if frozenset((atom, neighbor)) == blocked:
                continue
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def _rotate_mask_around_bond(
    coords: Float["n 3"],
    atom_mask: Bool["n"],
    start: int,
    end: int,
    angle: Float[""],
) -> Float["n 3"]:
    """Rotate selected atoms around one bond axis."""
    axis_start = coords[start]
    axis_end = coords[end]
    axis = axis_end - axis_start
    axis_norm = axis.norm()
    if axis_norm <= torch.finfo(coords.dtype).eps:
        return coords

    axis = axis / axis_norm
    selected = coords[atom_mask] - axis_start
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    projected = (selected @ axis).unsqueeze(-1) * axis
    crossed = torch.cross(axis.expand_as(selected), selected, dim=-1)
    rotated = selected * cos_angle + crossed * sin_angle + projected * (1.0 - cos_angle)

    coords = coords.clone()
    coords[atom_mask] = rotated + axis_start
    return coords


def _augment_template_torsions(
    coords: Float["n 3"],
    membership: Int["n"],
    bond_indices: Int["2 e"] | None,
    bond_is_rotatable: Bool["e"] | Sequence[bool] | None,
) -> Float["n 3"]:
    """Apply full random torsion updates around bridge-like rotatable bonds."""
    if bond_indices is None:
        return coords

    rotatable_flags = _as_bool_list(bond_is_rotatable)
    if rotatable_flags is None:
        return coords
    if bond_indices.shape[1] != len(rotatable_flags):
        raise ValueError(
            "bond_is_rotatable must align with directed bond_indices columns."
        )

    bond_pairs = bond_indices.detach().cpu().tolist()
    memberships = [int(value) for value in membership.detach().cpu().tolist()]
    adjacency = {atom_idx: set() for atom_idx in range(coords.shape[0])}
    rotatable_edges: dict[tuple[int, int], bool] = {}

    for src, dst, is_rotatable in zip(bond_pairs[0], bond_pairs[1], rotatable_flags):
        src = int(src)
        dst = int(dst)
        if memberships[src] != memberships[dst]:
            continue

        adjacency[src].add(dst)
        adjacency[dst].add(src)
        edge = tuple(sorted((src, dst)))
        rotatable_edges[edge] = rotatable_edges.get(edge, False) or is_rotatable

    for edge, is_rotatable in rotatable_edges.items():
        if not is_rotatable:
            continue

        start, end = edge
        start_side = _connected_side(adjacency, start, edge)
        if end in start_side:
            continue

        end_side = _connected_side(adjacency, end, edge)
        rotate_side = start_side if len(start_side) <= len(end_side) else end_side
        atom_mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
        atom_mask[list(rotate_side)] = True
        angle = torch.empty((), dtype=coords.dtype, device=coords.device).uniform_(
            -torch.pi,
            torch.pi,
        )
        coords = _rotate_mask_around_bond(coords, atom_mask, start, end, angle)

    return coords


def augment_template(
    coords: Float["n 3"],
    membership: Int["n"],
    bond_indices: Int["2 e"] | None = None,
    bond_is_rotatable: Bool["e"] | Sequence[bool] | None = None,
    center: bool = True,
    rotate: bool = False,
    translate: bool = False,
    s_trans: float = 1.0,
    torsion_perturb: bool = False,
    jitter: bool = False,
    jitter_sigma: float = 0.05,
) -> Float["n 3"]:
    """Apply per-molecule random rigid-body augmentation to template coordinates."""
    coords = coords.clone()

    # NOTE: Local geometry perturbations run before global normalization so final
    # centering and translation semantics stay controlled by the rigid transforms.
    if torsion_perturb:
        coords = _augment_template_torsions(
            coords,
            membership,
            bond_indices,
            bond_is_rotatable,
        )
    if jitter:
        coords = coords + torch.randn_like(coords) * jitter_sigma

    # Sort atoms by molecule index
    sort_order = membership.argsort(stable=True)
    sorted_coords = coords[sort_order]

    # Split into per-molecule coordinate lists
    mol_ids, mol_sizes = membership.unique(sorted=True, return_counts=True)
    coords_per_mol = sorted_coords.split(mol_sizes.tolist())

    num_mols = mol_ids.shape[0]
    max_atoms = mol_sizes.max().item()

    # Batch coordinates with padding
    batched_coords = pad_sequence(
        coords_per_mol, batch_first=True
    )  # (num_mols, max_atoms, 3)
    atom_mask = (
        torch.arange(max_atoms, device=coords.device) < mol_sizes.unsqueeze(-1)
    ).unsqueeze(-1)  # (num_mols, max_atoms, 1)

    # Center molecules
    if center:
        sum_coords = (batched_coords * atom_mask).sum(dim=1, keepdim=True)
        valid_counts = atom_mask.sum(dim=1, keepdim=True).clamp(min=1)
        centers = sum_coords / valid_counts
        batched_coords = batched_coords - centers

    # Random rotation
    if rotate:
        rotations = _random_rotations(
            num_mols, dtype=coords.dtype, device=coords.device
        )
        batched_coords = torch.einsum("mij,mnj->mni", rotations, batched_coords)

    # Random translation
    if translate:
        translations = (
            torch.randn(
                num_mols,
                1,
                3,
                dtype=coords.dtype,
                device=coords.device,
            )
            * s_trans
        )
        batched_coords = batched_coords + translations

    # Unpad and restore original order
    unpadded_coords = batched_coords[atom_mask.squeeze(-1)]
    restore_order = sort_order.argsort()
    return unpadded_coords[restore_order]


def augment_crystal(
    cart_coords: Float["n 3"],
    frac_coords: Float["n 3"],
    cell: Float["3 3"],
    rotate: bool = False,
    translate: bool = False,
    membership: Optional[Int["n"]] = None,
) -> tuple[Float["n 3"], Float["n 3"], Float["3 3"]]:
    """Apply crystal-aware rotation and translation augmentation."""
    # Random SO(3) rotation: rotate cell, frac_coords are invariant
    if rotate:
        R = _random_rotations(1, dtype=cell.dtype, device=cell.device).squeeze(0)
        cell = cell @ R.t()

    # Random fractional translation + wrapping into unit cell
    if translate:
        t = torch.rand(3, dtype=frac_coords.dtype, device=frac_coords.device)
        if membership is not None:
            # Molecule-aware: translate all atoms, then wrap per-molecule
            frac_coords = frac_coords + t
            for mol_id in membership.unique(sorted=True):
                mask = membership == mol_id
                centroid = frac_coords[mask].mean(dim=0)
                frac_coords[mask] += centroid.floor().neg()
        else:
            frac_coords = (frac_coords + t) % 1.0

    # Recompute Cartesian from (possibly rotated) cell and (possibly translated) frac_coords
    cart_coords = frac_coords @ cell

    return cart_coords, frac_coords, cell
