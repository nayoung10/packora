"""MLIP energy wrappers used for prediction-time steering."""

from __future__ import annotations

import torch
import torch.nn as nn
from ase import Atoms
from einops import rearrange

from src.utils.tensor_typing import Bool, Float, Int


class MLIPEnergy(nn.Module):
    """Evaluate batched crystal energies with a fairchem MLIP."""

    def __init__(
        self,
        model_name: str = "uma-s-1p2",
        task_name: str = "omc",
        device: str = "cuda",
        max_batch_size: int = 64,
    ) -> None:
        """Load a fairchem prediction unit for energy-only inference."""
        super().__init__()
        from fairchem.core import pretrained_mlip

        self.model_name = model_name
        self.task_name = task_name
        self.device = device
        self.max_batch_size = int(max_batch_size)
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1.")
        self.predictor = pretrained_mlip.get_predict_unit(model_name, device=device)

    def _cell_from_lattice(self, lattice: Float["d"]) -> object:
        """Convert one 6D lattice or flattened 3x3 cell to an ASE cell."""
        lattice_cpu = lattice.detach().cpu().to(dtype=torch.float64)
        if int(lattice_cpu.numel()) == 9:
            return rearrange(lattice_cpu, "(i j) -> i j", i=3, j=3).numpy()
        if int(lattice_cpu.numel()) == 6:
            return lattice_cpu.numpy()
        raise ValueError(
            f"Expected 6D or 9D lattice, got {int(lattice_cpu.numel())} values."
        )

    def _build_atoms(
        self,
        cart_coords: Float["n 3"],
        lattice: Float["d"],
        atomic_numbers: Int["n"],
        atom_mask: Bool["n"],
    ) -> Atoms:
        """Build one ASE Atoms object from padded tensors."""
        mask_np = atom_mask.detach().cpu().to(dtype=torch.bool).numpy()
        coords_np = cart_coords.detach().cpu().numpy()
        numbers_np = atomic_numbers.detach().cpu().to(dtype=torch.long).numpy()
        return Atoms(
            numbers=numbers_np[mask_np],
            positions=coords_np[mask_np],
            cell=self._cell_from_lattice(lattice),
            pbc=True,
        )

    def _build_batch(
        self,
        cart_coords: Float["b n 3"],
        lattice: Float["b d"],
        atomic_numbers: Int["b n"],
        atom_mask: Bool["b n"],
    ) -> object:
        """Convert padded tensors to a fairchem AtomicData batch."""
        from fairchem.core.datasets.atomic_data import (
            AtomicData,
            atomicdata_list_to_batch,
        )

        atoms_list = [
            self._build_atoms(
                cart_coords[index],
                lattice[index],
                atomic_numbers[index],
                atom_mask[index],
            )
            for index in range(int(cart_coords.shape[0]))
        ]
        data_list = [
            AtomicData.from_ase(atoms, task_name=self.task_name, r_edges=False)
            for atoms in atoms_list
        ]
        return atomicdata_list_to_batch(data_list).to(self.device)

    @torch.no_grad()
    def _forward_chunk(
        self,
        cart_coords: Float["b n 3"],
        lattice: Float["b d"],
        atomic_numbers: Int["b n"],
        atom_mask: Bool["b n"],
    ) -> Float["b"]:
        """Return per-atom energy for one MLIP batch chunk."""
        batch = self._build_batch(cart_coords, lattice, atomic_numbers, atom_mask)
        output = self.predictor.predict(batch)
        if "energy" not in output:
            raise KeyError(
                f"MLIP output does not contain an energy field: {output.keys()}"
            )
        energy = output["energy"].detach()
        num_atoms = atom_mask.sum(dim=1).to(device=energy.device, dtype=energy.dtype)
        return energy / num_atoms.clamp(min=1.0)

    @torch.no_grad()
    def forward(
        self,
        cart_coords: Float["b n 3"],
        lattice: Float["b d"],
        atomic_numbers: Int["b n"],
        atom_mask: Bool["b n"],
    ) -> Float["b"]:
        """Return per-atom energy for each structure in the batch."""
        if int(cart_coords.shape[0]) <= self.max_batch_size:
            return self._forward_chunk(cart_coords, lattice, atomic_numbers, atom_mask)

        chunk_energies: list[Float["c"]] = []
        for start in range(0, int(cart_coords.shape[0]), self.max_batch_size):
            stop = min(start + self.max_batch_size, int(cart_coords.shape[0]))
            chunk_energies.append(
                self._forward_chunk(
                    cart_coords[start:stop],
                    lattice[start:stop],
                    atomic_numbers[start:stop],
                    atom_mask[start:stop],
                )
            )
        return torch.cat(chunk_energies, dim=0)
