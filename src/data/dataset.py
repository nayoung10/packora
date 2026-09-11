import logging
import pickle
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.components.indexing import get_index_builder
from src.data.components.transforms import augment_crystal, augment_template
from src.data.types import Material

logger = logging.getLogger(__name__)


class MaterialDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        split: str,
        subset: Optional[dict[str, Any]] = None,
        crystal_rotate: bool = False,
        crystal_translate: bool = False,
        center_cart_coords: bool = True,
        template_rotate: bool = False,
        template_translate: bool = False,
        template_torsion_perturb: bool = False,
        template_jitter: bool = False,
        template_jitter_sigma: float = 0.05,
        indexing: Optional[dict] = None,
        max_num_atoms: Optional[int] = None,
        include_metadata: bool = False,
    ) -> None:
        """Initialize dataset from an existing LMDB cache."""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset or {}
        self.crystal_rotate = crystal_rotate
        self.crystal_translate = crystal_translate
        self.center_cart_coords = center_cart_coords
        self.template_rotate = template_rotate
        self.template_translate = template_translate
        self.template_torsion_perturb = template_torsion_perturb
        self.template_jitter = template_jitter
        self.template_jitter_sigma = template_jitter_sigma
        self.include_metadata = include_metadata
        self.max_num_atoms = max_num_atoms
        self.index_builder: Optional[Callable] = (
            get_index_builder(indexing) if indexing is not None else None
        )

        # Build LMDB cache path alongside the raw data files
        self.lmdb_path = self.data_dir / dataset_name / f"{split}.lmdb"
        self.num_atoms_cache_path = (
            self.data_dir / dataset_name / f"{split}.num_atoms.npy"
        )
        self.csd_families_cache_path = (
            self.data_dir / dataset_name / f"{split}.csd_families.npy"
        )
        self.num_atoms_by_index: Optional[np.ndarray] = None

        self._require_lmdb_exists()

        # Read dataset length and resolve optional subset selection
        env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False)
        with env.begin() as txn:
            self.num_samples_total = int(txn.get(b"__len__").decode())
            self.selected_indices = self._build_selected_indices(txn)
        env.close()

        self.num_samples: int = len(self.selected_indices)
        self.keys: list[bytes] = [f"{i:08d}".encode() for i in self.selected_indices]
        self._env: Optional[lmdb.Environment] = None

    def _require_lmdb_exists(self) -> None:
        """Raise a clear error when preprocessing has not been run."""
        if self.lmdb_path.is_dir():
            return
        raise FileNotFoundError(
            f"Preprocessed LMDB not found at {self.lmdb_path}. "
            "Run the preprocessing script before training or prediction. "
            "For CSD, run `python scripts/preprocess.py` followed by "
            "`python scripts/split_csd_lmdb.py`."
        )

    def _build_selected_indices(self, txn: Any) -> list[int]:
        """Build selected dataset indices from sample and atom limits."""
        sample_limit = self.subset.get("sample_limit")
        indices_path = self.subset.get("indices_path")

        all_indices = (
            self._load_subset_indices(Path(indices_path))
            if indices_path is not None
            else list(range(self.num_samples_total))
        )
        selected_indices = self._filter_by_max_num_atoms(txn, all_indices)
        if sample_limit is not None:
            selected_indices = selected_indices[:sample_limit]

        return selected_indices

    def _load_subset_indices(self, path: Path) -> list[int]:
        """Load explicit dataset row indices from a JSON, NPY, or text file."""
        if not path.is_file():
            raise FileNotFoundError(f"Subset index file not found: {path}")

        if path.suffix == ".npy":
            values = np.load(path, allow_pickle=False).tolist()
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = (
                payload.get("dataset_indices", payload)
                if isinstance(payload, dict)
                else payload
            )
        else:
            values = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        indices = [int(value) for value in values]
        bad_indices = [
            index for index in indices if index < 0 or index >= self.num_samples_total
        ]
        if bad_indices:
            raise ValueError(
                f"Subset index file contains out-of-range indices: {bad_indices[:5]}"
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"Subset index file contains duplicate indices: {path}")
        return indices

    def _load_or_build_num_atoms_cache(self, txn: Any) -> np.ndarray:
        """Load or build the per-sample atom-count cache."""
        # Load from cache if it exists
        if self.num_atoms_cache_path.exists():
            num_atoms = np.load(self.num_atoms_cache_path, allow_pickle=False)
            if num_atoms.shape == (self.num_samples_total,):
                return num_atoms.astype(np.int64, copy=False)
            logger.warning(
                "Ignoring stale atom-count cache at %s with shape %s",
                self.num_atoms_cache_path,
                num_atoms.shape,
            )

        # Save cache if not found or invalid
        num_atoms = np.empty(self.num_samples_total, dtype=np.int64)
        for idx in range(self.num_samples_total):
            material: Material = pickle.loads(txn.get(f"{idx:08d}".encode()))
            num_atoms[idx] = material.conditioning.atomic_numbers.shape[0]

        np.save(self.num_atoms_cache_path, num_atoms)
        return num_atoms

    def _filter_by_max_num_atoms(self, txn: Any, indices: list[int]) -> list[int]:
        """Filter selected indices by maximum atom count when configured."""
        if self.max_num_atoms is None:
            return indices

        if self.num_atoms_by_index is None:
            self.num_atoms_by_index = self._load_or_build_num_atoms_cache(txn)
        return [
            idx for idx in indices if self.num_atoms_by_index[idx] <= self.max_num_atoms
        ]

    def _get_env(self) -> lmdb.Environment:
        """Lazily open the LMDB environment (fork-safe for DataLoader workers)."""
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._env

    def _build_metadata(self, material: Material) -> Dict[str, Any]:
        """Build per-sample metadata payload used by offline prediction/evaluation."""
        info = material.info or {}
        conditioning = material.conditioning
        num_molecules = (
            int(conditioning.num_molecules)
            if conditioning is not None and hasattr(conditioning, "num_molecules")
            else None
        )
        metadata = {
            "dataset_name": info.get("dataset_name", self.dataset_name),
            "material_id": info.get("material_id"),
            "z_value": info.get("z_value"),
            "spacegroup": info.get("spacegroup", info.get("spacegroup_number")),
            "spacegroup_number": info.get("spacegroup_number"),
            "energy": info.get("energy"),
            "csd_refcode": info.get("csd_refcode"),
            "genarris_step": info.get("genarris_step"),
            "xtal_id": info.get("xtal_id"),
            "num_molecules": num_molecules,
            "csd_component_smiles": info.get("csd_component_smiles"),
            "flexibility": info.get("flexibility"),
        }
        metadata.update(self._build_benchmark_metadata(info))
        return metadata

    def _build_benchmark_metadata(self, info: dict[str, Any]) -> Dict[str, Any]:
        """Build benchmark-specific metadata fields."""
        return {
            "benchmark_name": info.get("benchmark_name"),
            "benchmark_row_index": info.get("benchmark_row_index"),
            "benchmark_refcode": info.get("benchmark_refcode"),
            "benchmark_truth_refcodes": info.get("benchmark_truth_refcodes"),
            "benchmark_truth_group": info.get("benchmark_truth_group"),
            "benchmark_source_csv": info.get("benchmark_source_csv"),
            "benchmark_set": info.get("benchmark_set"),
            "benchmark_label": info.get("benchmark_label"),
            "benchmark_smiles": info.get("benchmark_smiles"),
            "teaching_category": info.get("teaching_category"),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a Material from LMDB and return a dict of model-ready tensors."""
        # Read and deserialize from LMDB
        env = self._get_env()
        key = self.keys[idx]
        with env.begin() as txn:
            data = txn.get(key)
        material: Material = pickle.loads(data)

        num_atoms = material.conditioning.atomic_numbers.shape[0]

        # Convert arrays to tensors
        cart_coords = torch.tensor(material.cart_coords, dtype=torch.float32)
        frac_coords = torch.tensor(material.frac_coords, dtype=torch.float32)
        cell = torch.tensor(material.cell, dtype=torch.float32)

        # Build conditioning tensors
        cond = material.conditioning_tensors()

        # Crystal-aware augmentation (rotation + translation)
        cart_coords, frac_coords, cell = augment_crystal(
            cart_coords,
            frac_coords,
            cell,
            rotate=self.crystal_rotate,
            translate=self.crystal_translate,
            membership=cond.membership,
        )

        # Optionally fix the per-structure translation gauge
        if self.center_cart_coords:
            cart_coords = cart_coords - cart_coords.mean(dim=0)

        output: Dict[str, Any] = {
            "dataset_index": torch.tensor(self.selected_indices[idx], dtype=torch.long),
            "cart_coords": cart_coords,
            "frac_coords": frac_coords,
            "lattice": torch.tensor(material.lattice_parameters, dtype=torch.float32),
            "cell": cell,
            "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
            "conditioning": cond,
        }

        # Compute positional indices on real (unpadded) atoms
        if self.index_builder is not None:
            output["indices"] = self.index_builder(output)

        # Add auxiliary conditioning tensors
        info = material.info or {}
        bond_indices = None
        bond_is_rotatable = None
        if self.template_torsion_perturb:
            bond_indices = torch.from_numpy(
                material.conditioning.bond_indices.astype(np.int64),
            )
            if "bond_is_rotatable" in info:
                bond_is_rotatable = torch.tensor(
                    info["bond_is_rotatable"],
                    dtype=torch.bool,
                )

        cond = replace(
            cond,
            template_coords=augment_template(
                cond.template_coords,
                cond.membership,
                bond_indices=bond_indices,
                bond_is_rotatable=bond_is_rotatable,
                rotate=self.template_rotate,
                translate=self.template_translate,
                torsion_perturb=self.template_torsion_perturb,
                jitter=self.template_jitter,
                jitter_sigma=self.template_jitter_sigma,
            ),
        )
        output["conditioning"] = cond

        # Optional metadata is used by offline prediction/evaluation scripts
        if self.include_metadata:
            output["metadata"] = self._build_metadata(material)

        return output

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getstate__(self) -> Dict[str, Any]:
        """Drop LMDB env handle before pickling (for DataLoader workers)."""
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state and mark LMDB env for lazy re-opening."""
        self.__dict__.update(state)
        self._env = None

    def close(self) -> None:
        """Close the LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
