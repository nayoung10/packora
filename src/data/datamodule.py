from functools import partial
from typing import Any, Callable, Optional

import hydra
from lightning import LightningDataModule
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler

from src.data.components.collate import collate_fn
from src.data.components.samplers import DistributedLengthBucketBatchSampler
from src.data.dataset import MaterialDataset


class MaterialDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        subset: Optional[dict] = None,
        crystal_rotate: bool = False,
        crystal_translate: bool = False,
        center_cart_coords: bool = True,
        template_rotate: bool = False,
        template_translate: bool = False,
        template_torsion_perturb: bool = False,
        template_jitter: bool = False,
        template_jitter_sigma: float = 0.05,
        predict_split: str = "test",
        # Custom sampler hooks
        train_sampler: Optional[Sampler] = None,
        train_batch_sampler: Optional[Sampler] = None,
        train_num_samples: Optional[int] = None,
        sampler: Optional[dict[str, Any]] = None,
        length_bucketed_batches: bool = False,
        length_bucket_size_multiplier: int = 50,
        length_bucket_drop_last: bool = False,
        length_bucket_balance_across_ranks: bool = False,
        train_drop_fields: Optional[list[str]] = None,
        train_pad_to_atom_buckets: Optional[list[int]] = None,
        val_batch_size: Optional[int] = None,
        val_pad_to_atom_buckets: Optional[list[int]] = None,
        train_datasets: Optional[list[dict[str, Any]]] = None,
        val_dataset: Optional[dict[str, Any]] = None,
        effective_batch_size: Optional[int] = None,
        # Positional indexing strategy
        indexing: Optional[dict] = None,
        max_num_atoms: Optional[int] = None,
        conditioning_policy: Optional[dict[str, Any]] = None,
        conditioning_contexts: Optional[dict[str, str]] = None,
    ) -> None:
        """Initialize datamodule with dataset config and loader parameters."""
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Dataset instances created in setup()
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None

    def _spec_value(
        self,
        spec: Optional[dict[str, Any]],
        key: str,
        default: Any,
    ) -> Any:
        """Return a dataset spec value with a fallback default."""
        if spec is None:
            return default
        return spec.get(key, default)

    def _build_material_dataset(
        self,
        spec: Optional[dict[str, Any]],
        split: str,
        train: bool = False,
        include_metadata: bool = False,
    ) -> MaterialDataset:
        """Build one MaterialDataset from the shared config and optional spec."""
        dataset_name = self._spec_value(spec, "dataset_name", self.hparams.dataset_name)
        dataset_split = self._spec_value(spec, "split", split)
        data_dir = self._spec_value(spec, "data_dir", self.hparams.data_dir)
        subset = self._spec_value(spec, "subset", self.hparams.subset)

        return MaterialDataset(
            data_dir=data_dir,
            dataset_name=dataset_name,
            split=dataset_split,
            subset=subset,
            crystal_rotate=self.hparams.crystal_rotate if train else False,
            crystal_translate=self.hparams.crystal_translate if train else False,
            center_cart_coords=self.hparams.center_cart_coords,
            template_rotate=self.hparams.template_rotate if train else False,
            template_translate=self.hparams.template_translate if train else False,
            template_torsion_perturb=(
                self.hparams.template_torsion_perturb if train else False
            ),
            template_jitter=self.hparams.template_jitter if train else False,
            template_jitter_sigma=self.hparams.template_jitter_sigma,
            indexing=self.hparams.indexing,
            max_num_atoms=self.hparams.max_num_atoms,
            include_metadata=include_metadata,
        )

    def _build_train_dataset(self) -> Dataset:
        """Build the configured training dataset, concatenating when requested."""
        train_specs = self.hparams.train_datasets
        if not train_specs:
            return self._build_material_dataset(None, split="train", train=True)

        datasets = [
            self._build_material_dataset(spec, split="train", train=True)
            for spec in train_specs
        ]
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    def _conditioning_context(self, key: str) -> str:
        """Return the policy context selected for one dataloader role."""
        contexts = self.hparams.conditioning_contexts or {}
        if key in contexts:
            return str(contexts[key])
        if key == "test":
            return str(contexts.get("predict", "predict"))
        return key

    def _collate_fn(
        self,
        context_key: str,
        drop_fields: Optional[list[str]] = None,
        pad_to_atom_buckets: Optional[list[int]] = None,
    ) -> Callable[..., Any]:
        """Return a context-aware collate function."""
        kwargs: dict[str, Any] = {"drop_fields": drop_fields}
        if pad_to_atom_buckets is not None:
            kwargs["pad_to_atom_buckets"] = [
                int(bucket) for bucket in pad_to_atom_buckets
            ]
        if self.hparams.conditioning_policy is not None:
            kwargs["conditioning_policy"] = self.hparams.conditioning_policy
            kwargs["conditioning_context"] = self._conditioning_context(context_key)
        return partial(collate_fn, **kwargs)

    def _train_collate_fn(self) -> Callable[..., Any]:
        """Return the training collate function with configured padding."""
        drop_fields = (
            list(self.hparams.train_drop_fields)
            if self.hparams.train_drop_fields
            else None
        )
        pad_to_atom_buckets = self.hparams.train_pad_to_atom_buckets
        if pad_to_atom_buckets is None:
            pad_to_atom_buckets = self._sampler_atom_buckets()
        return self._collate_fn(
            "train_loss",
            drop_fields=drop_fields,
            pad_to_atom_buckets=pad_to_atom_buckets,
        )

    def _sampler_config(self) -> Optional[Any]:
        """Return the configured train batch sampler node."""
        sampler = self.hparams.get("sampler")
        if sampler is None:
            return None
        if isinstance(sampler, dict) and not sampler:
            return None
        return sampler

    def _sampler_atom_buckets(self) -> Optional[list[int]]:
        """Return configured static atom buckets when the sampler provides them."""
        sampler = self._sampler_config()
        if sampler is None or "atom_buckets" not in sampler:
            return None
        return [int(bucket) for bucket in sampler["atom_buckets"]]

    def _uses_datamodule_distributed_sampler(self) -> bool:
        """Return whether the datamodule owns distributed train sharding."""
        return self._sampler_config() is not None or bool(
            self.hparams.length_bucketed_batches
        )

    def _configured_train_batch_sampler(self) -> Optional[Sampler[list[int]]]:
        """Instantiate the configured train batch sampler."""
        sampler = self._sampler_config()
        if sampler is None:
            return None
        return hydra.utils.instantiate(
            sampler,
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            _recursive_=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create dataset instances for the requested stage."""
        if stage in ("fit", None):
            self.train_dataset = self._build_train_dataset()
            self.val_dataset = self._build_material_dataset(
                self.hparams.val_dataset,
                split="val",
                include_metadata=True,
            )

        if stage in ("test", None):
            self.test_dataset = MaterialDataset(
                data_dir=self.hparams.data_dir,
                dataset_name=self.hparams.dataset_name,
                split="test",
                subset=self.hparams.subset,
                center_cart_coords=self.hparams.center_cart_coords,
                indexing=self.hparams.indexing,
            )

        if stage == "predict":
            # predict_split lets the user run prediction on any split
            self.predict_dataset = MaterialDataset(
                data_dir=self.hparams.data_dir,
                dataset_name=self.hparams.dataset_name,
                split=self.hparams.predict_split,
                subset=self.hparams.subset,
                center_cart_coords=self.hparams.center_cart_coords,
                indexing=self.hparams.indexing,
            )

    def _build_dataloader(self, dataset: Dataset, **overrides) -> DataLoader:
        """Build a DataLoader with shared defaults, allowing per-loader overrides."""
        # Common kwargs shared by ALL loaders
        kwargs = dict(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                self.hparams.persistent_workers and self.hparams.num_workers > 0
            ),
            collate_fn=collate_fn,
        )
        # Merge caller-specific overrides (e.g. sampler, batch_sampler, shuffle)
        kwargs.update(overrides)
        # batch_sampler is mutually exclusive with batch_size/shuffle/sampler
        # per PyTorch semantics — strip them when batch_sampler is present
        if "batch_sampler" in kwargs:
            kwargs.pop("batch_size", None)
            kwargs.pop("shuffle", None)
            kwargs.pop("sampler", None)
        return DataLoader(**kwargs)

    def _eval_sampler(self, dataset: Dataset) -> Optional[Sampler]:
        """Return explicit DDP eval sampler when Lightning sampler injection is disabled."""
        if not self._uses_datamodule_distributed_sampler():
            return None
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return None
        return DistributedSampler(dataset, shuffle=False, drop_last=False)

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader with optional custom sampler."""
        if self.train_dataset is None:
            raise ValueError(
                "train_dataset is not initialized. Call setup('fit') first."
            )

        # Case 1: custom batch_sampler overrides batch_size/shuffle/sampler
        if self.hparams.train_batch_sampler is not None:
            return self._build_dataloader(
                self.train_dataset,
                batch_sampler=self.hparams.train_batch_sampler,
                collate_fn=self._train_collate_fn(),
            )

        # Case 2: custom sampler overrides default shuffle/replacement behavior
        if self.hparams.train_sampler is not None:
            return self._build_dataloader(
                self.train_dataset,
                shuffle=False,
                sampler=self.hparams.train_sampler,
                collate_fn=self._train_collate_fn(),
            )

        # Case 3: Hydra-configured batch sampler owns train sharding
        configured_batch_sampler = self._configured_train_batch_sampler()
        if configured_batch_sampler is not None:
            return self._build_dataloader(
                self.train_dataset,
                batch_sampler=configured_batch_sampler,
                collate_fn=self._train_collate_fn(),
            )

        # Case 4: length-bucketed batches reduce padding in quadratic model paths
        if self.hparams.length_bucketed_batches:
            batch_sampler = DistributedLengthBucketBatchSampler(
                dataset=self.train_dataset,
                batch_size=self.hparams.batch_size,
                bucket_size_multiplier=self.hparams.length_bucket_size_multiplier,
                drop_last=self.hparams.length_bucket_drop_last,
                balance_across_ranks=self.hparams.length_bucket_balance_across_ranks,
            )
            return self._build_dataloader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self._train_collate_fn(),
            )

        # Case 5: auto replacement sampling for subsets smaller than batch_size
        num_train_samples = len(self.train_dataset)
        if num_train_samples < self.hparams.batch_size:
            requested = self.hparams.train_num_samples or 10000
            sampler = RandomSampler(
                self.train_dataset,
                replacement=True,
                num_samples=requested,
            )
            return self._build_dataloader(
                self.train_dataset,
                shuffle=False,
                sampler=sampler,
                collate_fn=self._train_collate_fn(),
            )

        # Case 6: default shuffled training
        return self._build_dataloader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self._train_collate_fn(),
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        batch_size = self.hparams.val_batch_size
        if batch_size is None:
            batch_size = self.hparams.batch_size
        return self._build_dataloader(
            self.val_dataset,
            batch_size=int(batch_size),
            sampler=self._eval_sampler(self.val_dataset),
            collate_fn=self._collate_fn(
                "val_loss",
                pad_to_atom_buckets=self.hparams.val_pad_to_atom_buckets,
            ),
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return self._build_dataloader(
            self.test_dataset,
            sampler=self._eval_sampler(self.test_dataset),
            collate_fn=self._collate_fn("test"),
        )

    def predict_dataloader(self) -> DataLoader:
        """Return prediction dataloader for the configured split."""
        return self._build_dataloader(
            self.predict_dataset,
            sampler=self._eval_sampler(self.predict_dataset),
            collate_fn=self._collate_fn("predict"),
        )
