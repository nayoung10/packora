"""Preprocess CSD entries into one unsplit LMDB dataset."""

# ruff: noqa: E402

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.data.preprocess.base import BasePreprocessor
from src.data.preprocess.utils.cache import (
    cache_source_split,
    iter_cached_materials,
)
from src.data.preprocess.utils.benchmark import (
    BenchmarkCsvConfig,
    ComponentFilterConfig,
    ComponentSkip,
    build_component_index,
    build_family_index,
    extract_benchmark_components_from_csd,
    filter_materials_by_components,
    filter_materials_by_family,
    load_benchmark_refcodes,
    load_ccdc_solvent_smiles,
    records_to_dicts,
)
from src.data.preprocess.utils.config import cache_config, deduplicate_config
from src.data.preprocess.utils.deduplication import deduplicate_same_family
from src.data.preprocess.utils.lmdb import write_lmdb
from src.data.types import Material

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _material_refcode(material: Material) -> str:
    """Return the persisted CSD refcode for one material."""
    refcode = (material.info or {}).get("csd_refcode")
    return str(refcode) if refcode is not None else "unknown"


def _filter_known_atomic_numbers(
    materials: list[Material],
) -> tuple[list[Material], list[str]]:
    """Remove materials with unknown or dummy atomic numbers."""
    kept: list[Material] = []
    excluded_refcodes: list[str] = []
    for material in materials:
        atomic_numbers = np.asarray(material.conditioning.atomic_numbers)
        if bool(np.any(atomic_numbers <= 0)):
            excluded_refcodes.append(_material_refcode(material))
        else:
            kept.append(material)
    return kept, excluded_refcodes


def _summary_counts(materials: list[Material]) -> dict[str, dict[str, int]]:
    """Aggregate CSD preprocessing label counts across materials."""
    bond_types: Counter[str] = Counter()
    atom_chirality: Counter[str] = Counter()
    bond_stereochemistry: Counter[str] = Counter()
    for material in materials:
        summary = (material.info or {}).get("conditioning_summary", {})
        if not isinstance(summary, dict):
            continue
        bond_types.update(summary.get("bond_type_counts", {}))
        atom_chirality.update(summary.get("atom_chirality_counts", {}))
        bond_stereochemistry.update(summary.get("bond_stereochemistry_counts", {}))
    return {
        "bond_type_counts": dict(bond_types),
        "atom_chirality_counts": dict(atom_chirality),
        "bond_stereochemistry_counts": dict(bond_stereochemistry),
    }


def _rdkit_template_summary(materials: list[Material]) -> dict[str, Any]:
    """Summarize RDKit template-coordinate generation outcomes."""
    failure_breakdown: Counter[str] = Counter()
    failed_entries: list[dict[str, object]] = []
    attempted_count = 0
    success_count = 0

    for material in materials:
        info = material.info or {}
        rdkit_template = info["rdkit_template"]
        attempted_count += 1
        if bool(rdkit_template["present"]):
            success_count += 1
            continue

        reason = str(rdkit_template["failure_reason"])
        failure_breakdown[reason] += 1
        failed_entries.append(
            {
                "refcode": _material_refcode(material),
                "reason": reason,
                "detail": rdkit_template["failure_detail"],
                "components": rdkit_template["components"],
            }
        )

    failure_count = attempted_count - success_count
    return {
        "attempted_count": attempted_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_fraction": (
            success_count / attempted_count if attempted_count > 0 else 0.0
        ),
        "failure_fraction": (
            failure_count / attempted_count if attempted_count > 0 else 0.0
        ),
        "failure_breakdown": dict(failure_breakdown),
        "failed_entries": failed_entries,
    }


def _write_json(path: Path, payload: Any) -> None:
    """Write a JSON payload with parent directories created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def _resolve_input_path(path_value: object) -> Path:
    """Resolve a config path relative to the original launch directory."""
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return Path(get_original_cwd()) / path


def _remove_lmdb_if_requested(lmdb_path: Path, overwrite: bool) -> None:
    """Remove an existing LMDB directory when overwriting is enabled."""
    if not lmdb_path.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"LMDB already exists at {lmdb_path}; set overwrite_lmdb=true to replace it."
        )
    shutil.rmtree(lmdb_path)


def _benchmark_csv_configs(cfg: DictConfig) -> list[BenchmarkCsvConfig]:
    """Return benchmark CSV configs with resolved paths."""
    csvs = cfg.get("csvs")
    if csvs is None:
        raise ValueError("benchmark_carving.csvs must be configured.")
    configs: list[BenchmarkCsvConfig] = []
    for item in csvs:
        configs.append(
            BenchmarkCsvConfig(
                name=str(item.name),
                path=_resolve_input_path(item.path),
                refcode_column=str(item.refcode_column),
            )
        )
    return configs


def _component_filter_config(cfg: DictConfig) -> ComponentFilterConfig:
    """Return benchmark component-overlap filter config."""
    return ComponentFilterConfig(
        min_component_heavy_atoms=int(cfg.get("min_component_heavy_atoms", 8)),
        ignore_solvents=bool(cfg.get("ignore_solvents", True)),
    )


def _scoped_skip_dicts(
    benchmark_skips: list[ComponentSkip],
    candidate_skips: list[ComponentSkip],
) -> list[dict[str, Any]]:
    """Return skipped component diagnostics with benchmark/candidate scope."""
    rows: list[dict[str, Any]] = []
    for scope, records in (
        ("benchmark", benchmark_skips),
        ("candidate", candidate_skips),
    ):
        for row in records_to_dicts(records):
            rows.append({"scope": scope, **row})
    return rows


def _resolve_preprocessor(cfg: DictConfig) -> BasePreprocessor:
    """Instantiate the configured CSD preprocessor."""
    preprocessor: BasePreprocessor = instantiate(cfg.preprocessor)
    if cfg.get("n_jobs") is not None:
        preprocessor.n_jobs = int(cfg.n_jobs)
    if cfg.get("max_entries") is not None:
        if not hasattr(preprocessor, "max_entries"):
            raise ValueError(f"{type(preprocessor).__name__} lacks max_entries.")
        preprocessor.max_entries = int(cfg.max_entries)
    return preprocessor


@hydra.main(
    version_base="1.3",
    config_path="../configs/preprocess",
    config_name="csd",
)
def main(cfg: DictConfig) -> None:
    """Run CSD preprocessing from Hydra config."""
    start_time = time.perf_counter()
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    data_dir = output_dir.parent
    dataset_name = output_dir.name
    split = str(cfg.split)
    lmdb_path = output_dir / f"{split}.lmdb"

    preprocessor = _resolve_preprocessor(cfg)
    cfg_cache = cache_config(preprocessor)
    if not cfg_cache.enabled:
        raise ValueError("CSD preprocessing requires cache.enabled=true.")

    cache_source_split(preprocessor, data_dir, dataset_name, split, cfg_cache)
    materials = list(iter_cached_materials(data_dir, dataset_name, split, cfg_cache))
    raw_success_count = len(materials)
    logger.info("Loaded %d successful cached materials.", raw_success_count)
    rdkit_template_summary = _rdkit_template_summary(materials)
    _write_json(output_dir / "rdkit_template_summary.json", rdkit_template_summary)

    materials, invalid_atomic_number_refcodes = _filter_known_atomic_numbers(materials)
    logger.info(
        "Invalid atomic-number exclusion: entries=%d",
        len(invalid_atomic_number_refcodes),
    )

    benchmark_records = []
    benchmark_family_index = {}
    benchmark_components = []
    benchmark_component_index = {}
    benchmark_component_skips = []
    candidate_component_skips = []
    benchmark_family_exclusions = []
    benchmark_component_exclusions = []

    benchmark_cfg = cfg.get("benchmark_carving")
    if benchmark_cfg is not None and bool(benchmark_cfg.get("enabled", True)):
        benchmark_records = load_benchmark_refcodes(
            _benchmark_csv_configs(benchmark_cfg)
        )
        benchmark_family_index = build_family_index(benchmark_records)
        materials, benchmark_family_exclusions = filter_materials_by_family(
            materials,
            benchmark_family_index,
        )
        logger.info(
            "Benchmark family exclusion: families=%d, entries=%d",
            len(benchmark_family_index),
            len(benchmark_family_exclusions),
        )

        component_cfg = benchmark_cfg.get("component_overlap")
        if component_cfg is not None and bool(component_cfg.get("enabled", True)):
            component_filter = _component_filter_config(component_cfg)
            solvent_smiles = (
                load_ccdc_solvent_smiles()
                if component_filter.ignore_solvents
                else set()
            )
            benchmark_components, benchmark_component_skips = (
                extract_benchmark_components_from_csd(
                    benchmark_records,
                    solvent_smiles,
                    component_filter,
                    preprocessor,
                )
            )
            benchmark_component_index = build_component_index(benchmark_components)
            materials, benchmark_component_exclusions, candidate_component_skips = (
                filter_materials_by_components(
                    materials,
                    benchmark_component_index,
                    solvent_smiles,
                    component_filter,
                )
            )
            logger.info(
                "Benchmark component exclusion: smiles=%d, entries=%d",
                len(benchmark_component_index),
                len(benchmark_component_exclusions),
            )

    before_dedup_count = len(materials)
    materials = deduplicate_same_family(materials, deduplicate_config(preprocessor))
    deduplicated_count = before_dedup_count - len(materials)

    _remove_lmdb_if_requested(lmdb_path, bool(cfg.overwrite_lmdb))
    write_count, _ = write_lmdb(iter(materials), lmdb_path)

    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "output_dir": str(output_dir),
        "lmdb_path": str(lmdb_path),
        "split": split,
        "max_entries": cfg.get("max_entries"),
        "n_jobs": preprocessor.n_jobs,
        "elapsed_seconds": elapsed_seconds,
        "raw_success_count": raw_success_count,
        "invalid_atomic_number_excluded_count": len(invalid_atomic_number_refcodes),
        "benchmark_refcode_count": len(benchmark_records),
        "benchmark_family_count": len(benchmark_family_index),
        "benchmark_family_excluded_count": len(benchmark_family_exclusions),
        "benchmark_component_smiles_source": "component.smiles",
        "benchmark_component_smiles_count": len(benchmark_component_index),
        "benchmark_component_excluded_count": len(benchmark_component_exclusions),
        "benchmark_excluded_count": len(benchmark_family_exclusions)
        + len(benchmark_component_exclusions),
        "pre_dedup_count": before_dedup_count,
        "deduplicated_count": deduplicated_count,
        "final_count": write_count,
        "label_counts": _summary_counts(materials),
        "rdkit_template": rdkit_template_summary,
    }

    _write_json(
        output_dir / "benchmark_carving_family_exclusions.json",
        records_to_dicts(benchmark_family_exclusions),
    )
    _write_json(
        output_dir / "benchmark_carving_component_exclusions.json",
        records_to_dicts(benchmark_component_exclusions),
    )
    _write_json(
        output_dir / "benchmark_carving_skipped_components.json",
        _scoped_skip_dicts(benchmark_component_skips, candidate_component_skips),
    )
    _write_json(
        output_dir / "benchmark_carving_summary.json",
        {
            "benchmark_component_smiles_source": "component.smiles",
            "benchmark_refcode_count": len(benchmark_records),
            "benchmark_family_count": len(benchmark_family_index),
            "benchmark_component_smiles_count": len(benchmark_component_index),
            "family_excluded_count": len(benchmark_family_exclusions),
            "component_excluded_count": len(benchmark_component_exclusions),
            "excluded_count": len(benchmark_family_exclusions)
            + len(benchmark_component_exclusions),
            "raw_success_count": raw_success_count,
            "family_excluded_fraction": (
                len(benchmark_family_exclusions) / raw_success_count
                if raw_success_count > 0
                else 0.0
            ),
            "component_excluded_fraction": (
                len(benchmark_component_exclusions) / raw_success_count
                if raw_success_count > 0
                else 0.0
            ),
        },
    )
    _write_json(
        output_dir / "invalid_atomic_number_refcodes.json",
        invalid_atomic_number_refcodes,
    )
    _write_json(output_dir / "preprocessing_summary.json", summary)

    logger.info(
        "Finished CSD preprocessing in %.1f s: raw_success=%d, excluded=%d, "
        "deduplicated=%d, final=%d",
        elapsed_seconds,
        raw_success_count,
        len(benchmark_family_exclusions) + len(benchmark_component_exclusions),
        deduplicated_count,
        write_count,
    )


if __name__ == "__main__":
    main()
