"""Same-family structure deduplication helpers."""

import logging
import math
from collections.abc import Sequence
from typing import Any

from joblib import Parallel, delayed

from src.data.preprocess.utils.config import DeduplicateConfig
from src.data.preprocess.utils.config import StructureMatcherConfig
from src.data.preprocess.utils.refcode import refcode_family
from src.data.types import Material

logger = logging.getLogger(__name__)


def _pymatgen_structure(material: Material) -> Any:
    """Build a pymatgen Structure from a Material."""
    from pymatgen.core import Lattice, Structure

    species = [
        int(atomic_number) for atomic_number in material.conditioning.atomic_numbers
    ]
    invalid_species = [atomic_number for atomic_number in species if atomic_number <= 0]
    if invalid_species:
        refcode = _material_refcode(material)
        raise ValueError(
            f"Deduplication requires positive atomic numbers for {refcode}; "
            f"found {invalid_species}."
        )

    return Structure(
        Lattice(material.cell),
        species,
        material.cart_coords,
        coords_are_cartesian=True,
    )


def _material_refcode(material: Material) -> str:
    """Return a display refcode for logs and errors."""
    refcode = (material.info or {}).get("csd_refcode")
    return str(refcode) if refcode is not None else "unknown"


def _required_r_factor(material: Material) -> float:
    """Return numeric r_factor or raise a clear deduplication error."""
    refcode = _material_refcode(material)
    value = (material.info or {}).get("r_factor")
    try:
        r_factor = float(value)
    except (TypeError, ValueError) as exc:
        family = refcode_family(material)
        raise ValueError(
            f"Deduplication requires numeric r_factor for {refcode} "
            f"in refcode family {family}."
        ) from exc
    if not math.isfinite(r_factor):
        family = refcode_family(material)
        raise ValueError(
            f"Deduplication requires finite r_factor for {refcode} "
            f"in refcode family {family}."
        )
    return r_factor


def _record_cluster_selection(cluster: list[Material]) -> Material:
    """Record deduplication context and return the lowest-r-factor material."""
    kept = min(cluster, key=_required_r_factor)

    cluster_refcodes = [_material_refcode(m) for m in cluster]
    kept_refcode = _material_refcode(kept)
    kept.info = dict(kept.info or {})
    kept.info["deduplication"] = {
        "was_deduplicated": len(cluster) > 1,
        "matched_refcodes": cluster_refcodes,
        "kept_refcode": kept_refcode,
        "dropped_refcodes": [
            refcode for refcode in cluster_refcodes if refcode != kept_refcode
        ],
        "selection": "lowest_r_factor" if len(cluster) > 1 else "singleton",
        "r_factors": {_material_refcode(m): _required_r_factor(m) for m in cluster},
    }
    return kept


def _deduplicate_duplicate_family(
    family_materials: Sequence[Material],
    matcher_cfg: StructureMatcherConfig,
) -> list[Material]:
    """Deduplicate one non-singleton CSD family."""
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher(
        ltol=matcher_cfg.ltol,
        stol=matcher_cfg.stol,
        angle_tol=matcher_cfg.angle_tol,
    )
    clusters: list[list[tuple[Material, Any]]] = []
    for material in family_materials:
        structure = _pymatgen_structure(material)
        placed = False
        for cluster in clusters:
            _, representative_structure = cluster[0]
            if matcher.fit(representative_structure, structure):
                cluster.append((material, structure))
                placed = True
                break
        if not placed:
            clusters.append([(material, structure)])

    return [
        _record_cluster_selection([material for material, _ in cluster])
        for cluster in clusters
    ]


def _deduplicate_indexed_duplicate_family(
    family_idx: int,
    family_materials: Sequence[Material],
    matcher_cfg: StructureMatcherConfig,
) -> tuple[int, list[Material]]:
    """Deduplicate one indexed non-singleton family for parallel execution."""
    return family_idx, _deduplicate_duplicate_family(family_materials, matcher_cfg)


def deduplicate_same_family(
    materials: list[Material],
    config: DeduplicateConfig,
) -> list[Material]:
    """Deduplicate matching structures within the same refcode family."""
    if not config.enabled:
        return materials

    matcher_cfg = config.structure_matcher

    by_family: dict[str, list[Material]] = {}
    for material in materials:
        family = refcode_family(material)
        by_family.setdefault(family, []).append(material)

    family_results: list[list[Material] | None] = [None] * len(by_family)
    duplicate_jobs: list[tuple[int, list[Material]]] = []
    singleton_family_count = sum(
        1 for family_materials in by_family.values() if len(family_materials) == 1
    )
    duplicate_family_count = len(by_family) - singleton_family_count
    logger.info(
        "Deduplicating %d materials across %d CSD families "
        "(singletons=%d, duplicate_families=%d)",
        len(materials),
        len(by_family),
        singleton_family_count,
        duplicate_family_count,
    )

    for family_idx, family_materials in enumerate(by_family.values()):
        if len(family_materials) == 1:
            family_results[family_idx] = [_record_cluster_selection(family_materials)]
        else:
            duplicate_jobs.append((family_idx, family_materials))

    if duplicate_jobs:
        logger.info(
            "Running StructureMatcher deduplication for %d duplicate families "
            "with n_jobs=%d",
            len(duplicate_jobs),
            config.n_jobs,
        )
        duplicate_results = Parallel(
            n_jobs=config.n_jobs, return_as="generator_unordered"
        )(
            delayed(_deduplicate_indexed_duplicate_family)(
                family_idx,
                family_materials,
                matcher_cfg,
            )
            for family_idx, family_materials in duplicate_jobs
        )
        for completed, (family_idx, kept_materials) in enumerate(
            duplicate_results,
            start=1,
        ):
            family_results[family_idx] = kept_materials
            if completed % 1000 == 0:
                logger.info(
                    "Duplicate-family deduplication progress: %d/%d",
                    completed,
                    len(duplicate_jobs),
                )

    deduplicated: list[Material] = []
    for family_result in family_results:
        if family_result is None:
            raise RuntimeError("Deduplication did not produce a result for one family.")
        deduplicated.extend(family_result)
    return deduplicated
