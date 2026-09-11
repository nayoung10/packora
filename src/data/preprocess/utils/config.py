"""Configuration normalization helpers for preprocessing."""

from dataclasses import dataclass, field

from src.data.preprocess.base import BasePreprocessor
from src.data.preprocess.utils.lmdb import COMMIT_INTERVAL


@dataclass(frozen=True)
class StructureMatcherConfig:
    """StructureMatcher tolerances for preprocessing deduplication."""

    ltol: float = 0.2
    stol: float = 0.3
    angle_tol: float = 5.0


@dataclass(frozen=True)
class DeduplicateConfig:
    """Configuration for same-family structure deduplication."""

    enabled: bool = False
    n_jobs: int = 1
    structure_matcher: StructureMatcherConfig = field(
        default_factory=StructureMatcherConfig
    )


@dataclass(frozen=True)
class PreprocessCacheConfig:
    """Configuration for resumable intermediate preprocessing cache."""

    enabled: bool = False
    cache_dir_name: str = "_preprocess_cache"
    commit_interval: int = COMMIT_INTERVAL
    retry_failed: bool = False
    entry_timeout_seconds: float | None = None


def deduplicate_config(preprocessor: BasePreprocessor) -> DeduplicateConfig:
    """Return normalized same-family deduplication config."""
    config = getattr(preprocessor, "deduplicate", None)
    if config is None:
        return DeduplicateConfig()
    matcher_config = getattr(config, "structure_matcher", StructureMatcherConfig())
    return DeduplicateConfig(
        enabled=bool(getattr(config, "enabled", False)),
        n_jobs=int(getattr(config, "n_jobs", 1)),
        structure_matcher=StructureMatcherConfig(
            ltol=float(getattr(matcher_config, "ltol", 0.2)),
            stol=float(getattr(matcher_config, "stol", 0.3)),
            angle_tol=float(getattr(matcher_config, "angle_tol", 5.0)),
        ),
    )


def cache_config(preprocessor: BasePreprocessor) -> PreprocessCacheConfig:
    """Return normalized intermediate preprocessing cache config."""
    config = getattr(preprocessor, "cache", None)
    if config is None:
        return PreprocessCacheConfig()
    return PreprocessCacheConfig(
        enabled=bool(getattr(config, "enabled", False)),
        cache_dir_name=str(getattr(config, "cache_dir_name", "_preprocess_cache")),
        commit_interval=int(getattr(config, "commit_interval", COMMIT_INTERVAL)),
        retry_failed=bool(getattr(config, "retry_failed", False)),
        entry_timeout_seconds=(
            None
            if getattr(config, "entry_timeout_seconds", None) is None
            else float(getattr(config, "entry_timeout_seconds"))
        ),
    )
