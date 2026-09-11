import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

DATASET_STATS_FILENAME = "dataset_stats.json"


def save_dataset_stats(
    data_dir: str,
    dataset_name: str,
    stats: Dict[str, Any],
) -> Path:
    """Write dataset statistics to a JSON sidecar file."""
    out_path = Path(data_dir) / dataset_name / DATASET_STATS_FILENAME
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved dataset stats to %s", out_path)
    return out_path


def load_dataset_stats(
    data_dir: str,
    dataset_name: str,
) -> Dict[str, Any]:
    """Load dataset statistics from a JSON sidecar file."""
    stats_path = Path(data_dir) / dataset_name / DATASET_STATS_FILENAME
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Dataset stats not found at {stats_path}. "
            f"Run parameter extraction for '{dataset_name}' first."
        )
    with open(stats_path) as f:
        return json.load(f)
