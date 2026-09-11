#!/usr/bin/env python
"""Convert a public CSD CSV manifest into model-ready LMDB datasets."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from src.data.preprocess.csd_manifest import build_csd_dataset

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help=(
            "CSV containing id, split, and optional truth_refcodes and "
            "flexibility columns."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dataset directory that will receive one LMDB per split.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=64,
        help="Number of parallel CSD conversion workers (default: 64).",
    )
    parser.add_argument(
        "--entry-timeout-seconds",
        type=float,
        default=180.0,
        help="Whole-entry timeout for both conversion passes (default: 180).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing final LMDB and aligned cache artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Convert every safe split present in one public CSD CSV manifest."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = build_csd_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        entry_timeout_seconds=args.entry_timeout_seconds,
        overwrite=args.overwrite,
    )
    logger.info(
        "Finished CSV conversion: written=%d expected=%d failed=%d",
        summary["total_written"],
        summary["total_expected"],
        summary["total_failed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
