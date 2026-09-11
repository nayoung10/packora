"""Command-line interface for offline Packora structure prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.prediction.api import PackoraPredictor
from src.prediction.api.schemas import PredictionRequest


def build_parser() -> argparse.ArgumentParser:
    """Build the offline prediction argument parser."""
    parser = argparse.ArgumentParser(
        prog="packora-predict",
        description="Predict one molecular crystal structure from a JSON request.",
    )
    parser.add_argument("request", type=Path, help="JSON request file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("predictions"),
        help="directory for CIF and JSON outputs (default: predictions)",
    )
    parser.add_argument("--stem", default="packora_prediction", help="output stem")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument("--device", help="CUDA device, for example cuda:0")
    parser.add_argument("--num-steps", type=int, help="flow sampling steps")
    parser.add_argument("--max-atoms", type=int, help="expanded input atom cap")
    parser.add_argument("--model-manifest", type=Path, help="model manifest JSON")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="checkpoint override for the model selected in the request",
    )
    parser.add_argument("--z-prior", type=Path, help="empirical Z-prior JSON")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing CIF and JSON outputs",
    )
    return parser


def _read_request(path: Path) -> PredictionRequest:
    """Read and validate one JSON prediction request."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PredictionRequest.model_validate(payload)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one offline prediction and write its CIF and JSON outputs."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = _read_request(args.request)
        checkpoint_paths = (
            None if args.checkpoint is None else {request.model: args.checkpoint}
        )
        predictor = PackoraPredictor(
            device=args.device,
            max_atoms=args.max_atoms,
            num_steps=args.num_steps,
            model_manifest_path=args.model_manifest,
            checkpoint_paths=checkpoint_paths,
            z_prior_path=args.z_prior,
        )
        result = predictor.predict(request, seed=args.seed)
        paths = result.write(
            args.output_dir,
            stem=args.stem,
            overwrite=args.overwrite,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")
    print(
        json.dumps(
            {
                "cif": str(paths.cif),
                "json": str(paths.json),
                "summary": result.summary,
            },
            indent=2,
        )
    )
    return 0
