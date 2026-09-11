"""Verify a quick licensed README workflow and clean up all temporary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def run(
    command: list[str],
    env: dict[str, str],
    records: list[dict[str, Any]],
    timeout: int = 600,
) -> None:
    """Run a bounded subprocess and retain compact evidence instead of artifacts."""
    print("Running:", " ".join(command), flush=True)
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except BaseException:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    records.append(
        {
            "command": command,
            "seconds": round(time.monotonic() - start, 2),
            "returncode": process.returncode,
        }
    )
    if process.returncode:
        print(output[-12000:], flush=True)
        raise RuntimeError(f"Command failed with exit code {process.returncode}")
    print(output[-1200:], flush=True)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the tiny identifier-only inputs used for reconstruction."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "split", "truth_refcodes"])
        writer.writeheader()
        writer.writerows(rows)


def verify(work: Path, authenticated: bool, report: dict[str, Any]) -> None:
    """Exercise reconstruction, training, fine-tuning, pretrained inference, and evaluation."""
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(ROOT),
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PACKORA_DATA_ROOT": str(work / "data"),
            "PACKORA_LOG_ROOT": str(work / "logs"),
            "HF_HOME": str(work / "hf"),
            "HF_XET_CACHE": str(work / "xet"),
            "UV_CACHE_DIR": str(work / "uv"),
            "XDG_CACHE_HOME": str(work / "cache"),
            "TORCH_HOME": str(work / "torch"),
            "TORCHINDUCTOR_CACHE_DIR": str(work / "inductor"),
            "TRITON_CACHE_DIR": str(work / "triton"),
            "CUDA_CACHE_PATH": str(work / "cuda"),
            "MPLCONFIGDIR": str(work / "matplotlib"),
            "WANDB_MODE": "disabled",
            "TMPDIR": str(work),
            "OMP_NUM_THREADS": "2",
        }
    )
    # Keep the licensed dependency stack, but install this checkout in a disposable environment
    records = report["commands"]
    run(
        [
            sys.executable,
            "-m",
            "venv",
            "--without-pip",
            "--system-site-packages",
            str(work / "venv"),
        ],
        env,
        records,
    )
    python = str(work / "venv/bin/python")
    # A venv created from a venv inherits its base, so expose the tested dependency directory explicitly
    import sysconfig

    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), sysconfig.get_paths()["purelib"]])
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--no-deps",
            "-e",
            str(ROOT),
        ],
        env,
        records,
    )
    run(
        [
            python,
            "-c",
            "from pathlib import Path; import src, torch; from ccdc.io import EntryReader; "
            f"assert Path(src.__file__).resolve().parent == Path({str(ROOT / 'src')!r}); "
            "assert torch.cuda.is_available(); print(torch.__version__); print(EntryReader('CSD'))",
        ],
        env,
        records,
    )

    # Download only the published data sidecars and the smaller released model
    from huggingface_hub import HfApi, get_token

    token = None if authenticated else False
    if authenticated:
        saved_token = get_token()
        if saved_token:
            env["HF_TOKEN"] = saved_token
    else:
        env.pop("HF_TOKEN", None)
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    for repo, repo_type, destination, patterns in [
        (
            "nayoung10/Packora-data",
            "dataset",
            work / "data",
            ["csv_manifests/*.csv", "*/dataset_stats.json"],
        ),
        (
            "nayoung10/Packora-ckpt",
            "model",
            work / "models",
            ["packora-m/*", "z_distribution.json"],
        ),
    ]:
        info = HfApi().repo_info(repo, repo_type=repo_type, token=token)
        report.setdefault("artifacts", []).append(
            {"repo": repo, "revision": info.sha, "private": info.private}
        )
        run(
            [
                python,
                "-c",
                "import sys,json; from huggingface_hub import snapshot_download; "
                "snapshot_download(sys.argv[1], repo_type=sys.argv[2], revision=sys.argv[3], "
                "local_dir=sys.argv[4], allow_patterns=json.loads(sys.argv[5]), "
                f"token={token!r})",
                repo,
                repo_type,
                str(info.sha),
                str(destination),
                json.dumps(patterns),
            ],
            env,
            records,
        )

    # Use a few released benchmark identifiers to keep CSD reconstruction bounded
    with (work / "data/csv_manifests/rigid.csv").open() as handle:
        rows = list(csv.DictReader(handle))[:3]
    write_manifest(
        work / "tiny.csv",
        [
            {
                "id": row["id"],
                "split": "train" if index < 2 else "val",
                "truth_refcodes": row["truth_refcodes"],
            }
            for index, row in enumerate(rows)
        ],
    )
    run(
        [
            python,
            "scripts/convert_csv_to_dataset.py",
            "--manifest",
            str(work / "tiny.csv"),
            "--output-dir",
            str(work / "data/smoke"),
            "--n-jobs",
            "1",
            "--entry-timeout-seconds",
            "30",
        ],
        env,
        records,
    )
    reconstruction = json.loads((work / "data/smoke/verification.json").read_text())
    report["reconstruction"] = reconstruction
    if not reconstruction["passed"] or any(
        split["written"] != split["expected"]
        for split in reconstruction["splits"].values()
    ):
        raise RuntimeError(
            "Tiny dataset reconstruction omitted an entry or failed verification"
        )
    run(
        [
            python,
            "scripts/extract_dataset_stats.py",
            "--data_dir",
            str(work / "data"),
            "--dataset_name",
            "smoke",
        ],
        env,
        records,
    )
    common = [
        "trainer=default",
        "trainer.devices=1",
        "trainer.max_epochs=1",
        "++trainer.limit_train_batches=1",
        "++trainer.limit_val_batches=1",
        "++trainer.num_sanity_val_steps=0",
        "trainer.check_val_every_n_epoch=1",
        "data=csd",
        "data.dataset_name=smoke",
        "data.effective_batch_size=1",
        "data.batch_size=1",
        "data.val_batch_size=1",
        "data.max_num_atoms=512",
        "data.sampler=null",
        "data.length_bucketed_batches=false",
        "data.num_workers=0",
        "data.pin_memory=false",
        "data.persistent_workers=false",
        "extras.enforce_tags=false",
        "extras.print_config=false",
        "callbacks.model_checkpoint.every_n_epochs=1",
        f"paths.data_dir={work / 'data'}",
        f"paths.log_dir={work / 'logs'}",
    ]
    run(
        [
            python,
            "src/train.py",
            *common,
            "model.compile_target=null",
            f"paths.output_dir={work / 'train'}",
            f"hydra.run.dir={work / 'train'}",
        ],
        env,
        records,
    )
    checkpoint = work / "train/checkpoints/last.ckpt"
    if not checkpoint.is_file():
        raise RuntimeError("Training did not produce the expected checkpoint")
    run(
        [
            python,
            "src/finetune.py",
            *common,
            f"init_ckpt_path={checkpoint}",
            f"paths.output_dir={work / 'finetune'}",
            f"hydra.run.dir={work / 'finetune'}",
        ],
        env,
        records,
    )

    request = work / "request.json"
    request.write_text(
        json.dumps(
            {
                "model": "packora-m",
                "components": [{"smiles": "N#Cc1ccc(cc1)C#N", "ratio": 1}],
                "z": 1,
            }
        )
    )
    released_checkpoint = work / "models/packora-m/checkpoints/packora-m.ckpt"
    run(
        [
            python,
            "-m",
            "src.prediction.api",
            str(request),
            "--checkpoint",
            str(released_checkpoint),
            "--num-steps",
            "32",
            "--output-dir",
            str(work / "api"),
        ],
        env,
        records,
    )
    run(
        [
            python,
            "-c",
            "import json; from pymatgen.core import Structure; "
            f"s=Structure.from_file({str(work / 'api/packora_prediction.cif')!r}); assert len(s)>0; "
            f"json.load(open({str(work / 'api/packora_prediction.json')!r})); print('CIF and JSON valid')",
        ],
        env,
        records,
    )

    write_manifest(
        work / "benchmark.csv",
        [
            {
                "id": rows[0]["id"],
                "split": "rigid",
                "truth_refcodes": rows[0]["truth_refcodes"],
            }
        ],
    )
    run(
        [
            python,
            "scripts/convert_csv_to_dataset.py",
            "--manifest",
            str(work / "benchmark.csv"),
            "--output-dir",
            str(work / "data/csd_benchmarks"),
            "--n-jobs",
            "1",
            "--entry-timeout-seconds",
            "30",
        ],
        env,
        records,
    )
    run(
        [
            python,
            "src/predict.py",
            f"ckpt_path={released_checkpoint}",
            "source.benchmark=rigid",
            "sampling.samples_per_datapoint=1",
            "sampling.num_steps=32",
            "sampling.batch_size=1",
            "sampling.num_workers=0",
            "trainer=gpu",
            "trainer.devices=1",
            "extras.enforce_tags=false",
            "extras.print_config=false",
            f"paths.output_dir={work / 'predictions'}",
            f"hydra.run.dir={work / 'predictions'}",
            f"paths.data_dir={work / 'data'}",
        ],
        env,
        records,
    )
    run(
        [
            python,
            "src/evaluate.py",
            "--predictions-dir",
            str(work / "predictions"),
            "--protocol",
            "clari",
            "--workers",
            "1",
        ],
        env,
        records,
        timeout=120,
    )
    summary = work / "predictions/eval/clari/summary.json"
    if not summary.is_file():
        raise RuntimeError("Evaluation summary was not produced")
    report["evaluation"] = json.loads(summary.read_text())


def main() -> None:
    """Run the disposable smoke workflow and optionally save a verification report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authenticated",
        action="store_true",
        help="use existing HF credentials for a prerelease check",
    )
    parser.add_argument(
        "--report", type=Path, help="retain only a JSON verification report"
    )
    args = parser.parse_args()
    report: dict[str, Any] = {
        "commands": [],
        "status": "failed",
        "dependency_mode": "reuse installed stack; disposable editable install",
    }
    existing_metadata = set(ROOT.glob("*.egg-info"))
    temporary = ""
    try:
        with tempfile.TemporaryDirectory(prefix="packora-smoke-") as temporary:
            verify(Path(temporary), args.authenticated, report)
            report["status"] = "passed"
    finally:
        for metadata in set(ROOT.glob("*.egg-info")) - existing_metadata:
            shutil.rmtree(metadata)
        report["temporary_artifacts_cleaned"] = (
            not temporary or not Path(temporary).exists()
        )
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("Smoke verification passed; temporary artifacts removed.")


if __name__ == "__main__":
    main()
