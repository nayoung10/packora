from pathlib import Path

from omegaconf import OmegaConf

from src.prediction.artifacts import cleanup_raw_payloads


def test_cleanup_raw_payloads_removes_raw_by_default(tmp_path: Path) -> None:
    """Remove raw prediction payloads unless keep_raw is enabled."""
    raw_dir = tmp_path / "raw" / "rank_0"
    raw_dir.mkdir(parents=True)
    (raw_dir / "batch_000000.pt").write_bytes(b"payload")
    cfg = OmegaConf.create({"output": {}})

    cleanup_raw_payloads(tmp_path, cfg)

    assert not (tmp_path / "raw").exists()


def test_cleanup_raw_payloads_preserves_raw_when_requested(tmp_path: Path) -> None:
    """Preserve raw prediction payloads when keep_raw is enabled."""
    raw_dir = tmp_path / "raw" / "rank_0"
    raw_dir.mkdir(parents=True)
    (raw_dir / "batch_000000.pt").write_bytes(b"payload")
    cfg = OmegaConf.create({"output": {"keep_raw": True}})

    cleanup_raw_payloads(tmp_path, cfg)

    assert raw_dir.is_dir()
