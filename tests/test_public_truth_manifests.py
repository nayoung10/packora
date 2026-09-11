"""Test polymorph evaluation against the public identifier-only manifests."""

from pathlib import Path

import pytest

# Load RDKit before CCDC to avoid their native-library import-order conflict
pytest.importorskip("rdkit.Chem")
pytest.importorskip("ccdc")

from eval.oxtal import load_truth_map, truth_set_for_refcode  # noqa: E402


def test_default_manifests_and_singleton(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve public manifests at call time and preserve all polymorph aliases."""
    manifests = tmp_path / "csv_manifests"
    manifests.mkdir()
    (manifests / "rigid.csv").write_text(
        "id,split,truth_refcodes\nABCDEF,rigid,ABCDEF;ABCDEF01\n"
    )
    (manifests / "flexible.csv").write_text(
        "id,split,truth_refcodes\nGHIJKL,flexible,\n"
    )
    monkeypatch.setenv("PACKORA_DATA_ROOT", str(tmp_path))
    mapping = load_truth_map()
    assert mapping["ABCDEF01"] == ["ABCDEF", "ABCDEF01"]
    assert mapping["GHIJKL"] == ["GHIJKL"]
    assert truth_set_for_refcode("UVWXYZ", mapping) == ["UVWXYZ"]


def test_missing_manifest_has_download_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail clearly when the external benchmark manifests are unavailable."""
    monkeypatch.setenv("PACKORA_DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="nayoung10/Packora-data"):
        load_truth_map()


def test_explicit_manifest_and_conflicting_groups(tmp_path: Path) -> None:
    """Accept a single manifest and reject contradictory polymorph definitions."""
    manifest = tmp_path / "benchmark.csv"
    manifest.write_text(
        "id,split,truth_refcodes\nABCDEF,test,ABCDEF;ABCDEF01\nABCDEF01,test,ABCDEF01\n"
    )
    with pytest.raises(ValueError, match="Conflicting truth groups"):
        load_truth_map(manifest)
    manifest.write_text("id,split\nABCDEF,test\n")
    assert load_truth_map(manifest) == {"ABCDEF": ["ABCDEF"]}
