"""Intermediate SQLite cache helpers for preprocessing."""

import logging
import pickle
import sqlite3
from concurrent.futures import FIRST_COMPLETED, TimeoutError, wait
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from src.data.preprocess.base import (
    BasePreprocessor,
    ProcessResult,
    PreprocessContext,
    PreprocessingReport,
    RawEntryRef,
    _process_raw_entry,
)
from src.data.preprocess.utils.config import PreprocessCacheConfig
from src.data.types import Material

logger = logging.getLogger(__name__)


def _cache_path(
    data_dir: Path,
    dataset_name: str,
    source_split: str,
    config: PreprocessCacheConfig,
) -> Path:
    """Return intermediate SQLite cache path for one source split."""
    cache_dir = data_dir / dataset_name / config.cache_dir_name
    return cache_dir / f"{source_split}.sqlite3"


def _connect_cache_db(sqlite_path: Path) -> sqlite3.Connection:
    """Open and initialize an intermediate SQLite cache."""
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entries (
            key TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            material_blob BLOB,
            refcode TEXT,
            failure_reason TEXT,
            failure_detail TEXT,
            stage TEXT,
            skipped INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_status ON entries (status)")
    conn.commit()
    return conn


def _completed_cache_keys(
    conn: sqlite3.Connection,
    retry_failed: bool,
) -> set[str]:
    """Return raw entry keys that should be skipped on resume."""
    statuses = (
        ("success", "skipped")
        if retry_failed
        else (
            "success",
            "skipped",
            "failed",
            "timed_out",
        )
    )
    placeholders = ", ".join("?" for _ in statuses)
    rows = conn.execute(
        f"SELECT key FROM entries WHERE status IN ({placeholders})",
        statuses,
    )
    return {str(row[0]) for row in rows}


def _iter_worker_results(
    preprocessor: BasePreprocessor,
    refs: list[RawEntryRef],
    context: PreprocessContext,
    timeout_seconds: float | None,
) -> Iterator[tuple[RawEntryRef, ProcessResult]]:
    """Yield worker results, applying per-entry timeout when configured."""
    from pebble import ProcessPool

    ref_iter = iter(refs)
    max_pending = max(1, preprocessor.n_jobs * 2)
    pending = {}

    with ProcessPool(max_workers=preprocessor.n_jobs) as pool:

        def submit_next() -> bool:
            """Schedule one raw entry if any remain."""
            try:
                ref = next(ref_iter)
            except StopIteration:
                return False
            future = pool.schedule(
                _process_raw_entry,
                args=(preprocessor, ref, context),
                timeout=timeout_seconds,
            )
            pending[future] = ref
            return True

        for _ in range(max_pending):
            if not submit_next():
                break

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                ref = pending.pop(future)
                try:
                    yield future.result()
                except TimeoutError:
                    yield (
                        ref,
                        ProcessResult(
                            material=None,
                            failure_reason="timed_out",
                            skipped=False,
                            refcode=ref.key,
                            failure_detail=(
                                f"Entry exceeded {timeout_seconds:g} second timeout."
                            ),
                            stage="process_raw_entry",
                        ),
                    )
                except Exception as exc:
                    yield (
                        ref,
                        ProcessResult(
                            material=None,
                            failure_reason=exc.__class__.__name__,
                            skipped=False,
                            refcode=ref.key,
                            failure_detail=str(exc),
                            stage="process_raw_entry",
                        ),
                    )
                submit_next()


def _upsert_cache_rows(
    conn: sqlite3.Connection,
    rows: list[tuple[object, ...]],
) -> None:
    """Write processed raw-entry cache rows to SQLite."""
    conn.executemany(
        """
        INSERT INTO entries (
            key,
            status,
            material_blob,
            refcode,
            failure_reason,
            failure_detail,
            stage,
            skipped
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            status = excluded.status,
            material_blob = excluded.material_blob,
            refcode = excluded.refcode,
            failure_reason = excluded.failure_reason,
            failure_detail = excluded.failure_detail,
            stage = excluded.stage,
            skipped = excluded.skipped
        """,
        rows,
    )
    conn.commit()


def cache_source_split(
    preprocessor: BasePreprocessor,
    data_dir: Path,
    dataset_name: str,
    source_split: str,
    config: PreprocessCacheConfig,
) -> None:
    """Populate the resumable intermediate cache for one source split."""
    sqlite_path = _cache_path(data_dir, dataset_name, source_split, config)
    conn = _connect_cache_db(sqlite_path)
    context = PreprocessContext(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split=source_split,
        source_split=source_split,
    )
    refs = preprocessor.iter_raw_entries(context)
    completed = _completed_cache_keys(conn, config.retry_failed)
    pending_refs = [ref for ref in refs if ref.key not in completed]
    logger.info(
        "Intermediate cache %s: completed=%d, pending=%d",
        source_split,
        len(completed),
        len(pending_refs),
    )

    if not pending_refs:
        preprocessor.write_preprocessing_report(
            _cache_preprocessing_report(
                conn,
                data_dir,
                dataset_name,
                source_split,
                len(refs),
            )
        )
        conn.close()
        return

    rows: list[tuple[object, ...]] = []
    completed_since_commit = 0
    desc = f"{dataset_name}/{source_split} cache"
    results = _iter_worker_results(
        preprocessor,
        pending_refs,
        context,
        config.entry_timeout_seconds,
    )

    try:
        for ref, result in tqdm(results, desc=desc, total=len(pending_refs)):
            result_refcode = getattr(result, "refcode", None) or ref.key
            status = (
                "success"
                if result.material is not None
                else "timed_out"
                if result.failure_reason == "timed_out"
                else "skipped"
                if result.skipped
                else "failed"
            )
            material_blob = (
                pickle.dumps(result.material) if result.material is not None else None
            )
            rows.append(
                (
                    ref.key,
                    status,
                    material_blob,
                    str(result_refcode),
                    result.failure_reason,
                    result.failure_detail,
                    result.stage,
                    int(bool(result.skipped)),
                )
            )
            completed_since_commit += 1
            if completed_since_commit >= config.commit_interval:
                _upsert_cache_rows(conn, rows)
                rows = []
                completed_since_commit = 0
    finally:
        if rows:
            _upsert_cache_rows(conn, rows)

        preprocessor.write_preprocessing_report(
            _cache_preprocessing_report(
                conn,
                data_dir,
                dataset_name,
                source_split,
                len(refs),
            )
        )
        conn.close()


def _cache_preprocessing_report(
    conn: sqlite3.Connection,
    data_dir: Path,
    dataset_name: str,
    source_split: str,
    total_input: int,
) -> PreprocessingReport:
    """Build a preprocessing report from intermediate cache statuses."""
    failures: dict[str, list[str]] = {}
    skipped_by_reason: dict[str, int] = {}
    skipped_crystals_by_reason: dict[str, list[str]] = {}
    failure_details: list[dict[str, object]] = []
    skipped_details: list[dict[str, object]] = []
    yielded = 0
    success_count = 0

    rows = conn.execute(
        """
        SELECT
            key,
            status,
            refcode,
            failure_reason,
            failure_detail,
            stage,
            skipped
        FROM entries
        ORDER BY key
        """
    )
    for key, status, refcode, reason, detail_text, stage, skipped in rows:
        result_refcode = str(refcode or key)
        reason = str(reason or "unknown_error")
        detail = {
            "ref_key": key,
            "refcode": result_refcode,
            "reason": reason,
            "stage": stage,
            "detail": detail_text,
            "skipped": bool(skipped),
        }
        if status == "success":
            yielded += 1
            success_count += 1
        elif status == "skipped":
            skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
            skipped_crystals_by_reason.setdefault(reason, []).append(result_refcode)
            skipped_details.append(detail)
        else:
            failures.setdefault(reason, []).append(result_refcode)
            failure_details.append(detail)

    return PreprocessingReport(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split=source_split,
        total_input=total_input,
        yielded=yielded,
        success_count=success_count,
        skipped_by_reason=skipped_by_reason,
        failures=failures,
        skipped_crystals_by_reason=skipped_crystals_by_reason,
        failure_details=failure_details,
        skipped_details=skipped_details,
    )


def iter_cached_materials(
    data_dir: Path,
    dataset_name: str,
    source_split: str,
    config: PreprocessCacheConfig,
) -> Iterator[Material]:
    """Yield successful intermediate Materials from SQLite cache."""
    sqlite_path = _cache_path(data_dir, dataset_name, source_split, config)
    conn = _connect_cache_db(sqlite_path)
    try:
        rows = conn.execute(
            """
            SELECT key, material_blob FROM entries
            WHERE status = 'success'
            ORDER BY key
            """
        )
        for ref_key, material_blob in rows:
            if material_blob is None:
                raise ValueError(
                    f"Cache corruption: success row {ref_key} has no material blob. "
                    f"Delete {sqlite_path.parent} and rerun."
                )
            yield pickle.loads(material_blob)
    finally:
        conn.close()
