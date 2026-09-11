"""LMDB writing helpers for preprocessing outputs."""

import logging
import pickle
from pathlib import Path
from typing import Iterator

import lmdb

from src.data.types import Material

logger = logging.getLogger(__name__)

# LMDB virtual address space reservation (1 TB); actual file grows on demand
LMDB_MAP_SIZE = 1 << 40

# Number of entries to process before committing an LMDB write transaction
COMMIT_INTERVAL = 10000


def write_lmdb(
    materials: Iterator[Material],
    lmdb_path: Path,
) -> tuple[int, int]:
    """Write an iterator of Materials to an LMDB database."""
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), map_size=LMDB_MAP_SIZE)
    txn = env.begin(write=True)

    write_count = 0
    total_count = 0

    for material in materials:
        key = f"{write_count:08d}".encode()
        value = pickle.dumps(material)
        txn.put(key, value)
        write_count += 1
        total_count += 1

        # Commit periodically to limit transaction memory
        if write_count % COMMIT_INTERVAL == 0:
            txn.commit()
            txn = env.begin(write=True)

    # Store total count as metadata and finalize
    txn.put(b"__len__", str(write_count).encode())
    txn.commit()
    env.close()

    logger.info(
        "Wrote %d entries to %s",
        write_count,
        lmdb_path,
    )
    return write_count, total_count
