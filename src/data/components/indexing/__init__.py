from functools import partial
from typing import Callable

from src.data.components.indexing.atomic_number import (
    atomic_number_random,
    atomic_number_xyz,
)
from src.data.components.indexing.coordinate import xyz
from src.data.components.indexing.sequential import sequential

INDEX_BUILDERS: dict[str, Callable] = {
    "sequential": sequential,
    "atomic_number_random": atomic_number_random,
    "atomic_number_xyz": atomic_number_xyz,
    "xyz": xyz,
}


def get_index_builder(cfg: dict) -> Callable:
    """Build an index builder from config, binding any extra kwargs via partial."""
    name = cfg["index_type"]
    if name not in INDEX_BUILDERS:
        available = sorted(INDEX_BUILDERS.keys())
        raise KeyError(
            f"No index builder registered for '{name}'. "
            f"Available: {available}"
        )
    builder = INDEX_BUILDERS[name]

    # Bind config kwargs (e.g. descending) excluding index_type
    kwargs = {k: v for k, v in cfg.items() if k != "index_type"}
    if kwargs:
        builder = partial(builder, **kwargs)

    return builder
