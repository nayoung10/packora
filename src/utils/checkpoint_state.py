"""Utilities for adapting checkpoint state dictionaries."""

from collections.abc import Mapping
from typing import Any

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

_COMPILED_MODULE_NAME = "_orig_mod"


def _remove_compile_wrappers_from_key(key: str) -> str:
    """Return a state key without torch.compile wrapper components."""
    return ".".join(part for part in key.split(".") if part != _COMPILED_MODULE_NAME)


def _insert_compile_wrapper_candidates(key: str) -> list[str]:
    """Return state key candidates with one torch.compile wrapper inserted."""
    parts = key.split(".")
    return [
        ".".join([*parts[:insert_at], _COMPILED_MODULE_NAME, *parts[insert_at:]])
        for insert_at in range(1, len(parts))
    ]


def _resolve_state_dict_key(key: str, target_keys: set[str]) -> str:
    """Resolve one checkpoint key against the target model key namespace."""
    if key in target_keys:
        return key

    candidates = [_remove_compile_wrappers_from_key(key)]
    candidates.extend(_insert_compile_wrapper_candidates(key))
    matches = [candidate for candidate in candidates if candidate in target_keys]
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) > 1:
        raise ValueError(
            f"Ambiguous torch.compile key mapping for {key}: {unique_matches}"
        )
    if unique_matches:
        return unique_matches[0]
    return key


def remap_state_dict_keys(
    state_dict: Mapping[str, Any],
    target_keys: set[str],
) -> dict[str, Any]:
    """Adapt torch.compile state keys to the target model namespace."""
    adapted: dict[str, Any] = {}
    renamed_count = 0
    for key, value in state_dict.items():
        adapted_key = _resolve_state_dict_key(key, target_keys)
        if adapted_key in adapted:
            raise ValueError(
                f"Checkpoint key collision while adapting torch.compile keys: "
                f"{key} -> {adapted_key}"
            )
        adapted[adapted_key] = value
        renamed_count += int(adapted_key != key)
    if renamed_count > 0:
        log.info(
            f"Adapted {renamed_count} checkpoint keys for torch.compile namespace."
        )
    return adapted
