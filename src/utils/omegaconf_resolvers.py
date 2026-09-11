from typing import Any

from omegaconf import OmegaConf


def _eval_arithmetic(expression: str) -> Any:
    """Evaluate a simple arithmetic OmegaConf expression."""
    return eval(expression, {"__builtins__": {}}, {})  # noqa: S307


def register_omegaconf_resolvers() -> None:
    """Register project-wide OmegaConf resolvers."""
    OmegaConf.register_new_resolver("eval", _eval_arithmetic, replace=True)
