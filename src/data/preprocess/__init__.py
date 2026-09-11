from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Type

from src.data.preprocess.base import BasePreprocessor

if TYPE_CHECKING:
    from src.data.preprocess.csd import (
        CSDFilterConfig as CSDFilterConfig,
        CSDPreprocessor as CSDPreprocessor,
        RDKitTemplateConfig as RDKitTemplateConfig,
    )


def get_preprocessor_cls(dataset_name: str) -> Type[BasePreprocessor]:
    """Return the preprocessor class for a dataset name."""
    if dataset_name == "csd":
        from src.data.preprocess.csd import CSDPreprocessor

        return CSDPreprocessor
    raise KeyError(
        f"No preprocessor registered for '{dataset_name}'. "
        f"Available: {list(PREPROCESSOR_REGISTRY.keys())}"
    )


class _PreprocessorRegistry(Mapping[str, Type[BasePreprocessor]]):
    """Lazy preprocessor registry that avoids optional dataset imports."""

    _names = ("csd",)

    def __getitem__(self, dataset_name: str) -> Type[BasePreprocessor]:
        """Return one registered preprocessor class."""
        if dataset_name not in self._names:
            raise KeyError(dataset_name)
        return get_preprocessor_cls(dataset_name)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered dataset names."""
        return iter(self._names)

    def __len__(self) -> int:
        """Return the number of registered preprocessors."""
        return len(self._names)


PREPROCESSOR_REGISTRY = _PreprocessorRegistry()


def __getattr__(name: str) -> object:
    """Lazily expose optional preprocessor classes."""
    if name == "CSDFilterConfig":
        from src.data.preprocess.csd import CSDFilterConfig

        return CSDFilterConfig
    if name == "RDKitTemplateConfig":
        from src.data.preprocess.csd import RDKitTemplateConfig

        return RDKitTemplateConfig
    if name == "CSDPreprocessor":
        return get_preprocessor_cls("csd")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
