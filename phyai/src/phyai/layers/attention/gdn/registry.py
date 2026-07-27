"""Backend registry for :mod:`phyai.layers.attention.gdn`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

from phyai.layers.attention.gdn.base import GatedDeltaNetBackend


if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


BackendFactory = Callable[["ModelRunner | None"], GatedDeltaNetBackend]
_FactoryT = TypeVar("_FactoryT", bound=BackendFactory)

_BACKENDS: dict[str, BackendFactory] = {}


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def register_backend(name: str) -> Callable[[_FactoryT], _FactoryT]:
    """Register a GDN backend factory under ``name``."""
    canonical = _canonical(name)

    def deco(factory: _FactoryT) -> _FactoryT:
        if isinstance(factory, type) and issubclass(factory, GatedDeltaNetBackend):
            factory.name = canonical
        if canonical in _BACKENDS:
            raise ValueError(
                f"@register_backend({name!r}): already registered in gdn/."
            )
        _BACKENDS[canonical] = factory
        return factory

    return deco


def get_backend_factory(name: str) -> BackendFactory:
    """Look up a GDN backend factory by name."""
    canonical = _canonical(name)
    if canonical not in _BACKENDS:
        raise ValueError(
            f"GatedDeltaNet backend {name!r} is not registered. Available: "
            f"{list_backends()}"
        )
    return _BACKENDS[canonical]


def list_backends() -> list[str]:
    """Return registered GDN backend names."""
    return sorted(_BACKENDS)


__all__ = [
    "BackendFactory",
    "get_backend_factory",
    "list_backends",
    "register_backend",
]
