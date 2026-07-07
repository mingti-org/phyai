"""`phyai.layers.attention.ar` — autoregressive LM-side paged attention."""

from __future__ import annotations

from phyai.layers.attention.ar.backends import (
    FlashInferARBackend,
    FlashInferARPlan,
)
from phyai.layers.attention.ar.base import (
    ARAttentionBackend,
    ARAttentionLayerProto,
    ARAttnCtx,
    ARAttnMetadata,
    ARAttnPlanHandle,
)
from phyai.layers.attention.ar.layer import ARAttention
from phyai.layers.attention.ar.registry import (
    BackendFactory,
    get_backend_factory,
    list_backends,
    register_backend,
)


__all__ = [
    "ARAttention",
    "ARAttentionBackend",
    "ARAttentionLayerProto",
    "ARAttnCtx",
    "ARAttnMetadata",
    "ARAttnPlanHandle",
    "BackendFactory",
    "FlashInferARBackend",
    "FlashInferARPlan",
    "get_backend_factory",
    "list_backends",
    "register_backend",
]

# RadixAttentionPlanner / RadixSequence are intentionally NOT in __all__: they
# are lazily re-exported via __getattr__ below (the radix bridge pulls in the
# optional ``phyai-ext`` extra). Keeping them out of __all__ means
# ``from ...ar import *`` does not eagerly resolve them and so does not require
# the extension. Import them explicitly: ``from ...ar import RadixAttentionPlanner``.


def __getattr__(name: str):
    # Lazy re-export: the radix bridge pulls in the optional ``phyai-ext``
    # extra, so importing the base AR package (layers/backends) must not import
    # it eagerly. Resolved on first attribute access.
    # See tests/.../test_radix_planner.py::test_ar_import_does_not_pull_radix_extension.
    if name in ("RadixAttentionPlanner", "RadixSequence"):
        from phyai.layers.attention.ar import radix

        value = getattr(radix, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
