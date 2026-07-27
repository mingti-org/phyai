"""Importer selection: pick the front-end that recognizes a checkpoint.

``build_quant_plan(sources)`` is the single public entry point — hand it
the aggregated :class:`ConfigSources` and it returns a :class:`QuantPlan`
(or ``None`` for an unquantized model). Detection is an explicit ordered
list, not a scattered per-config guess.
"""

from __future__ import annotations

from phyai.layers.quant.importers.base import ConfigSources, QuantImporter
from phyai.layers.quant.importers.compressed_tensors import CompressedTensorsImporter
from phyai.layers.quant.importers.fp8 import Fp8Importer
from phyai.layers.quant.importers.modelopt import ModelOptImporter
from phyai.layers.quant.plan import QuantPlan

DEFAULT_IMPORTERS: tuple[QuantImporter, ...] = (
    Fp8Importer(),
    CompressedTensorsImporter(),
    ModelOptImporter(),
)


def select_importer(
    src: ConfigSources,
    importers: tuple[QuantImporter, ...] = DEFAULT_IMPORTERS,
) -> QuantImporter | None:
    """Return the first importer that recognizes ``src``, else ``None``."""
    for importer in importers:
        if importer.detect(src):
            return importer
    return None


def build_quant_plan(
    src: ConfigSources,
    importers: tuple[QuantImporter, ...] = DEFAULT_IMPORTERS,
) -> QuantPlan | None:
    """Build the QuantPlan for ``src``, or ``None`` if no format matches."""
    importer = select_importer(src, importers)
    return importer.build_plan(src) if importer is not None else None


__all__ = ["DEFAULT_IMPORTERS", "select_importer", "build_quant_plan"]
