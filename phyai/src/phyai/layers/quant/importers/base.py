"""Config front-ends: parse a checkpoint's quant config into a QuantPlan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from phyai.layers.quant.plan import QuantPlan


@dataclass
class ConfigSources:
    """Aggregates every place a quant config may come from."""

    hf_quant_config: dict | None = None  # config.json's quantization_config
    standalone: dict | None = None  # standalone hf_quant_config.json (modelopt)
    torchao_cli: str | None = None  # --torchao-config


@runtime_checkable
class QuantImporter(Protocol):
    name: str

    def detect(self, src: ConfigSources) -> bool: ...

    def build_plan(self, src: ConfigSources) -> QuantPlan: ...


__all__ = ["ConfigSources", "QuantImporter"]
