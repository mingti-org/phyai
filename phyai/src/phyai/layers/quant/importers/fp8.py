"""Fp8Importer — HF flat fp8 ``quantization_config`` 2 QuantPlan."""

from __future__ import annotations

from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.importers.base import ConfigSources
from phyai.layers.quant.plan import Matcher, QuantPlan, Rule
from phyai.layers.quant.scheme import QDType, QuantScheme, TensorQuant


class Fp8Importer:
    name = "fp8"

    def detect(self, src: ConfigSources) -> bool:
        cfg = src.hf_quant_config
        return bool(cfg) and cfg.get("quant_method") == "fp8"

    def build_plan(self, src: ConfigSources) -> QuantPlan:
        cfg = src.hf_quant_config or {}
        block = cfg.get("weight_block_size")
        if block is not None:
            weight = TensorQuant(
                QDType.FP8_E4M3,
                Granularity.BLOCK,
                block_shape=(int(block[0]), int(block[1])),
            )
        else:
            weight = TensorQuant(QDType.FP8_E4M3, Granularity.PER_CHANNEL)

        dynamic = cfg.get("activation_scheme", "dynamic") != "static"
        act = TensorQuant(QDType.FP8_E4M3, Granularity.PER_CHANNEL, dynamic=dynamic)
        default = QuantScheme(weight=weight, input=act)

        ignored = cfg.get("ignored_layers") or cfg.get("modules_to_not_convert") or []
        rules = tuple(Rule(Matcher("name", name), None) for name in ignored)
        return QuantPlan(rules=rules, default=default)


__all__ = ["Fp8Importer"]
