"""ModelOptImporter — NVIDIA ModelOpt fp8/nvfp4 checkpoints 2 QuantPlan.

ModelOpt stores its quant config either in a standalone
``hf_quant_config.json`` (``{"quantization": {"quant_algo": "FP8", ...}}``)
or inline as ``config.json``'s ``quantization_config`` with
``quant_method`` in ``modelopt`` / ``modelopt_fp8`` / ``modelopt_fp4``. In
both shapes a single ``quant_algo`` applies to every linear layer except
``exclude_modules``.
"""

from __future__ import annotations

from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.importers.base import ConfigSources
from phyai.layers.quant.plan import Matcher, QuantPlan, Rule
from phyai.layers.quant.scheme import QDType, QuantScheme, TensorQuant

_MODELOPT_METHODS = {"modelopt", "modelopt_fp8", "modelopt_fp4"}


def _quant_block(src: ConfigSources) -> dict | None:
    """Return the dict holding ``quant_algo``, from either config shape."""
    if src.standalone and isinstance(src.standalone.get("quantization"), dict):
        return src.standalone["quantization"]
    cfg = src.hf_quant_config
    if cfg and cfg.get("quant_method") in _MODELOPT_METHODS:
        return cfg
    return None


def _scheme_for_algo(quant_algo: str) -> QuantScheme:
    algo = quant_algo.upper()
    if "FP4" in algo:  # NVFP4 / FP4
        weight = TensorQuant(QDType.NVFP4, Granularity.BLOCK, micro_scaled=True)
        act = TensorQuant(
            QDType.NVFP4, Granularity.PER_CHANNEL, micro_scaled=True, dynamic=True
        )
        return QuantScheme(weight=weight, input=act)
    if "FP8" in algo:
        weight = TensorQuant(QDType.FP8_E4M3, Granularity.PER_CHANNEL)
        act = TensorQuant(QDType.FP8_E4M3, Granularity.PER_CHANNEL, dynamic=True)
        return QuantScheme(weight=weight, input=act)
    raise NotImplementedError(f"modelopt: unsupported quant_algo {quant_algo!r}")


class ModelOptImporter:
    name = "modelopt"

    def detect(self, src: ConfigSources) -> bool:
        block = _quant_block(src)
        return block is not None and block.get("quant_algo") is not None

    def build_plan(self, src: ConfigSources) -> QuantPlan:
        block = _quant_block(src) or {}
        scheme = _scheme_for_algo(block["quant_algo"])
        excluded = block.get("exclude_modules") or block.get("ignore") or []
        rules = tuple(Rule(Matcher("name", name), None) for name in excluded)
        return QuantPlan(rules=rules, default=scheme)


__all__ = ["ModelOptImporter"]
