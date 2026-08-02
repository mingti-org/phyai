"""CompressedTensorsImporter — llm-compressor ``config_groups`` 2 QuantPlan."""

from __future__ import annotations

from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.importers.base import ConfigSources
from phyai.layers.quant.plan import Matcher, QuantPlan, Rule
from phyai.layers.quant.scheme import QDType, QuantScheme, TensorQuant

_STRATEGY_TO_GRAN = {
    "tensor": Granularity.PER_TENSOR,
    "channel": Granularity.PER_CHANNEL,
    "block": Granularity.BLOCK,
    "token": Granularity.PER_CHANNEL,  # per-token rowwise activation
    "group": Granularity.PER_CHANNEL,  # no group granularity in the enum yet
}


def _dtype_of(qtype: str, num_bits: int) -> QDType:
    if qtype == "float" and num_bits == 8:
        return QDType.FP8_E4M3
    if qtype == "int" and num_bits == 8:
        return QDType.INT8
    if qtype == "int" and num_bits == 4:
        return QDType.INT4
    raise NotImplementedError(
        f"compressed-tensors: unsupported type={qtype!r} num_bits={num_bits}"
    )


def _to_tensorquant(args: dict) -> TensorQuant:
    strategy = args.get("strategy", "tensor")
    gran = _STRATEGY_TO_GRAN.get(strategy)
    if gran is None:
        raise NotImplementedError(
            f"compressed-tensors: unsupported strategy {strategy!r}"
        )
    block = args.get("block_structure")
    block_shape = (
        (int(block[0]), int(block[1])) if strategy == "block" and block else None
    )
    return TensorQuant(
        dtype=_dtype_of(args.get("type", "int"), int(args.get("num_bits", 8))),
        granularity=gran,
        symmetric=bool(args.get("symmetric", True)),
        dynamic=bool(args.get("dynamic", False)),
        block_shape=block_shape,
    )


def _target_matcher(target: str) -> Matcher:
    if target.startswith("re:"):
        return Matcher("regex", target[3:])
    # A bare CamelCase token (no dot, leading uppercase) is an nn.Module class name.
    if "." not in target and target[:1].isupper():
        return Matcher("module_cls", target)
    return Matcher("name", target)


class CompressedTensorsImporter:
    name = "compressed-tensors"

    def detect(self, src: ConfigSources) -> bool:
        cfg = src.hf_quant_config
        return bool(cfg) and cfg.get("quant_method") == "compressed-tensors"

    def build_plan(self, src: ConfigSources) -> QuantPlan:
        cfg = src.hf_quant_config or {}
        rules: list[Rule] = []
        for name in cfg.get("ignore") or []:
            rules.append(Rule(_target_matcher(name), None))
        for group in (cfg.get("config_groups") or {}).values():
            weight = _to_tensorquant(group["weights"])
            input_args = group.get("input_activations")
            act = _to_tensorquant(input_args) if input_args else None
            scheme = QuantScheme(weight=weight, input=act)
            for target in group.get("targets") or []:
                rules.append(Rule(_target_matcher(target), scheme))
        return QuantPlan(rules=tuple(rules), default=None)


__all__ = ["CompressedTensorsImporter"]
