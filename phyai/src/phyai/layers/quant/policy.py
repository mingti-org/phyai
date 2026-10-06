"""Parse load-time quantization rules from a kernel policy document."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.plan import Matcher, QuantPlan, Rule
from phyai.layers.quant.scheme import QDType, QuantScheme, TensorQuant


_TENSOR_FIELDS = frozenset(
    {"dtype", "granularity", "block_shape", "symmetric", "dynamic", "micro_scaled"}
)
_INPUT_DTYPES = frozenset({QDType.INT8, QDType.FP8_E4M3, QDType.FP8_E5M2})
_MICRO_BLOCKS = {QDType.MXFP4: (1, 32), QDType.NVFP4: (1, 16)}


def _mapping(value: object, path: str, allowed: set[str] | frozenset[str]) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} field names must be strings")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {path} field(s): {sorted(unknown)}")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _tensor_quant(value: object, path: str, *, activation: bool = False) -> TensorQuant:
    raw = _mapping(value, path, _TENSOR_FIELDS)
    dtype_name = raw.get("dtype")
    if not isinstance(dtype_name, str):
        raise ValueError(f"{path}.dtype must be a quantized dtype string")
    try:
        dtype = QDType(dtype_name.lower())
    except ValueError:
        raise ValueError(f"unsupported {path}.dtype {dtype_name!r}") from None
    if dtype is QDType.BF16:
        raise ValueError(
            f"{path}.dtype cannot be bf16; use scheme: null to skip a layer"
        )
    if activation and dtype not in _INPUT_DTYPES:
        raise ValueError(f"{path}.dtype must be int8, fp8_e4m3, or fp8_e5m2")

    block = raw.get("block_shape", _MICRO_BLOCKS.get(dtype))
    if block is not None:
        if (
            not isinstance(block, (list, tuple))
            or len(block) != 2
            or any(type(size) is not int or size <= 0 for size in block)
        ):
            raise ValueError(f"{path}.block_shape must contain two positive integers")
        block = tuple(block)
    granularity_name = raw.get(
        "granularity", "block" if block is not None else "per_channel"
    )
    if not isinstance(granularity_name, str):
        raise ValueError(f"{path}.granularity must be a string")
    try:
        granularity = Granularity(granularity_name.lower())
    except ValueError:
        raise ValueError(
            f"{path}.granularity must be per_tensor, per_channel, or block"
        ) from None
    if (granularity is Granularity.BLOCK) != (block is not None):
        raise ValueError(
            f"{path}.block_shape is required exactly for block granularity"
        )

    symmetric = _boolean(raw.get("symmetric", True), f"{path}.symmetric")
    dynamic = _boolean(raw.get("dynamic", activation), f"{path}.dynamic")
    if dynamic != activation:
        expectation = "dynamic" if activation else "static"
        raise ValueError(f"load-time RTN requires {expectation} {path} quantization")
    micro_scaled = _boolean(
        raw.get("micro_scaled", dtype in _MICRO_BLOCKS), f"{path}.micro_scaled"
    )
    if micro_scaled != (dtype in _MICRO_BLOCKS):
        raise ValueError(f"{path}.micro_scaled must be true only for mxfp4 or nvfp4")
    if dtype in _MICRO_BLOCKS and block != _MICRO_BLOCKS[dtype]:
        raise ValueError(
            f"{path}.block_shape for {dtype.value} must be {_MICRO_BLOCKS[dtype]}"
        )
    return TensorQuant(
        dtype=dtype,
        granularity=granularity,
        symmetric=symmetric,
        dynamic=dynamic,
        micro_scaled=micro_scaled,
        block_shape=block,
    )


def _scheme(value: object, path: str) -> QuantScheme | None:
    if value is None:
        return None
    raw = _mapping(value, path, {"scheme", "weight", "input"})
    if "scheme" in raw:
        if raw["scheme"] is not None or set(raw) != {"scheme"}:
            raise ValueError(
                f"{path}.scheme only accepts null, without weight or input"
            )
        return None
    if "weight" not in raw or raw["weight"] is None:
        raise ValueError(f"{path}.weight is required; use scheme: null to skip a layer")
    weight = _tensor_quant(raw["weight"], f"{path}.weight")
    activation = raw.get("input")
    return QuantScheme(
        weight=weight,
        input=(
            _tensor_quant(activation, f"{path}.input", activation=True)
            if activation is not None
            else None
        ),
        online=True,
    )


def quant_plan_from_mapping(value: object) -> QuantPlan | None:
    """Compile optional ``quantization`` into ordered, load-time RTN rules.

    Rules contain one ``match`` selector and either ``weight`` / ``input``
    settings or ``scheme: null`` to retain the layer's original precision.
    An omitted or null section leaves checkpoint quantization unchanged.
    """
    if value is None:
        return None
    raw = _mapping(value, "quantization", {"method", "stage", "rules", "default"})
    if raw.get("method", "rtn") != "rtn":
        raise ValueError("quantization.method must be 'rtn'")
    if raw.get("stage", "load") != "load":
        raise ValueError("quantization.stage must be 'load'")
    raw_rules = raw.get("rules", [])
    if not isinstance(raw_rules, (list, tuple)):
        raise ValueError("quantization.rules must be a list")
    rules = []
    for index, value in enumerate(raw_rules):
        path = f"quantization.rules[{index}]"
        rule = _mapping(value, path, {"match", "scheme", "weight", "input"})
        match = _mapping(
            rule.get("match"), f"{path}.match", {"name", "glob", "regex", "module_cls"}
        )
        if len(match) != 1:
            raise ValueError(f"{path}.match must contain exactly one layer selector")
        kind, pattern = next(iter(match.items()))
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"{path}.match.{kind} must be a nonempty string")
        if kind == "regex":
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"{path}.match.regex is invalid: {error}") from error
        scheme = _scheme(
            {key: val for key, val in rule.items() if key != "match"}, path
        )
        rules.append(Rule(Matcher(kind, pattern), scheme))
    return QuantPlan(
        rules=tuple(rules),
        default=_scheme(raw.get("default"), "quantization.default"),
    )


def quant_plan_payload(plan: QuantPlan) -> dict[str, Any]:
    """Return stable semantic data for policy fingerprints."""

    def tensor_payload(tensor: TensorQuant) -> dict[str, Any]:
        return {
            "dtype": tensor.dtype.value,
            "granularity": tensor.granularity.value,
            "symmetric": tensor.symmetric,
            "dynamic": tensor.dynamic,
            "micro_scaled": tensor.micro_scaled,
            "block_shape": list(tensor.block_shape) if tensor.block_shape else None,
        }

    def scheme_payload(scheme: QuantScheme | None) -> dict[str, Any] | None:
        if scheme is None:
            return None
        return {
            "weight": tensor_payload(scheme.weight),
            "input": tensor_payload(scheme.input) if scheme.input else None,
            "online": scheme.online,
        }

    return {
        "rules": [
            {
                "match": {rule.matcher.kind: rule.matcher.pattern},
                "scheme": scheme_payload(rule.scheme),
            }
            for rule in plan.rules
        ],
        "default": scheme_payload(plan.default),
    }


__all__ = ["quant_plan_from_mapping", "quant_plan_payload"]
