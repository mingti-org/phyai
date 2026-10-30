"""Capability, fallback and public policy behavior for opt-in SM110 kernels."""

from pathlib import Path

import pytest

from phyai.kernel.facts import Facts
from phyai.kernel.policy import load_policy, policy_from_mapping
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import NoKernelError, Selector
from phyai.kernel.types import KernelQuery, ModelContext

POLICIES = Path(__file__).resolve().parents[3] / "examples/pi05/kernel_policies"


def gemm_facts(**updates):
    values = {
        "op": "gemm",
        "device.vendor": "nvidia",
        "device.arch": "sm110",
        "model.family": "pi05",
        "lib.flashinfer": True,
        "lib.phyai_kernel": True,
        "lib.cutlass": True,
        "quant.format": "bf16",
        "dtype.input": "bf16",
        "dtype.output": "bf16",
        "dtype.weight": "bf16",
        "dtype.bias": "bf16",
        "shape.M": 50,
        "shape.N": 32,
        "shape.K": 1024,
        "attrs.bias": True,
        "attrs.weight_contiguous": True,
        "attrs.weight_aligned": True,
        "attrs.requires_grad": False,
        "attrs.stream_override": False,
        "attrs.execution": "inference",
    }
    values.update(updates)
    return Facts(values=values)


def gemm_query(*, rows=50, n=32, k=1024, attrs=None):
    return KernelQuery.build(
        "gemm",
        model=ModelContext("pi05"),
        dtype={"input": "bf16", "output": "bf16", "weight": "bf16", "bias": "bf16"},
        quant={"format": "bf16"},
        shape={"M": rows, "N": n, "K": k},
        attrs={
            "bias": True,
            "weight_contiguous": True,
            "weight_aligned": True,
            "requires_grad": False,
            "stream_override": False,
            "execution": "inference",
            **(attrs or {}),
        },
    )


@pytest.fixture
def libraries(monkeypatch):
    monkeypatch.setattr(
        "phyai.kernel.selector.library_facts",
        lambda names: {f"lib.{name}": True for name in names},
    )


def test_sm110_eligibility_is_narrow():
    row = build_catalog().get("phyai_kernel.gemm.sm110")
    assert row.when.eval(gemm_facts()) is None
    for field, value in (
        ("device.arch", "sm90"),
        ("model.family", "qwen"),
        ("dtype.input", "fp16"),
        ("shape.M", 51),
        ("attrs.bias", False),
        ("attrs.weight_contiguous", False),
        ("attrs.weight_aligned", False),
        ("dtype.bias", "fp32"),
        ("attrs.requires_grad", True),
        ("attrs.stream_override", True),
        ("attrs.execution", "rollout"),
    ):
        assert row.when.eval(gemm_facts(**{field: value})) is not None, (field, value)
    for field in (
        "attrs.bias",
        "attrs.weight_contiguous",
        "attrs.weight_aligned",
        "attrs.requires_grad",
        "attrs.stream_override",
        "attrs.execution",
    ):
        values = dict(gemm_facts().values)
        values.pop(field)
        assert row.when.eval(Facts(values=values)) is not None, field


def test_actual_gemm_policy_default_prefer_and_strict_override(libraries):
    catalog = build_catalog()
    selector = Selector(catalog, device="nvidia:sm110")
    assert selector.select(gemm_query()).kernel_id == "flashinfer.gemm.bf16"
    policy = load_policy(POLICIES / "thor_bf16_gemm.yaml", catalog)
    selected = Selector(catalog, policy, device="nvidia:sm110")
    assert (
        selected.select(gemm_query(n=1024, k=32)).kernel_id == "phyai_kernel.gemm.sm110"
    )
    assert selected.select(gemm_query()).kernel_id == "flashinfer.gemm.bf16"
    assert selected.select(gemm_query(rows=51)).kernel_id == "flashinfer.gemm.bf16"
    assert (
        selected.select(gemm_query(attrs={"stream_override": True})).kernel_id
        == "flashinfer.gemm.bf16"
    )
    strict = policy_from_mapping(
        {
            "overrides": [
                {
                    "id": "force",
                    "match": {"op": "gemm"},
                    "use": "phyai_kernel.gemm.sm110",
                }
            ]
        },
        catalog,
    )
    with pytest.raises(NoKernelError, match="cannot handle"):
        Selector(catalog, strict, device="nvidia:sm110").select(gemm_query(rows=51))


def test_recommended_gemm_policy_limits_prefix712_reference_fallback(libraries):
    catalog = build_catalog()
    policy = load_policy(POLICIES / "thor_bf16_gemm.yaml", catalog)
    selector = Selector(catalog, policy, device="nvidia:sm110")
    for rows in (528, 560, 624, 712):
        for n, k in ((2048, 2048), (2048, 16384), (2560, 2048), (32768, 2048)):
            query = gemm_query(rows=rows, n=n, k=k, attrs={"bias": False})
            expected = (
                "flashinfer.gemm.bf16"
                if rows == 712 and k == 2048
                else "phyai_kernel.gemm.sm110"
            )
            assert selector.select(query).kernel_id == expected, (rows, n, k)


@pytest.mark.parametrize("filename", ["thor_bf16_gemm.yaml"])
def test_shipped_policies_parse(filename):
    load_policy(POLICIES / filename, build_catalog())
