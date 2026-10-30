"""Numerical tests for the torch GEMM entry points.

The bf16 path runs on CPU; fp8 paths require CUDA ≥ sm89 and gate
accordingly. We compare against the obvious reference implementation
to catch wiring bugs (scale broadcast order, ``weight.t()`` direction,
block-scale expansion).
"""

from __future__ import annotations

import torch
import pytest
import torch.nn.functional as F
from torch import nn
from phyai.layers.quant import AllocationRequest
from phyai.layers.linear.spec import Fp8Spec, Bf16Spec, Nvfp4Spec
from phyai.layers.linear.backend import Granularity
from phyai.layers.linear.backends.torch import (
    _unpack_e2m1,
    gemm_fp8_block,
    _expand_block_scale,
    gemm_fp8_per_tensor,
    gemm_fp8_per_channel,
    gemm_nvfp4_reference,
)


def _sm() -> int:
    maj, mnr = torch.cuda.get_device_capability()
    return maj * 10 + mnr


def _build_layer(spec, *, N, K, device, dtype=torch.bfloat16, bias=False):
    layer = nn.Module()
    layer.spec = spec
    spec.allocate(
        layer,
        AllocationRequest(
            weight_shape=(N, K),
            logical_widths=[N],
            fused_dim=0,
            params_dtype=dtype,
        ),
    )
    layer.weight.data = layer.weight.data.to(device)
    if hasattr(layer, "weight_scale"):
        layer.weight_scale.data = layer.weight_scale.data.to(device)
    if hasattr(layer, "input_scale"):
        layer.input_scale.data = layer.input_scale.data.to(device)
    layer.bias = (
        nn.Parameter(torch.zeros(N, dtype=dtype, device=device), requires_grad=False)
        if bias
        else None
    )
    return layer


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
#
# Which storage formats the torch backend serves, and on what hardware, is now
# declared per catalog row instead of by a ``can_handle`` predicate. Asserting
# it against the catalog also covers the sm thresholds, which used to be
# written out independently in four places.


def _gemm_eligible(signature: dict, *, device: str, N: int = 128, K: int = 128):
    """Kernel ids the catalog considers eligible for one GEMM call."""

    from phyai.kernel.types import KernelQuery, PhysicalSignature
    from phyai.kernel.bootstrap import get_kernel_selector

    selector = get_kernel_selector()
    spec = selector.catalog.op("gemm")
    query = selector.normalize(
        KernelQuery.build(
            "gemm",
            device=device,
            dtype={"input": "bf16", "output": "bf16"},
            quant=PhysicalSignature(**signature),
            shape={"M": 16, "N": N, "K": K},
        )
    )
    facts = selector.facts_for(query, spec)
    eligible, _ = selector.assess(spec, facts, query.mode)
    return {item.kernel_id for item in eligible}


BF16 = {"format": "bf16"}
FP8_PER_TENSOR = {
    "format": "fp8_e4m3",
    "granularity": "per_tensor",
    "scale_dtype": "fp32",
}
NVFP4_LINEAR = {
    "format": "nvfp4",
    "layout": "linear",
    "block_shape": (1, 16),
    "granularity": "block",
    "scale_dtype": "fp8_e4m3",
}
NVFP4_128X4 = {**NVFP4_LINEAR, "layout": "128x4"}


def test_torch_serves_bf16_on_any_device():
    assert "torch.gemm.bf16" in _gemm_eligible(BF16, device="cpu")
    assert "torch.gemm.bf16" in _gemm_eligible(BF16, device="nvidia:SM100")


def test_dense_row_accepts_fp32_router_but_flashinfer_does_not():
    from phyai.kernel.types import KernelQuery, PhysicalSignature
    from phyai.kernel.bootstrap import get_kernel_selector

    selector = get_kernel_selector()
    query = selector.normalize(
        KernelQuery.build(
            "gemm",
            device="nvidia:SM110",
            dtype={"input": "fp32", "output": "fp32", "weight": "bf16"},
            quant=PhysicalSignature(format="bf16"),
            shape={"M": 50, "N": 37, "K": 64},
        )
    )
    facts = selector.facts_for(query, selector.catalog.op("gemm"))
    eligible, _ = selector.assess(selector.catalog.op("gemm"), facts, query.mode)
    kernel_ids = {item.kernel_id for item in eligible}
    assert "torch.gemm.bf16" in kernel_ids
    assert "flashinfer.gemm.bf16" not in kernel_ids


def test_force_torch_policy_selects_fp32_router_backend():
    from phyai.kernel.types import KernelQuery
    from phyai.kernel.policy import policy_from_mapping
    from phyai.kernel.registry import build_catalog
    from phyai.kernel.selector import Selector

    catalog = build_catalog()
    policy = policy_from_mapping(
        {
            "schema": "phyai.kernel/v1",
            "rules": [
                {
                    "id": "force-torch-gemm",
                    "match": {"op": "gemm"},
                    "restrict_to": "torch.gemm.*",
                }
            ],
        },
        catalog,
    )
    selector = Selector(catalog, policy, device="nvidia:SM110")
    router = KernelQuery.build(
        "gemm",
        role="router",
        dtype={"input": "fp32", "output": "fp32", "weight": "bf16"},
        quant={"format": "bf16"},
        shape={"M": 50, "N": 37, "K": 64},
    )
    ordinary = KernelQuery.build(
        "gemm",
        dtype={"input": "bf16", "output": "bf16", "weight": "bf16"},
        quant={"format": "bf16"},
        shape={"M": 50, "N": 37, "K": 64},
    )
    assert selector.explain(router).selected == "torch.gemm.bf16"
    assert selector.explain(ordinary).selected == "torch.gemm.bf16"


def test_torch_fp8_needs_sm89():
    assert "torch.gemm.fp8_per_tensor" not in _gemm_eligible(
        FP8_PER_TENSOR, device="nvidia:SM80"
    )
    assert "torch.gemm.fp8_per_tensor" in _gemm_eligible(
        FP8_PER_TENSOR, device="nvidia:SM89"
    )
    assert "torch.gemm.fp8_per_tensor" in _gemm_eligible(
        FP8_PER_TENSOR, device="nvidia:SM90"
    )


def test_torch_fp8_rejects_unaligned_k():
    """``torch._scaled_mm`` needs K and N divisible by 16."""

    assert "torch.gemm.fp8_per_tensor" not in _gemm_eligible(
        FP8_PER_TENSOR, device="nvidia:SM90", N=16, K=15
    )
    assert "torch.gemm.fp8_per_tensor" in _gemm_eligible(
        FP8_PER_TENSOR, device="nvidia:SM90", N=16, K=16
    )


def test_an_unknown_format_matches_nothing():
    """Each row names its exact format, so an unsupported one is simply unserved.

    Previously the predicate tested ``spec_id.startswith("fp8_")``, which let
    an e5m2 signature through to a kernel that assumes e4m3.
    """

    assert _gemm_eligible({"format": "int4"}, device="nvidia:SM90") == set()
    assert (
        _gemm_eligible(
            {"format": "fp8_e5m2", "granularity": "per_tensor"},
            device="nvidia:SM90",
        )
        == set()
    )


def test_torch_serves_the_linear_nvfp4_scale_layout_only():
    """The 128x4 layout is FlashInfer's; torch has the reference unpacker."""

    assert "torch.gemm.nvfp4_linear" in _gemm_eligible(NVFP4_LINEAR, device="cpu")
    assert "torch.gemm.nvfp4_linear" not in _gemm_eligible(
        NVFP4_128X4, device="nvidia:SM100"
    )


def test_torch_gemm_rows_are_capture_safe():
    from phyai.kernel.bootstrap import get_kernel_selector

    catalog = get_kernel_selector().catalog
    for row in catalog.impls("gemm"):
        if row.kernel_id.startswith("torch."):
            assert row.capture_safe


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def test_torch_bf16_matches_F_linear():
    N, K = 16, 32
    spec = Bf16Spec()
    layer = _build_layer(spec, N=N, K=K, device="cuda", bias=True)
    torch.nn.init.normal_(layer.weight, std=0.02)
    torch.nn.init.normal_(layer.bias, std=0.02)

    x = torch.randn(4, K, dtype=torch.bfloat16, device="cuda")
    y = F.linear(x, layer.weight, layer.bias)
    ref = F.linear(x, layer.weight, layer.bias)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_torch_bf16_preserves_batch_dims():
    N, K = 8, 16
    layer = _build_layer(Bf16Spec(), N=N, K=K, device="cuda")
    torch.nn.init.normal_(layer.weight, std=0.02)
    x = torch.randn(2, 3, K, dtype=torch.bfloat16, device="cuda")
    y = F.linear(x, layer.weight, None)
    assert y.shape == (2, 3, N)


# ---------------------------------------------------------------------------
# fp8 apply — CUDA only
# ---------------------------------------------------------------------------


def test_torch_fp8_per_tensor_close_to_bf16_reference():
    if _sm() < 89:
        pytest.skip("fp8 requires sm≥89")
    N, K = 64, 128
    device = torch.device("cuda")

    # Build a bf16 reference and a fp8 layer with matched weights.
    w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.1

    spec = Fp8Spec(granularity=Granularity.PER_TENSOR)
    layer = _build_layer(spec, N=N, K=K, device=device)
    # Pretend per-tensor scale is 1.0 (amax 1.0), store weight as fp8.
    layer.weight.data = w_bf16.to(torch.float8_e4m3fn)
    # Static input_scale stays 1.0; fan weight_scale out to per-channel.
    spec.process_after_loading(layer)
    assert layer.weight_scale.shape == (N,)

    x = torch.randn(8, K, device=device, dtype=torch.bfloat16) * 0.1
    y = gemm_fp8_per_tensor(layer, x, None)
    assert y.shape == (8, N)
    assert y.dtype == torch.bfloat16

    # Rough equivalence: fp8 round-trip within a few percent of bf16.
    ref = F.linear(x, w_bf16)
    rel_err = (y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    assert rel_err < 0.1, f"fp8 per-tensor rel_err={rel_err.item():.4f}"


def test_torch_fp8_per_channel_close_to_bf16_reference():
    if _sm() < 89:
        pytest.skip("fp8 requires sm≥89")
    N, K = 64, 128
    device = torch.device("cuda")

    w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.1

    spec = Fp8Spec(granularity=Granularity.PER_CHANNEL)
    layer = _build_layer(spec, N=N, K=K, device=device)
    layer.weight.data = w_bf16.to(torch.float8_e4m3fn)
    # weight_scale is already (N,) of ones from allocate.

    x = torch.randn(8, K, device=device, dtype=torch.bfloat16) * 0.1
    y = gemm_fp8_per_channel(layer, x, None)
    assert y.shape == (8, N)

    ref = F.linear(x, w_bf16)
    rel_err = (y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    assert rel_err < 0.15, f"fp8 per-channel rel_err={rel_err.item():.4f}"


def test_torch_fp8_per_tensor_with_bias():
    if _sm() < 89:
        pytest.skip("fp8 requires sm≥89")
    N, K = 32, 64
    device = torch.device("cuda")

    spec = Fp8Spec(granularity=Granularity.PER_TENSOR)
    layer = _build_layer(spec, N=N, K=K, device=device, bias=True)
    layer.weight.data = torch.zeros(N, K, device=device, dtype=torch.float8_e4m3fn)
    layer.bias.data = torch.ones(N, device=device, dtype=torch.bfloat16)
    spec.process_after_loading(layer)

    x = torch.zeros(4, K, device=device, dtype=torch.bfloat16)
    y = gemm_fp8_per_tensor(layer, x, layer.bias)
    # weight is zero ⇒ y equals bias broadcast
    assert torch.allclose(y, layer.bias.expand(4, N))


# ---------------------------------------------------------------------------
# block FP8 reference path — dequant + F.linear
# ---------------------------------------------------------------------------


def test_expand_block_scale_shape_and_values():
    # (N=4, K=4) with block (2, 2) ⇒ 2x2 scale tensor, each block 2x2
    sc = torch.tensor([[0.5, 1.0], [2.0, 4.0]])
    out = _expand_block_scale(sc, (4, 4), (2, 2))
    expected = torch.tensor(
        [
            [0.5, 0.5, 1.0, 1.0],
            [0.5, 0.5, 1.0, 1.0],
            [2.0, 2.0, 4.0, 4.0],
            [2.0, 2.0, 4.0, 4.0],
        ]
    )
    assert torch.equal(out, expected)


def test_torch_fp8_block_reference_numeric():
    # This path uses dequant + F.linear.
    N, K = 8, 8
    spec = Fp8Spec(granularity=Granularity.BLOCK, block_shape=(4, 4))
    layer = _build_layer(spec, N=N, K=K, device="cuda")
    # Use a weight_scale that uniformly multiplies the fp8 weight by 2.
    layer.weight_scale.data = torch.full((2, 2), 2.0, device="cuda")
    # Weight tile = ones in fp8
    layer.weight.data = torch.ones(N, K, dtype=torch.float8_e4m3fn, device="cuda")

    x = torch.ones(4, K, dtype=torch.bfloat16, device="cuda")
    y = gemm_fp8_block(layer, x, None)
    # Dequant weight = 2.0 everywhere ⇒ y = (2.0 * K) broadcast
    expected = torch.full((4, N), 2.0 * K, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(y, expected, atol=0.5, rtol=0.01)


# ---------------------------------------------------------------------------
# nvfp4 reference path — dequant + F.linear
# ---------------------------------------------------------------------------


def test_unpack_e2m1_values():
    packed = torch.tensor([[0x21, 0xB7]], dtype=torch.uint8)
    out = _unpack_e2m1(packed)
    expected = torch.tensor([[0.5, 1.0, 6.0, -1.5]], dtype=torch.float32)
    torch.testing.assert_close(out, expected)


def test_torch_nvfp4_reference_numeric():
    N, K = 2, 16
    spec = Nvfp4Spec(scale_layout="linear")
    layer = _build_layer(spec, N=N, K=K, device="cuda")
    # Low nibble then high nibble: both 0x2 encode E2M1 value 1.0.
    layer.weight.data = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device="cuda")
    layer.weight_scale.data = torch.ones(
        N, K // 16, dtype=torch.float8_e4m3fn, device="cuda"
    )
    layer.weight_global_scale.data.fill_(2.0)

    x = torch.ones(3, K, dtype=torch.bfloat16, device="cuda")
    y = gemm_nvfp4_reference(layer, x, None)
    expected = torch.full((3, N), 2.0 * K, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(y, expected, atol=0.5, rtol=0.01)


def test_torch_nvfp4_linear_quantized_random_accuracy():
    torch.manual_seed(0)
    N, K = 64, 128
    spec = Nvfp4Spec(scale_layout="linear")
    layer = _build_layer(spec, N=N, K=K, device="cuda")

    weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    spec.quantize_loaded_weight(layer, weight)

    x = torch.randn(8, K, dtype=torch.bfloat16, device="cuda")
    y = gemm_nvfp4_reference(layer, x, None)
    ref = F.linear(x, weight)

    rel_err = (y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-8)
    assert rel_err < 0.1, f"NVFP4 linear-layout rel_err={rel_err.item():.4f}"
