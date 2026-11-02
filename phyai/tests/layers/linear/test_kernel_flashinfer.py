"""Capability tests for the FlashInfer GEMM entry points.

The capability tests only exercise Python predicates, independent of the
GPU generation present. The numeric test needs sm_100+ and compares
FlashInfer's FP4 GEMM with an explicit dequantized reference.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phyai.layers.linear.backends.flashinfer import gemm_bf16, gemm_nvfp4
from phyai.layers.quant import AllocationRequest, Nvfp4Spec
from phyai.weights.shards import replicated


def _sm() -> int:
    maj, mnr = torch.cuda.get_device_capability()
    return maj * 10 + mnr


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_cublaslt_bias_with_independent_output_dtype(out_dtype):
    torch.manual_seed(72)
    layer = torch.nn.Linear(64, 32, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    actual = gemm_bf16(layer, x, layer.bias, out_dtype=out_dtype, backend="cublaslt")
    expected = F.linear(x.float(), layer.weight.float(), layer.bias.float())
    assert actual.dtype == out_dtype
    tolerance = 1e-2 if out_dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)


def _gemm_eligible(signature: dict, *, device: str, K: int = 128):
    """Kernel ids the catalog considers eligible for one GEMM call."""

    from phyai.kernel.bootstrap import get_kernel_selector
    from phyai.kernel.types import KernelQuery, PhysicalSignature

    selector = get_kernel_selector()
    spec = selector.catalog.op("gemm")
    query = selector.normalize(
        KernelQuery.build(
            "gemm",
            device=device,
            dtype={"input": "bf16", "output": "bf16"},
            quant=PhysicalSignature(**signature),
            shape={"M": 16, "N": 128, "K": K},
        )
    )
    facts = selector.facts_for(query, spec)
    eligible, _ = selector.assess(spec, facts, query.mode)
    return {item.kernel_id for item in eligible}


def _build_layer(spec, N, K, device, weight):
    """Allocate and load one quantized layer, the way a real Linear would."""

    layer = torch.nn.Module()
    layer.spec = spec
    spec.allocate(
        layer,
        AllocationRequest(weight_shape=(N, K), logical_widths=[N], device=device),
    )
    spec.load_weight(layer, weight, None, replicated())
    spec.process_after_loading(layer)
    return layer


NVFP4_128X4 = {
    "format": "nvfp4",
    "layout": "128x4",
    "block_shape": (1, 16),
    "granularity": "block",
    "scale_dtype": "fp8_e4m3",
}


def test_flashinfer_nvfp4_is_sm100_only():
    assert "flashinfer.gemm.nvfp4_128x4" not in _gemm_eligible(
        NVFP4_128X4, device="nvidia:SM90"
    )
    assert "flashinfer.gemm.nvfp4_128x4" in _gemm_eligible(
        NVFP4_128X4, device="nvidia:SM100"
    )


def test_flashinfer_rejects_nvfp4_unaligned_k():
    assert "flashinfer.gemm.nvfp4_128x4" not in _gemm_eligible(
        NVFP4_128X4, device="nvidia:SM100", K=120
    )


def test_flashinfer_numeric_accuracy():
    if _sm() < 100:
        pytest.skip("NVFP4 requires sm_100+ GPU")

    torch.manual_seed(0)
    N, K = 128, 128
    weight = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    x = torch.randn((1, K), dtype=torch.bfloat16, device="cuda")
    y_bf16 = F.linear(x, weight)

    spec = Nvfp4Spec(scale_layout="128x4")
    layer = _build_layer(spec, N, K, "cuda", weight)

    y_flashinfer = gemm_nvfp4(layer, x, None)

    bf16_rel_err = (
        y_flashinfer.float() - y_bf16.float()
    ).norm() / y_bf16.float().norm().clamp_min(1e-8)
    assert bf16_rel_err < 0.15, (
        f"FlashInfer NVFP4 end-to-end relative error {bf16_rel_err:.4f} "
        "against bf16 reference exceeds 15%"
    )


def test_flashinfer_numeric_accuracy_with_bias():
    if _sm() < 100:
        pytest.skip("NVFP4 requires sm_100+ GPU")

    torch.manual_seed(0)
    N, K = 128, 128
    weight = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((N,), dtype=torch.bfloat16, device="cuda")
    x = torch.randn((1, K), dtype=torch.bfloat16, device="cuda")
    y_bf16 = F.linear(x, weight, bias)

    spec = Nvfp4Spec(scale_layout="128x4")
    layer = _build_layer(spec, N, K, "cuda", weight)

    y_flashinfer = gemm_nvfp4(layer, x, bias)

    bf16_rel_err = (
        y_flashinfer.float() - y_bf16.float()
    ).norm() / y_bf16.float().norm().clamp_min(1e-8)
    assert bf16_rel_err < 0.15, (
        f"FlashInfer NVFP4 end-to-end relative error {bf16_rel_err:.4f} "
        "against bf16 reference exceeds 15%"
    )
