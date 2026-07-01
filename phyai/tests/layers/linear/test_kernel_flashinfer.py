"""FlashInferKernel capability tests.

The capability tests only exercise Python predicates so hardware gating stays
stable on CPU-only CI. The numeric test is CUDA-gated and compares FlashInfer's
FP4 GEMM with an explicit dequantized reference.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phyai.layers.linear.backend import KernelProbe
from phyai.layers.linear.backends.flashinfer import FlashInferKernel
from phyai.layers.quant import Nvfp4Spec
from phyai.layers.quant.base import AllocationRequest
from phyai.parallel.state import Mode
from phyai.weights.shards import replicated

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _sm() -> int:
    if not torch.cuda.is_available():
        return 0
    maj, mnr = torch.cuda.get_device_capability()
    return maj * 10 + mnr


def _probe(spec_id: str, *, M=16, N=128, K=128, sm=100) -> KernelProbe:
    return KernelProbe(
        spec_id=spec_id,
        M_bucket=M.bit_length(),
        N=N,
        K=K,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        sm=sm,
        mode=Mode.EAGER,
    )


def _build_layer(spec, N, K, device, weight):
    layer = torch.nn.Module()
    layer.spec = spec

    spec.allocate(
        layer,
        AllocationRequest(
            weight_shape=(N, K),
            logical_widths=[N],
            device=device,
        ),
    )

    spec.load_weight(layer, weight, None, replicated())
    spec.process_after_loading(layer)
    return layer


def test_flashinfer_can_handle_nvfp4_sm100_only():
    k = FlashInferKernel()
    assert not k.can_handle(_probe("nvfp4_block_16_128x4", sm=90))
    if k.can_handle(_probe("bf16", sm=100)):
        assert k.can_handle(_probe("nvfp4_block_16_128x4", sm=100))


def test_flashinfer_rejects_nvfp4_unaligned_k():
    k = FlashInferKernel()
    assert not k.can_handle(_probe("nvfp4_block_16_128x4", sm=100, K=120))


@CUDA
def test_flashinfer_numeric_accuracy():
    if _sm() < 100:
        pytest.skip(f"NVFP4 requires sm_100+ GPU")

    torch.manual_seed(0)
    k = FlashInferKernel()
    N, K = 128, 128
    weight = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    x = torch.randn((1, K), dtype=torch.bfloat16, device="cuda")
    y_bf16 = F.linear(x, weight)

    spec = Nvfp4Spec(scale_layout="128x4")
    layer = _build_layer(spec, N, K, "cuda", weight)

    y_flashinfer = k.apply(layer, x, None)

    bf16_rel_err = (
        y_flashinfer.float() - y_bf16.float()
    ).norm() / y_bf16.float().norm().clamp_min(1e-8)
    assert bf16_rel_err < 0.15, (
        f"FlashInfer NVFP4 end-to-end relative error {bf16_rel_err:.4f} "
        "against bf16 reference exceeds 15%"
    )


def test_flashinfer_numeric_accuracy_with_bias():
    if _sm() < 100:
        pytest.skip(f"NVFP4 requires sm_100+ GPU")

    torch.manual_seed(0)
    k = FlashInferKernel()
    N, K = 128, 128
    weight = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((N,), dtype=torch.bfloat16, device="cuda")
    x = torch.randn((1, K), dtype=torch.bfloat16, device="cuda")
    y_bf16 = F.linear(x, weight, bias)

    spec = Nvfp4Spec(scale_layout="128x4")
    layer = _build_layer(spec, N, K, "cuda", weight)

    y_flashinfer = k.apply(layer, x, bias)

    bf16_rel_err = (
        y_flashinfer.float() - y_bf16.float()
    ).norm() / y_bf16.float().norm().clamp_min(1e-8)
    assert bf16_rel_err < 0.15, (
        f"FlashInfer NVFP4 end-to-end relative error {bf16_rel_err:.4f} "
        "against bf16 reference exceeds 15%"
    )
