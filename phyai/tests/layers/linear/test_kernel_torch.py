"""TorchKernel numerical tests.

The bf16 path runs on CPU; fp8 paths require CUDA ≥ sm89 and gate
accordingly. We compare against the obvious reference implementation
to catch wiring bugs (scale broadcast order, ``weight.t()`` direction,
block-scale expansion).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.layers.linear.backend import Granularity, KernelProbe
from phyai.layers.linear.backends.torch import (
    TorchKernel,
    _expand_block_scale,
    _unpack_e2m1,
)
from phyai.layers.linear.spec import Bf16Spec, Fp8Spec, Nvfp4Spec
from phyai.layers.quant import AllocationRequest
from phyai.parallel.state import Mode


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _sm() -> int:
    if not torch.cuda.is_available():
        return 0
    maj, mnr = torch.cuda.get_device_capability()
    return maj * 10 + mnr


def _probe(spec_id: str, *, M=16, N=128, K=128, sm=90) -> KernelProbe:
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
# can_handle
# ---------------------------------------------------------------------------


def test_torch_can_handle_bf16_always():
    k = TorchKernel()
    assert k.can_handle(_probe("bf16", sm=0))
    assert k.can_handle(_probe("bf16", sm=100))


def test_torch_can_handle_fp8_needs_sm89():
    k = TorchKernel()
    assert not k.can_handle(_probe("fp8_per_tensor", sm=80))
    assert k.can_handle(_probe("fp8_per_tensor", sm=89))
    assert k.can_handle(_probe("fp8_per_tensor", sm=90))


def test_torch_can_handle_fp8_rejects_unaligned_K():
    k = TorchKernel()
    # K=15 not divisible by 16 -> reject fp8
    assert not k.can_handle(_probe("fp8_per_tensor", N=16, K=15, sm=90))
    assert k.can_handle(_probe("fp8_per_tensor", N=16, K=16, sm=90))


def test_torch_can_handle_unknown_spec_rejects():
    k = TorchKernel()
    assert not k.can_handle(_probe("awq"))


def test_torch_can_handle_nvfp4_linear_reference():
    k = TorchKernel()
    assert k.can_handle(_probe("nvfp4_block_16_linear", sm=0, N=8, K=16))
    assert not k.can_handle(_probe("nvfp4_block_16_128x4", sm=100, N=128, K=128))


def test_torch_supports_capture():
    assert TorchKernel().supports_capture() is True


# ---------------------------------------------------------------------------
# bf16 apply numerical (CPU OK)
# ---------------------------------------------------------------------------


def test_torch_bf16_matches_F_linear_cpu():
    N, K = 16, 32
    spec = Bf16Spec()
    layer = _build_layer(spec, N=N, K=K, device="cpu", bias=True)
    torch.nn.init.normal_(layer.weight, std=0.02)
    torch.nn.init.normal_(layer.bias, std=0.02)

    x = torch.randn(4, K, dtype=torch.bfloat16)
    y = TorchKernel().apply(layer, x, layer.bias)
    ref = F.linear(x, layer.weight, layer.bias)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_torch_bf16_preserves_batch_dims():
    N, K = 8, 16
    layer = _build_layer(Bf16Spec(), N=N, K=K, device="cpu")
    torch.nn.init.normal_(layer.weight, std=0.02)
    x = torch.randn(2, 3, K, dtype=torch.bfloat16)
    y = TorchKernel().apply(layer, x, None)
    assert y.shape == (2, 3, N)


# ---------------------------------------------------------------------------
# fp8 apply — CUDA only
# ---------------------------------------------------------------------------


@CUDA
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
    y = TorchKernel().apply(layer, x, None)
    assert y.shape == (8, N)
    assert y.dtype == torch.bfloat16

    # Rough equivalence: fp8 round-trip within a few percent of bf16.
    ref = F.linear(x, w_bf16)
    rel_err = (y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    assert rel_err < 0.1, f"fp8 per-tensor rel_err={rel_err.item():.4f}"


@CUDA
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
    y = TorchKernel().apply(layer, x, None)
    assert y.shape == (8, N)

    ref = F.linear(x, w_bf16)
    rel_err = (y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    assert rel_err < 0.15, f"fp8 per-channel rel_err={rel_err.item():.4f}"


@CUDA
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
    y = TorchKernel().apply(layer, x, layer.bias)
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
    # This path uses dequant + F.linear; works on CPU with fp8 dtype.
    N, K = 8, 8
    spec = Fp8Spec(granularity=Granularity.BLOCK, block_shape=(4, 4))
    layer = _build_layer(spec, N=N, K=K, device="cpu")
    # Use a weight_scale that uniformly multiplies the fp8 weight by 2.
    layer.weight_scale.data = torch.full((2, 2), 2.0)
    # Weight tile = ones in fp8
    layer.weight.data = torch.ones(N, K, dtype=torch.float8_e4m3fn)

    x = torch.ones(4, K, dtype=torch.bfloat16)
    y = TorchKernel().apply(layer, x, None)
    # Dequant weight = 2.0 everywhere ⇒ y = (2.0 * K) broadcast
    expected = torch.full((4, N), 2.0 * K, dtype=torch.bfloat16)
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
    layer = _build_layer(spec, N=N, K=K, device="cpu")
    # Low nibble then high nibble: both 0x2 encode E2M1 value 1.0.
    layer.weight.data = torch.full((N, K // 2), 0x22, dtype=torch.uint8)
    layer.weight_scale.data = torch.ones(N, K // 16, dtype=torch.float8_e4m3fn)
    layer.weight_global_scale.data.fill_(2.0)

    x = torch.ones(3, K, dtype=torch.bfloat16)
    y = TorchKernel().apply(layer, x, None)
    expected = torch.full((3, N), 2.0 * K, dtype=torch.bfloat16)
    torch.testing.assert_close(y, expected, atol=0.5, rtol=0.01)


def test_torch_nvfp4_linear_quantized_random_accuracy():
    torch.manual_seed(0)
    N, K = 64, 128
    spec = Nvfp4Spec(scale_layout="linear")
    layer = _build_layer(spec, N=N, K=K, device="cpu")

    weight = torch.randn(N, K, dtype=torch.bfloat16)
    spec.quantize_loaded_weight(layer, weight)

    x = torch.randn(8, K, dtype=torch.bfloat16)
    y = TorchKernel().apply(layer, x, None)
    ref = F.linear(x, weight)

    rel_err = (y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-8)
    assert rel_err < 0.1, f"NVFP4 linear-layout rel_err={rel_err.item():.4f}"
