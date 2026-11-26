"""CUDA additive-bias attention, backend eligibility, and graph replay."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from phyai.kernel.bootstrap import initialize_kernel_system
from phyai.kernel.config import KernelConfig
from phyai.kernel.selector import NoKernelError
from phyai.layers.attention.mask import AttnMask
from phyai.layers.attention.nocache.layer import Attention
from phyai.layers.attention.nocache.backends.flashinfer import (
    FlashInferAttentionBackend,
)
from phyai.runtime.cuda_graph_manager import CudaGraph


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("same_dtype", [False, True])
def test_additive_bias_with_gqa_and_rectangular_mask(
    dtype, causal, padding, same_dtype
):
    q = torch.randn(2, 5, 4, 64, dtype=dtype, device="cuda")
    k = torch.randn(2, 9, 2, 64, dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    bias = torch.randn(
        1, 4, 5, 9, dtype=dtype if same_dtype else torch.float32, device="cuda"
    )
    lengths = torch.tensor([9, 6], dtype=torch.int32, device="cuda")
    mask = AttnMask.from_lengths(lengths) if padding else None
    layer = Attention(
        4, 64, num_kv_heads=2, scale=1.0, causal=causal, backend="flashinfer"
    )

    actual = layer(q, k, v, mask=mask, attn_bias=bias)
    allowed = torch.ones(2, 1, 5, 9, dtype=torch.bool, device="cuda")
    columns = torch.arange(9, device="cuda")
    if padding:
        allowed &= columns[None, None, None, :] < lengths[:, None, None, None]
    if causal:
        positions = torch.arange(5, device="cuda") + 4
        allowed &= columns[None, :] <= positions[:, None]
    with sdpa_kernel(SDPBackend.MATH):
        expected = (
            F.scaled_dot_product_attention(
                q.float().transpose(1, 2),
                k.float().transpose(1, 2),
                v.float().transpose(1, 2),
                attn_mask=bias.float().masked_fill(~allowed, float("-inf")),
                scale=1.0,
                enable_gqa=True,
            )
            .transpose(1, 2)
            .to(dtype)
        )
    torch.testing.assert_close(
        actual,
        expected,
        atol={torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 2e-3}[dtype],
        rtol=0.02 if dtype == torch.bfloat16 else 1e-3,
    )
    assert torch.isfinite(actual).all()
    assert {backend.name for backend in layer._resolved_backends.values()} == {"sdpa"}


def test_strict_flashinfer_override_rejects_bias():
    initialize_kernel_system(
        KernelConfig(
            policy_config={
                "schema": "phyai.kernel/v1",
                "overrides": [
                    {"match": {"op": "attention"}, "use": "flashinfer.attention.fa2"}
                ],
            }
        )
    )
    layer = Attention(2, 64, causal=False)
    q = torch.randn(1, 7, 2, 64, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(1, 2, 7, 7, device="cuda", dtype=q.dtype)
    with pytest.raises(NoKernelError, match="has_attn_bias"):
        layer(q, q, q, attn_bias=bias)


def test_bias_cannot_be_ignored_by_an_explicit_backend():
    layer = Attention(2, 64, causal=False, backend="sdpa")
    q = torch.randn(1, 3, 2, 64, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(3, 3, device="cuda")
    from phyai.layers.attention.enums import AttnLayout

    ctx = layer._build_default_ctx(
        q, q, q, AttnLayout.PADDED_4D, None, None, attn_bias=bias
    )
    backend = object.__new__(FlashInferAttentionBackend)
    with pytest.raises(ValueError, match="additive logit bias"):
        layer(q, q, q, ctx=replace(ctx, backend=backend))


@pytest.mark.parametrize("kind", ["shape", "dtype", "ragged"])
def test_invalid_bias_is_rejected(kind):
    layer = Attention(2, 64, causal=False)
    q = torch.randn(1, 3, 2, 64, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(3, 3, device="cuda")
    if kind == "shape":
        bias = torch.randn(4, 3, device="cuda")
    elif kind == "dtype":
        bias = bias.long()
    else:
        q = q[0]
    with pytest.raises(ValueError, match="attn_bias"):
        layer(q, q, q, attn_bias=bias)


def test_additive_bias_cuda_graph_reads_new_values():
    layer = Attention(2, 64, scale=1.0, causal=False, backend="sdpa")
    inputs = {
        "q": torch.randn(1, 7, 2, 64, device="cuda", dtype=torch.bfloat16),
        "k": torch.randn(1, 9, 2, 64, device="cuda", dtype=torch.bfloat16),
        "v": torch.randn(1, 9, 2, 64, device="cuda", dtype=torch.bfloat16),
        "attn_bias": torch.randn(1, 2, 7, 9, device="cuda"),
    }
    graph = CudaGraph()
    with torch.inference_mode():
        graph.capture(layer.forward, inputs)
        inputs["q"].normal_()
        inputs["attn_bias"].normal_()
        actual = graph.replay(inputs).clone()
        expected = layer(**inputs)
    torch.testing.assert_close(actual, expected)
    graph.reset()
