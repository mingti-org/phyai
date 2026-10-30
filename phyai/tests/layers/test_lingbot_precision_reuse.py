"""Contracts for the shared Layer and GEMM paths used by LingBot V2."""

import torch
import pytest
import torch.nn.functional as F
from phyai.layers.quant import Bf16Spec
from phyai.layers.linear import ReplicatedLinear
from phyai.layers.layer_norm import RMSNorm, LayerNorm
from phyai.layers.backbones.qwen3_vl.modeling import (
    Qwen3VLVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)


@pytest.mark.parametrize("width", [64, 128])
def test_explicit_fp32_norms_match_frozen_lingbot_math(width):
    torch.manual_seed(width)
    x = torch.randn(3, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(width, device="cuda", dtype=torch.bfloat16)

    rms = RMSNorm(
        width,
        dtype=torch.bfloat16,
        device="cuda",
        cast_before_affine=True,
        explicit_fp32_math=True,
        prefix="text.norm",
    )
    rms.weight.data.copy_(weight)
    rms_expected = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + 1e-6
    )
    rms_expected = weight * rms_expected.to(x.dtype)
    torch.testing.assert_close(rms(x), rms_expected, atol=0, rtol=0)
    assert rms.weight.hf_keys == [("text.norm.weight", None)]

    layer = LayerNorm(
        width,
        eps=1e-6,
        dtype=torch.bfloat16,
        device="cuda",
        explicit_fp32_math=True,
        prefix="vision.norm",
    )
    layer.weight.data.copy_(weight)
    layer.bias.data.copy_(bias)
    centered = x.float() - x.float().mean(-1, keepdim=True)
    normalized = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-6)
    expected = (normalized * weight.float() + bias.float()).to(x.dtype)
    torch.testing.assert_close(layer(x), expected, atol=0, rtol=0)
    assert layer.weight.dtype == layer.bias.dtype == torch.bfloat16
    assert layer.weight.hf_keys == [("vision.norm.weight", None)]
    assert layer.bias.hf_keys == [("vision.norm.bias", None)]


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("tokens", [1, 3, 17])
def test_router_shared_linear_preserves_fp32_math(fake_mesh, bias, tokens):
    fake_mesh()
    torch.manual_seed(tokens)
    layer = ReplicatedLinear(
        64,
        37,
        bias=bias,
        params_dtype=torch.bfloat16,
        spec=Bf16Spec(),
        device="cuda",
        prefix="moe.gate",
    )
    layer.weight.data.normal_()
    if bias:
        layer.bias.data.normal_()
    x = torch.randn(tokens, 64, device="cuda", dtype=torch.bfloat16)
    with torch.amp.autocast("cuda", enabled=False):
        expected = F.linear(
            x.float(),
            layer.weight.float(),
            None if layer.bias is None else layer.bias.float(),
        )
    actual, deferred_bias = layer(x.float())
    assert deferred_bias is None
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert layer.weight.hf_keys == [("moe.gate.weight", None)]


def test_vision_rope_preserves_bf16_frequency_rounding():
    dim = 64
    rope = Qwen3VLVisionRotaryEmbedding(dim, dtype=torch.bfloat16, device="cuda")
    original = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.tensor([[0, 1], [3, 9], [12, 15]], device="cuda")
    expected = (
        positions.unsqueeze(-1) * original.to(device="cuda", dtype=torch.bfloat16)
    ).flatten(1)
    torch.testing.assert_close(
        rope.inv_freq, original.to(device="cuda", dtype=torch.bfloat16), atol=0, rtol=0
    )
    torch.testing.assert_close(rope(positions), expected, atol=0, rtol=0)
    q = torch.randn(3, 2, dim, dtype=torch.bfloat16, device="cuda")
    cos = torch.randn(3, dim, device="cuda")
    sin = torch.randn(3, dim, device="cuda")
    q_out, k_out = apply_rotary_pos_emb_vision(q, q, cos, sin)
    assert q_out.dtype == k_out.dtype == q.dtype


def test_explicit_fp32_paths_recompute_on_stream_and_graph(fake_mesh):
    fake_mesh()
    layer = ReplicatedLinear(
        64,
        37,
        bias=False,
        params_dtype=torch.bfloat16,
        spec=Bf16Spec(),
        device="cuda",
    )
    layer.weight.data.normal_()
    norm = RMSNorm(
        37,
        dtype=torch.bfloat16,
        device="cuda",
        cast_before_affine=True,
        explicit_fp32_math=True,
    )
    norm.weight.data.normal_()
    stream = torch.cuda.Stream()
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            norm(layer(x.float())[0].to(torch.bfloat16))
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = norm(layer(x.float())[0].to(torch.bfloat16))
    for seed in (7, 13):
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        expected = F.linear(x.float(), layer.weight.float()).to(torch.bfloat16)
        expected = norm.weight * (
            expected.float()
            * torch.rsqrt(expected.float().square().mean(-1, keepdim=True) + 1e-6)
        ).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
