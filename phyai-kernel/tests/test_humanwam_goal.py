"""HumanWAM-Goal fusion contracts across the supported local head counts."""

import torch
import pytest
from phyai_kernel.triton.humanwam_goal import _prepare_qkv, _add_rms_norm, _silu_mul


@pytest.mark.parametrize("raise_inside", [False, True])
def test_projection_context_restores_blas_policy(raise_inside):
    from phyai.models.humanwam_goal.modeling_parallel_humanwam_goal import (
        full_precision_gemm,
    )

    def policy():
        return (
            torch.backends.cuda.preferred_blas_library(),
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
        )

    before = policy()
    try:
        with full_precision_gemm():
            assert policy()[1:] == (False, False)
            if raise_inside:
                raise RuntimeError("projection failed")
    except RuntimeError:
        if not raise_inside:
            raise
    assert policy() == before


@pytest.mark.parametrize("rows", [16, 17, 37, 318, 544, 643, 742])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@pytest.mark.parametrize("seed", [19, 71])
def test_rounded_residual_norm(rows, eps, seed):
    torch.manual_seed(seed)
    residual = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
    projected = torch.randn_like(residual)
    weight = torch.randn(4096, device="cuda", dtype=residual.dtype)
    hidden, output = torch.empty_like(residual), torch.empty_like(residual)

    def launch():
        _add_rms_norm[(rows,)](
            residual,
            projected,
            weight,
            hidden,
            output,
            eps,
            num_warps=1,
            enable_fp_fusion=False,
        )

    def check():
        expected_hidden = residual + projected
        value = expected_hidden.float()
        expected = (
            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        ).to(residual.dtype) * weight
        assert torch.equal(hidden.view(torch.uint8), expected_hidden.view(torch.uint8))
        assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))

    launch()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(2):
        residual.normal_()
        projected.normal_()
        weight.normal_()
        projected[0].copy_(-residual[0])
        graph.replay()
        check()


@pytest.mark.parametrize("heads", [2, 4, 8])
@pytest.mark.parametrize("interleaved", [False, True])
def test_qkv_preparation_and_graph_refresh(heads, interleaved):
    torch.manual_seed(31)
    tokens, text, width = 13, 7, 128
    if interleaved:
        text = 0
    packed = torch.randn(tokens, heads * 6 * width, device="cuda", dtype=torch.bfloat16)
    query, key, value = [
        v.view(tokens, h, width)
        for v, h in zip(
            packed.split((4 * heads * width, heads * width, heads * width), -1),
            (4 * heads, heads, heads),
        )
    ]
    q_weight = torch.randn(width, device="cuda", dtype=packed.dtype)
    k_weight = torch.randn_like(q_weight)
    angles = torch.randn(tokens, width, device="cuda")
    cos, sin = angles.cos().to(packed.dtype), angles.sin().to(packed.dtype)
    text_key = torch.randn(text, heads, width, device="cuda", dtype=packed.dtype)
    text_value = torch.randn_like(text_key)
    output_query = torch.empty(query.shape, device="cuda", dtype=packed.dtype)
    output_key = torch.empty(
        text + tokens, heads, width, device="cuda", dtype=packed.dtype
    )
    output_value = torch.empty_like(output_key)
    if interleaved:
        workspace = torch.empty(
            (tokens, 2, heads, width), device="cuda", dtype=packed.dtype
        )
        output_key, output_value = workspace[:, 0], workspace[:, 1]

    def launch(key_dest=output_key, value_dest=output_value, packed_output=interleaved):
        _prepare_qkv[(text + tokens, 5)](
            query,
            key,
            value,
            q_weight,
            k_weight,
            cos,
            sin,
            text_key,
            text_value,
            output_query,
            key_dest,
            value_dest,
            text,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            cos.stride(0),
            sin.stride(0),
            *text_key.stride(),
            *text_value.stride(),
            1e-6,
            1e-6,
            heads,
            INTERLEAVED_KV=packed_output,
            num_warps=heads,
            enable_fp_fusion=False,
        )

    def reference(x, weight):
        f = x.float()
        normalized = (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)).to(
            x.dtype
        ) * weight
        first, second = normalized.chunk(2, dim=-1)
        return (
            normalized * cos[:, None]
            + torch.cat((-second, first), dim=-1) * sin[:, None]
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(2):
        packed.normal_()
        cos.normal_()
        sin.normal_()
        text_key.normal_()
        text_value.normal_()
        graph.replay()
        torch.testing.assert_close(
            output_query, reference(query, q_weight), rtol=0.016, atol=0.016
        )
        torch.testing.assert_close(
            output_key[text:], reference(key, k_weight), rtol=0.016, atol=0.016
        )
        assert torch.equal(output_key[:text], text_key)
        assert torch.equal(output_value, torch.cat((text_value, value)))
        if interleaved:
            expected_query = output_query.clone()
            plain_key = torch.empty_like(
                output_key, memory_format=torch.contiguous_format
            )
            plain_value = torch.empty_like(plain_key)
            launch(plain_key, plain_value, False)
            assert torch.equal(
                output_query.view(torch.uint8), expected_query.view(torch.uint8)
            )
            assert torch.equal(
                output_key.view(torch.uint8), plain_key.view(torch.uint8)
            )
            assert torch.equal(
                output_value.view(torch.uint8), plain_value.view(torch.uint8)
            )


@pytest.mark.parametrize("up_mode", ["one", "negative", "zero", "random"])
def test_silu_mul_finite_bfloat16(up_mode):
    raw = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    gate = raw[torch.isfinite(raw)].reshape(1, -1)
    up = (
        torch.randn_like(gate)
        if up_mode == "random"
        else torch.full_like(gate, {"one": 1, "negative": -1, "zero": 0}[up_mode])
    )
    fused = torch.cat((gate, up), dim=-1)
    output = torch.empty_like(gate)
    width = gate.shape[-1]

    def launch():
        _silu_mul[((output.numel() + 511) // 512,)](
            fused,
            output,
            output.numel(),
            width,
            512,
            num_warps=4,
            enable_fp_fusion=False,
        )

    def check():
        a, b = fused.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(a) * b
        assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))

    launch()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    fused[:, width:].copy_(up.flip(-1))
    graph.replay()
    check()
