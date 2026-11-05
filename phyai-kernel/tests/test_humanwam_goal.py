"""HumanWAM-Goal fusion contracts across the supported local head counts."""

import torch
import pytest
from phyai_kernel.triton.humanwam_goal import _prepare_qkv, _add_rms_norm


@pytest.mark.parametrize("rows", [1, 37, 318])
def test_rounded_residual_norm(rows):
    torch.manual_seed(19)
    residual = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
    projected = torch.randn_like(residual)
    weight = torch.randn(4096, device="cuda", dtype=residual.dtype)
    hidden, output = torch.empty_like(residual), torch.empty_like(residual)
    _add_rms_norm[(rows,)](
        residual,
        projected,
        weight,
        hidden,
        output,
        1e-6,
        num_warps=4,
        enable_fp_fusion=False,
    )
    expected_hidden = residual + projected
    value = expected_hidden.float()
    expected = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).to(
        residual.dtype
    ) * weight
    assert torch.equal(hidden, expected_hidden)
    torch.testing.assert_close(output, expected, rtol=0.016, atol=0.016)


@pytest.mark.parametrize("heads", [2, 4, 8])
def test_qkv_preparation_and_graph_refresh(heads):
    torch.manual_seed(31)
    tokens, text, width = 13, 7, 128
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

    def launch():
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
            output_key,
            output_value,
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
