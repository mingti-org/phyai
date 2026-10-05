import pytest
import torch

from phyai.layers.axial_rotary_embedding import AxialRotaryEmbedding
from phyai.layers.layer_norm import LayerNorm
from phyai.layers.rotary_embedding import apply_rope
from phyai_kernel.triton.rotary_embedding import apply_rope_precomputed


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_axial_rope_matches_complex_rotation(dtype):
    device = torch.device("cuda")
    dims = (16, 56, 56)
    rope = AxialRotaryEmbedding(dims, device=device)
    positions = torch.tensor([[0, -4, -3], [1, -0.5, 2], [8193, 4, 3]], device=device)
    q = torch.randn(2, 3, 4, sum(dims), device=device, dtype=dtype)
    k = torch.randn(2, 3, 2, sum(dims), device=device, dtype=dtype)
    frequencies = torch.cat(
        [
            positions[:, i, None]
            / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))
            for i, dim in enumerate(dims)
        ],
        dim=-1,
    )
    phases = torch.polar(torch.ones_like(frequencies), frequencies)[None, :, None]
    actual = rope(positions, q, k)
    for source, output in zip((q, k), actual):
        reference = (
            torch.view_as_real(
                torch.view_as_complex(source.float().reshape(*source.shape[:-1], -1, 2))
                * phases
            )
            .flatten(-2)
            .to(dtype)
        )
        torch.testing.assert_close(output, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_non_affine_layer_norm_has_no_checkpoint_parameters(dtype):
    norm = LayerNorm(128, eps=1e-6, elementwise_affine=False, device="cuda")
    identity_affine = LayerNorm(128, eps=1e-6, bias=False, device="cuda")
    assert not list(norm.parameters())
    assert not norm.state_dict()
    value = torch.randn(2, 5, 128, device="cuda", dtype=dtype)
    actual = norm(value)
    assert actual.dtype == dtype
    # Changing parameter storage must preserve the existing kernel arithmetic.
    # Cross-backend numerical agreement is covered in test_layer_norm.py.
    torch.testing.assert_close(actual, identity_affine(value), atol=0, rtol=0)


def test_axial_rope_dtype_conversion_preserves_frequency_precision():
    rope = AxialRotaryEmbedding((16, 56, 56), device="cuda")
    positions = torch.tensor([[8193.0, -47.5, 83.5]], device="cuda")
    expected = rope.get_cos_sin(positions)
    rope.to(dtype=torch.bfloat16)
    actual = rope.get_cos_sin(positions)
    for before, after in zip(expected, actual):
        torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["ragged", "qkv_view", "strided_channels"])
def test_axial_rope_fusion_preserves_fp32_rounding(dtype, layout):
    rope = AxialRotaryEmbedding((16, 56, 56), device="cuda")
    prefix = (7,) if layout == "ragged" else (2, 7)
    if layout == "qkv_view":
        q, k, _ = torch.randn(*prefix, 3, 4, 128, device="cuda", dtype=dtype).unbind(-3)
        k = k[..., :2, :]
    elif layout == "strided_channels":
        q = torch.randn(*prefix, 4, 256, device="cuda", dtype=dtype)[..., ::2]
        k = torch.randn(*prefix, 2, 256, device="cuda", dtype=dtype)[..., ::2]
    else:
        q = torch.randn(*prefix, 4, 128, device="cuda", dtype=dtype)
        k = torch.randn(*prefix, 2, 128, device="cuda", dtype=dtype)
    phases = torch.randn(7, 256, device="cuda")
    # Noncontiguous, independently valued channels test the full cos/sin API.
    cos, sin = phases[..., ::2], phases[..., 1::2]
    expected = apply_rope(q.float(), k.float(), cos, sin, interleave=True)
    actual = rope.apply(q, k, cos, sin)
    for source, reference, output in zip((q, k), expected, actual):
        assert output.is_contiguous()
        assert output.dtype == source.dtype
        torch.testing.assert_close(output, reference.to(source.dtype), atol=0, rtol=0)


@pytest.mark.parametrize("interleave", [True, False])
def test_precomputed_rope_batched_broadcast_and_mixed_dtype(interleave):
    q = torch.randn(2, 5, 3, 72, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 5, 1, 72, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(2, 1, 72, device="cuda")
    sin = torch.randn(1, 5, 72, device="cuda")
    expected = apply_rope(q.float(), k.float(), cos, sin, interleave=interleave)
    actual = apply_rope_precomputed(q, k, cos, sin, interleave)
    for source, reference, output in zip((q, k), expected, actual):
        torch.testing.assert_close(output, reference.to(source.dtype), atol=0, rtol=0)


def test_axial_rope_fusion_cuda_graph_replays_new_inputs():
    rope = AxialRotaryEmbedding((16, 56, 56), device="cuda")
    q, k = torch.randn(2, 1, 9, 4, 128, device="cuda", dtype=torch.bfloat16).unbind()
    cos, sin = rope.get_cos_sin(torch.randn(9, 3, device="cuda"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            rope.apply(q, k, cos, sin)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = rope.apply(q, k, cos, sin)
    q.add_(0.25)
    k.mul_(0.5)
    graph.replay()
    expected = apply_rope(q.float(), k.float(), cos, sin, interleave=True)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference.to(q.dtype), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_axial_rope_fusion_fullgraph_compile(dtype):
    torch.manual_seed(3201)
    rope = AxialRotaryEmbedding((16, 56, 56), device="cuda")
    q = torch.randn(1, 256, 16, 128, device="cuda", dtype=dtype)
    k = torch.randn(1, 256, 8, 128, device="cuda", dtype=dtype)
    cos, sin = rope.get_cos_sin(torch.randn(256, 3, device="cuda") * 100)
    compiled = torch.compile(rope.apply, fullgraph=True)
    actual = compiled(q, k, cos, sin)
    expected = apply_rope(q.float(), k.float(), cos, sin, interleave=True)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference.to(q.dtype), atol=0, rtol=0)


def test_precomputed_rope_empty_tokens():
    q = torch.empty(2, 0, 3, 72, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(2, 0, 1, 72, device="cuda", dtype=torch.bfloat16)
    phases = torch.empty(0, 72, device="cuda")
    actual = apply_rope_precomputed(q, k, phases, phases)
    assert actual[0].shape == q.shape
    assert actual[1].shape == k.shape


def test_axial_rope_fullgraph_dynamic_sequence_qkv_views():
    rope = AxialRotaryEmbedding((16, 56, 56), device="cuda")
    compiled = torch.compile(rope.apply, fullgraph=True, dynamic=True)
    for sequence in (13, 29):
        q, k, _ = torch.randn(
            2, sequence, 3, 4, 128, device="cuda", dtype=torch.bfloat16
        ).unbind(-3)
        k = k[..., :2, :]
        cos, sin = rope.get_cos_sin(torch.randn(sequence, 3, device="cuda"))
        actual = compiled(q, k, cos, sin)
        expected = apply_rope(q.float(), k.float(), cos, sin, interleave=True)
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference.to(q.dtype), atol=0, rtol=0)
