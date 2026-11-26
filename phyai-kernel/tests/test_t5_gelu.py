"""T5 GELU rounding, CUDA views, caller buffers and graph/compile behavior."""

import math

import pytest
import torch

from phyai_kernel.cuda.jit.t5_gelu import t5_gelu


def reference(x):
    return (
        0.5
        * x
        * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3.0))))
    )


def assert_exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = ~expected.isnan()
    assert torch.equal(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite]
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice", "offset"])
def test_all_half_precision_values(dtype, layout):
    values = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.uint16)
        .view(dtype)
    )
    if layout == "transpose":
        values = values.view(256, 256).T
    elif layout == "slice":
        values = values.repeat_interleave(2)[::2]
    elif layout == "offset":
        values = torch.cat((values[:1], values))[1:]
    assert_exact(t5_gelu(values), reference(values))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "layout",
    [
        "contiguous",
        "gate",
        "gate_offset",
        "slice",
        "transpose",
        "permute",
        "offset",
        "expand",
        "expand_inner",
        "overlapping",
        "scalar",
        "empty",
        "singleton",
    ],
)
def test_views(dtype, layout):
    generator = torch.Generator(device="cuda").manual_seed(42)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    if layout == "contiguous":
        x = rand(2, 17, 10240)
    elif layout == "gate":
        x = rand(2, 17, 10240).chunk(2, dim=-1)[0]
    elif layout == "gate_offset":
        x = rand(2, 17, 10240).chunk(2, dim=-1)[1]
    elif layout == "slice":
        x = rand(7, 19, 521)[1::2, 2::3, 3::2]
    elif layout == "transpose":
        x = rand(129, 257).T
    elif layout == "permute":
        x = rand(13, 17, 31).permute(2, 0, 1)
    elif layout == "offset":
        x = rand(5000)[1:4098]
    elif layout == "expand":
        x = rand(1, 17, 129).expand(3, -1, -1)
    elif layout == "expand_inner":
        x = rand(17, 1).expand(-1, 129)
    elif layout == "overlapping":
        x = rand(300).as_strided((17, 129), (3, 1), storage_offset=2)
    elif layout == "scalar":
        x = rand(1).squeeze()
    elif layout == "empty":
        x = rand(3, 0, 17).transpose(0, 2)
    else:
        x = rand(13, 17)[None, :, None, :, None]
    snapshot = x.clone()
    actual = t5_gelu(x)
    assert actual.is_contiguous()
    assert_exact(actual, reference(x))
    torch.testing.assert_close(x, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("columns", [8, 24, 1280, 2560, 5120, 10240, 10248])
def test_partial_vector_tiles(columns):
    x = torch.randn(3, 2 * columns, device="cuda", dtype=torch.bfloat16)[:, :columns]
    assert_exact(t5_gelu(x), reference(x))


def test_tall_matrix_view():
    x = torch.randn(65536, 16, device="cuda", dtype=torch.bfloat16)[:, :8]
    assert_exact(t5_gelu(x), reference(x))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_high_rank_view(dtype):
    x = torch.randn(3, 5, 7, 11, 13, device="cuda", dtype=dtype).permute(4, 2, 0, 3, 1)
    assert_exact(t5_gelu(x), reference(x))


def test_tall_transposed_broadcast():
    base = torch.randn(64 * 65536 + 8, device="cuda", dtype=torch.bfloat16)
    x = base[:, None].expand(-1, 8)
    assert_exact(t5_gelu(x), reference(base)[:, None].expand_as(x))


@pytest.mark.parametrize("layout", ["slice", "transpose"])
def test_output_view_and_cuda_graph(layout):
    if layout == "transpose":
        x = torch.randn(129, 17, device="cuda", dtype=torch.bfloat16).T
    else:
        x = torch.randn(17, 258, device="cuda", dtype=torch.bfloat16)[:, 1::2]
    storage = torch.full((x.numel() + 2,), 42, device="cuda", dtype=x.dtype)
    out = storage[1:-1].view(x.shape)
    t5_gelu(x, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        t5_gelu(x, out=out)
    x.normal_()
    graph.replay()
    assert_exact(out, reference(x))
    torch.testing.assert_close(
        storage[[0, -1]], torch.full((2,), 42, device="cuda", dtype=x.dtype)
    )


def test_invalid_output_is_rejected():
    x = torch.randn(3, 17, device="cuda", dtype=torch.bfloat16)
    for out in (
        x,
        torch.empty(3, 18, device="cuda", dtype=x.dtype),
        x.T.contiguous().T,
    ):
        with pytest.raises(RuntimeError):
            t5_gelu(x, out=out)


def test_fullgraph_compile_preserves_view_semantics():
    x = torch.randn(3, 514, device="cuda", dtype=torch.bfloat16)[:, 1::2]
    compiled = torch.compile(t5_gelu, fullgraph=True)
    assert_exact(compiled(x), reference(x))
    x.normal_()
    assert_exact(compiled(x), reference(x))
