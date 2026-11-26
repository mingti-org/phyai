"""Compare CuTe T5 GELU with a tuned Triton implementation at equal precision.

Run from the workspace root::

    uv run python phyai-kernel/benchmark/bench_t5_gelu.py
    uv run python phyai-kernel/benchmark/bench_t5_gelu.py --layouts gate transpose

Both kernels preserve the eager expression's intermediate rounding. Timings
exclude compilation and allocation, use CUDA graphs with repeated buffers,
and alternate provider order. The Triton block/warp/store-cache search runs per shape,
including two-dimensional tiles for transposed input.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
from collections.abc import Callable

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from phyai_kernel import t5_gelu


@triton.jit
def triton_activation(x, BF16: tl.constexpr):
    dtype = x.dtype
    x = x.to(tl.float32)
    square = (x * x).to(dtype).to(tl.float32)
    cube = (square * x).to(dtype).to(tl.float32)
    cubic = (0.044715 * cube).to(dtype).to(tl.float32)
    polynomial = (x + cubic).to(dtype).to(tl.float32)
    argument = (0.7978845608028654 * polynomial).to(dtype).to(tl.float32)
    if BF16:
        tangent = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[argument],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        tangent = libdevice.tanh(argument)
    tangent = tangent.to(dtype).to(tl.float32)
    factor = (1.0 + tangent).to(dtype).to(tl.float32)
    half_x = (0.5 * x).to(dtype).to(tl.float32)
    return (half_x * factor).to(dtype)


@triton.jit
def triton_t5_gelu(
    X,
    Y,
    ROW_STRIDE: tl.constexpr,
    COLUMN_STRIDE: tl.constexpr,
    COLUMNS: tl.constexpr,
    ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
    BF16: tl.constexpr,
    STORE_CACHE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if COLUMN_STRIDE == 1 and ROW_STRIDE == COLUMNS:
        source = offsets
    else:
        source = offsets // COLUMNS * ROW_STRIDE + offsets % COLUMNS * COLUMN_STRIDE
    x = tl.load(X + source, offsets < ELEMENTS, 0)
    result = triton_activation(x, BF16)
    tl.store(Y + offsets, result, offsets < ELEMENTS, cache_modifier=STORE_CACHE)


@triton.jit
def triton_t5_gelu_tiled(
    X,
    Y,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COLUMN_STRIDE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
    BF16: tl.constexpr,
    STORE_CACHE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.program_id(1) * BLOCK_COLUMNS + tl.arange(0, BLOCK_COLUMNS)
    valid = (rows[:, None] < ROWS) & (columns[None, :] < COLUMNS)
    source = rows[:, None] * ROW_STRIDE + columns[None, :] * COLUMN_STRIDE
    x = tl.load(X + source, valid, 0)
    result = triton_activation(x, BF16)
    tl.store(
        Y + rows[:, None] * COLUMNS + columns[None, :],
        result,
        valid,
        cache_modifier=STORE_CACHE,
    )


def reference(x: torch.Tensor) -> torch.Tensor:
    return (
        0.5
        * x
        * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3.0))))
    )


def assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    non_nan = ~expected.isnan()
    assert torch.equal(
        actual.view(torch.int16)[non_nan], expected.view(torch.int16)[non_nan]
    )


def triton_function(
    x: torch.Tensor,
    out: torch.Tensor,
    block: int | tuple[int, int],
    warps: int,
    store_cache: str = "",
):
    def execute():
        options = {
            "ROW_STRIDE": x.stride(0),
            "COLUMN_STRIDE": x.stride(1),
            "COLUMNS": x.shape[1],
            "BF16": x.dtype == torch.bfloat16,
            "STORE_CACHE": store_cache,
            "num_warps": warps,
            "enable_fp_fusion": False,
            "enable_reflect_ftz": False,
        }
        if isinstance(block, tuple):
            block_rows, block_columns = block
            grid = (
                triton.cdiv(x.shape[0], block_rows),
                triton.cdiv(x.shape[1], block_columns),
            )
            triton_t5_gelu_tiled[grid](
                x,
                out,
                ROWS=x.shape[0],
                BLOCK_ROWS=block_rows,
                BLOCK_COLUMNS=block_columns,
                **options,
            )
        else:
            triton_t5_gelu[(triton.cdiv(x.numel(), block),)](
                x, out, ELEMENTS=x.numel(), BLOCK=block, **options
            )

    return execute


def capture(fn: Callable, iterations: int) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            fn()
    return graph


def measure(graphs, iterations: int, repeats: int) -> list[list[float]]:
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    times: list[list[float]] = [[] for _ in graphs]
    for graph in graphs:
        graph.replay()
    for repeat in range(repeats):
        order = range(len(graphs)) if repeat % 2 == 0 else reversed(range(len(graphs)))
        for index in order:
            start.record()
            graphs[index].replay()
            end.record()
            end.synchronize()
            times[index].append(start.elapsed_time(end) * 1000.0 / iterations)
    return times


def make_input(
    rows: int, columns: int, dtype: torch.dtype, layout: str
) -> torch.Tensor:
    def rand(*shape):
        return torch.randn(shape, device="cuda", dtype=dtype)

    if layout == "contiguous":
        return rand(rows, columns)
    if layout == "gate":
        return rand(rows, 2 * columns)[:, :columns]
    if layout == "offset":
        return rand(rows * columns + 1)[1:].view(rows, columns)
    if layout == "transpose":
        return rand(columns, rows).T
    if layout == "slice":
        return rand(rows, 2 * columns)[:, ::2]
    if layout == "expand":
        return rand(1, columns).expand(rows, -1)
    raise ValueError(layout)


def check_all_values(dtype: torch.dtype) -> None:
    x = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.uint16)
        .view(dtype)
        .view(256, 256)
    )
    for view in (x, x.T):
        expected = reference(view)
        out = torch.empty_like(view, memory_format=torch.contiguous_format)
        for block in (1024, (32, 32)):
            triton_function(view, out, block, 4)()
            assert_exact(out, expected)
        assert_exact(t5_gelu(view), expected)


def benchmark_case(rows, columns, dtype, layout, *, iterations=200, repeats=11):
    x = make_input(rows, columns, dtype, layout)
    expected = reference(x)
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    cute_fn = lambda: t5_gelu(x, out=out)
    cute_fn()
    assert_exact(out, expected)

    configurations = list(
        itertools.product((256, 512, 1024, 2048, 4096), (4, 8), ("", ".cs"))
    )
    if x.stride(0) == 1 and x.stride(1) > 1:
        configurations.extend(
            itertools.product(
                ((16, 32), (32, 32), (32, 64), (64, 32), (64, 64)), (4, 8), ("", ".cs")
            )
        )
    candidates = []
    for block, warps, store_cache in configurations:
        fn = triton_function(x, out, block, warps, store_cache)
        fn()
        assert_exact(out, expected)
        candidates.append(capture(fn, iterations))
    tuning = measure(candidates, iterations, 5)
    best_index = min(range(len(candidates)), key=lambda i: statistics.median(tuning[i]))
    block, warps, store_cache = configurations[best_index]
    triton_graph = candidates[best_index]
    cute_graph = capture(cute_fn, iterations)
    samples = measure([cute_graph, triton_graph], iterations, repeats)
    cute_us, triton_us = map(statistics.median, samples)
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "rows": rows,
        "columns": columns,
        "layout": layout,
        "strides": list(x.stride()),
        "cute_us": cute_us,
        "triton_us": triton_us,
        "speedup": triton_us / cute_us,
        "triton_kind": "tiled" if isinstance(block, tuple) else "flat",
        "triton_block": block,
        "triton_warps": warps,
        "triton_store_cache": store_cache or "default",
        "cute_samples_us": samples[0],
        "triton_samples_us": samples[1],
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--dtypes", nargs="+", choices=("bf16", "fp16"), default=["bf16", "fp16"]
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[512, 1024])
    parser.add_argument(
        "--columns", type=int, nargs="+", default=[1280, 2560, 5120, 10240]
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("contiguous", "gate", "offset", "transpose", "slice", "expand"),
        default=["contiguous", "gate"],
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(*args.rows, *args.columns, args.iterations, args.repeats) <= 0:
        parser.error("shapes, iterations and repeats must be positive")
    torch.cuda.set_device(args.device)
    torch.manual_seed(42)
    results = []
    print(
        "dtype      layout        rows columns   CuTe(us) Triton(us) speedup",
        flush=True,
    )
    for name in args.dtypes:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[name]
        check_all_values(dtype)
        for rows, columns, layout in itertools.product(
            args.rows, args.columns, args.layouts
        ):
            result = benchmark_case(
                rows,
                columns,
                dtype,
                layout,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            results.append(result)
            print(
                f"{name:<10} {layout:<12} {rows:5d} {columns:7d} "
                f"{result['cute_us']:10.3f} {result['triton_us']:10.3f} {result['speedup']:7.3f}x",
                flush=True,
            )
            if args.output:
                args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
