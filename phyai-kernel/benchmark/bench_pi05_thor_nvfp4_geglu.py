"""Benchmark the fixed pi0.5 Thor NVFP4 GeGLU kernel.

The baseline has the same packed-NVFP4 input and output contract, but stages
FlashInfer NVFP4 GEMM, PyTorch GeGLU, and FlashInfer NVFP4 quantization. Both
paths are captured into CUDA Graphs before timing.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from flashinfer import gemm as flashinfer_gemm
from flashinfer.quantization import SfLayout, nvfp4_quantize

from phyai_kernel.cuda.pi05_thor_nvfp4_geglu import (
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MERGED_SIZE,
    SUPPORTED_M,
    nvfp4_scale_shape,
    pi05_thor_nvfp4_geglu,
)


def _capture(function: Callable[[], object]) -> tuple[torch.cuda.CUDAGraph, object]:
    for _ in range(5):
        result = function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = function()
    return graph, result


def _time_graph(graph: torch.cuda.CUDAGraph, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _benchmark_shape(
    m: int,
    interleaved_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    seed: int,
    warmup: int,
    samples: int,
    iterations: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed + m)
    activation_bf16 = torch.randn(
        m,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    activation, activation_scale = nvfp4_quantize(
        activation_bf16,
        global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        enable_pdl=False,
    )
    workspace = torch.zeros(m, INTERMEDIATE_SIZE, dtype=torch.uint8, device="cuda")
    output = torch.empty(m, INTERMEDIATE_SIZE // 2, dtype=torch.uint8, device="cuda")
    output_scale = torch.empty(
        nvfp4_scale_shape(m, INTERMEDIATE_SIZE),
        dtype=torch.uint8,
        device="cuda",
    )
    staged_merged = torch.empty(m, MERGED_SIZE, dtype=torch.bfloat16, device="cuda")

    def candidate() -> tuple[torch.Tensor, torch.Tensor]:
        return pi05_thor_nvfp4_geglu(
            activation,
            activation_scale,
            interleaved_weight,
            weight_scale,
            workspace,
            output,
            output_scale,
        )

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        merged = flashinfer_gemm.mm_fp4(
            activation,
            interleaved_weight.t(),
            activation_scale,
            weight_scale.t().view(torch.uint8),
            global_scale,
            torch.bfloat16,
            staged_merged,
            block_size=16,
            use_nvfp4=True,
            backend="cudnn",
            enable_pdl=False,
        )
        hidden = F.gelu(merged[:, 0::2], approximate="tanh") * merged[:, 1::2]
        return nvfp4_quantize(
            hidden,
            global_scale,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
            enable_pdl=False,
        )

    candidate_graph, candidate_result = _capture(candidate)
    baseline_graph, baseline_result = _capture(baseline)
    # Keep graph-captured intermediates alive for the full timing region.
    graph_storage = (candidate_result, baseline_result, staged_merged)

    for _ in range(warmup):
        candidate_graph.replay()
        baseline_graph.replay()
    torch.cuda.synchronize()

    candidate_samples: list[float] = []
    baseline_samples: list[float] = []
    for sample_index in range(samples):
        order = (
            (("candidate", candidate_graph), ("baseline", baseline_graph))
            if sample_index % 2 == 0
            else (("baseline", baseline_graph), ("candidate", candidate_graph))
        )
        for name, graph in order:
            elapsed = _time_graph(graph, iterations)
            if name == "candidate":
                candidate_samples.append(elapsed)
            else:
                baseline_samples.append(elapsed)

    del graph_storage
    candidate_median = statistics.median(candidate_samples)
    baseline_median = statistics.median(baseline_samples)
    return {
        "m": m,
        "candidate_ms": candidate_median,
        "baseline_ms": baseline_median,
        "speedup": baseline_median / candidate_median,
        "delta_percent": (candidate_median / baseline_median - 1.0) * 100.0,
        "candidate_raw_ms": candidate_samples,
        "baseline_raw_ms": baseline_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", default=",".join(str(m) for m in sorted(SUPPORTED_M)))
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("this benchmark requires an SM110 CUDA device")
    shapes = [int(value) for value in args.m.split(",") if value]
    if any(m not in SUPPORTED_M for m in shapes):
        raise ValueError(f"supported M values are {sorted(SUPPORTED_M)}")
    if args.warmup < 0 or args.samples <= 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative; samples and iterations positive")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    weight_bf16 = (
        torch.randn(
            MERGED_SIZE,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    interleaved_weight, weight_scale = nvfp4_quantize(
        weight_bf16,
        global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        enable_pdl=False,
    )

    rows = [
        _benchmark_shape(
            m,
            interleaved_weight,
            weight_scale,
            global_scale,
            seed=args.seed,
            warmup=args.warmup,
            samples=args.samples,
            iterations=args.iterations,
        )
        for m in shapes
    ]
    candidate_geomean = math.exp(
        statistics.fmean(math.log(float(row["candidate_ms"])) for row in rows)
    )
    baseline_geomean = math.exp(
        statistics.fmean(math.log(float(row["baseline_ms"])) for row in rows)
    )
    report = {
        "schema": "phyai.pi05_thor_nvfp4_geglu.benchmark.v1",
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "baseline": "FlashInfer NVFP4 GEMM + PyTorch GeGLU + FlashInfer NVFP4 quantize",
        "timing": {
            "mode": "CUDA Graph replay with CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "iterations_per_sample": args.iterations,
            "rotated_order": True,
        },
        "rows": rows,
        "geomean": {
            "candidate_ms": candidate_geomean,
            "baseline_ms": baseline_geomean,
            "speedup": baseline_geomean / candidate_geomean,
            "delta_percent": (candidate_geomean / baseline_geomean - 1.0) * 100.0,
        },
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
