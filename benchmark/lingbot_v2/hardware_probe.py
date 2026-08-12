from __future__ import annotations

import argparse
import json
import math
import re

import torch


def validate_measurement_args(*, warmup: int, iters: int) -> None:
    """Reject loop counts that cannot produce one timed measurement."""

    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}.")
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}.")


def require_positive_duration(seconds: float, operation: str) -> float:
    """Return a usable elapsed time or fail with a diagnostic error."""

    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(
            f"{operation} produced a non-positive measured duration: {seconds!r}"
        )
    return seconds


def device_slug(name: str) -> str:
    """Return a filesystem-safe device name."""

    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "gpu"


def probe_device(index: int = 0) -> dict:
    """Collect static facts from the CUDA runtime."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot probe a GPU.")
    props = torch.cuda.get_device_properties(index)
    return {
        "name": torch.cuda.get_device_name(index),
        "index": index,
        "sm_count": props.multi_processor_count,
        "total_mem_gb": round(props.total_memory / 1e9, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def measure_bf16_peak_tflops(
    *,
    size: int = 8192,
    warmup: int = 10,
    iters: int = 50,
    device: int = 0,
) -> float:
    """Measure sustained BF16 dense-GEMM throughput."""

    validate_measurement_args(warmup=warmup, iters=iters)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")

    dev = torch.device(f"cuda:{device}")
    a = torch.randn(size, size, dtype=torch.bfloat16, device=dev)
    b = torch.randn(size, size, dtype=torch.bfloat16, device=dev)
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize(dev)

    flop = 2.0 * size**3
    best_seconds = float("inf")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        torch.mm(a, b)
        end.record()
        torch.cuda.synchronize(dev)
        best_seconds = min(best_seconds, start.elapsed_time(end) / 1e3)
    best_seconds = require_positive_duration(best_seconds, "BF16 GEMM")
    return flop / best_seconds / 1e12


def measure_memory_bandwidth_tb_s(
    *,
    nbytes: int = 4 * 1024**3,
    warmup: int = 10,
    iters: int = 50,
    device: int = 0,
) -> float:
    """Measure large device-copy bandwidth, counting read plus write traffic."""

    validate_measurement_args(warmup=warmup, iters=iters)
    if nbytes < torch.tensor([], dtype=torch.bfloat16).element_size():
        raise ValueError(
            f"nbytes must hold at least one BF16 element (2 bytes), got {nbytes}."
        )

    dev = torch.device(f"cuda:{device}")
    n_elems = nbytes // 2
    source = torch.empty(n_elems, dtype=torch.bfloat16, device=dev)
    destination = torch.empty_like(source)
    for _ in range(warmup):
        destination.copy_(source)
    torch.cuda.synchronize(dev)

    moved = 2.0 * source.numel() * source.element_size()
    best_seconds = float("inf")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        destination.copy_(source)
        end.record()
        torch.cuda.synchronize(dev)
        best_seconds = min(best_seconds, start.elapsed_time(end) / 1e3)
    best_seconds = require_positive_duration(best_seconds, "device copy")
    return moved / best_seconds / 1e12


def measure_roofline(
    *,
    device: int = 0,
    gemm_size: int = 8192,
    copy_bytes: int = 4 * 1024**3,
    warmup: int = 10,
    iters: int = 50,
) -> dict:
    """Return device facts and measured BF16 roofline values."""

    validate_measurement_args(warmup=warmup, iters=iters)
    if gemm_size <= 0:
        raise ValueError(f"gemm_size must be positive, got {gemm_size}.")
    if copy_bytes < 2:
        raise ValueError(
            f"copy_bytes must hold at least one BF16 element, got {copy_bytes}."
        )

    info = probe_device(device)
    peak = measure_bf16_peak_tflops(
        size=gemm_size,
        warmup=warmup,
        iters=iters,
        device=device,
    )
    bandwidth = measure_memory_bandwidth_tb_s(
        nbytes=copy_bytes,
        warmup=warmup,
        iters=iters,
        device=device,
    )
    if bandwidth <= 0:
        raise RuntimeError(
            f"measured memory bandwidth must be positive, got {bandwidth!r}"
        )
    info.update(
        {
            "peak_bf16_tflops": round(peak, 1),
            "memory_bandwidth_tb_s": round(bandwidth, 3),
            "ridge_point_flop_per_byte": round(peak / bandwidth, 1),
            "microbench": {
                "gemm_size": gemm_size,
                "copy_bytes": copy_bytes,
                "warmup": warmup,
                "iters": iters,
            },
        }
    )
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gemm-size", type=int, default=8192)
    parser.add_argument("--copy-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    info = measure_roofline(
        device=args.device,
        gemm_size=args.gemm_size,
        copy_bytes=args.copy_bytes,
        iters=args.iters,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
