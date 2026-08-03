from __future__ import annotations

import argparse
import json
import re

import torch


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
    return flop / best_seconds / 1e12


def measure_memory_bandwidth_tb_s(
    *,
    nbytes: int = 4 * 1024**3,
    warmup: int = 10,
    iters: int = 50,
    device: int = 0,
) -> float:
    """Measure large device-copy bandwidth, counting read plus write traffic."""

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
