from __future__ import annotations

import argparse
import json
import re

import torch


def device_slug(name: str) -> str:
    """Filesystem-safe lowercase slug for a device name."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "gpu"


def cuda_device(index: int) -> torch.device:
    """Validated CUDA device for a logical device index."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot probe a GPU.")
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index must be in [0, {torch.cuda.device_count() - 1}], "
            f"got {index}."
        )
    return torch.device("cuda", index)


def probe_device(index: int = 0) -> dict:
    """Static CUDA device facts."""
    device = cuda_device(index)
    props = torch.cuda.get_device_properties(device)
    return {
        "name": torch.cuda.get_device_name(device),
        "index": index,
        "sm_count": props.multi_processor_count,
        "total_mem_gb": round(props.total_memory / 1e9, 2),
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def measure_bf16_peak_tflops(
    *,
    size: int = 8192,
    warmup: int = 10,
    iters: int = 50,
    device: int = 0,
) -> float:
    """Peak sustained BF16 dense GEMM throughput in TFLOPS."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}.")
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}.")

    dev = cuda_device(device)
    with torch.cuda.device(dev):
        a = torch.randn(size, size, dtype=torch.bfloat16, device=dev)
        b = torch.randn(size, size, dtype=torch.bfloat16, device=dev)
        for _ in range(warmup):
            torch.mm(a, b)
        _sync(dev)

        flop = 2.0 * size**3
        best_s = float("inf")
        stream = torch.cuda.current_stream(dev)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record(stream)
            torch.mm(a, b)
            end.record(stream)
            _sync(dev)
            best_s = min(best_s, start.elapsed_time(end) / 1e3)
    return flop / best_s / 1e12


def measure_hbm_bandwidth_tb_s(
    *,
    nbytes: int = 4 * 1024**3,
    warmup: int = 10,
    iters: int = 50,
    device: int = 0,
) -> float:
    """Peak HBM bandwidth from a large device-to-device copy in TB/s."""
    if nbytes < 2:
        raise ValueError(f"nbytes must be at least 2, got {nbytes}.")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}.")
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}.")

    dev = cuda_device(device)
    with torch.cuda.device(dev):
        n_elems = nbytes // 2
        src = torch.empty(n_elems, dtype=torch.bfloat16, device=dev)
        dst = torch.empty_like(src)
        for _ in range(warmup):
            dst.copy_(src)
        _sync(dev)

        moved = 2.0 * src.numel() * src.element_size()
        best_s = float("inf")
        stream = torch.cuda.current_stream(dev)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record(stream)
            dst.copy_(src)
            end.record(stream)
            _sync(dev)
            best_s = min(best_s, start.elapsed_time(end) / 1e3)
    return moved / best_s / 1e12


def measure_roofline(
    *,
    device: int = 0,
    gemm_size: int = 8192,
    copy_bytes: int = 4 * 1024**3,
    warmup: int = 10,
    iters: int = 50,
) -> dict:
    """Detected device facts plus measured peak compute, bandwidth, and ridge."""
    info = probe_device(device)
    peak = measure_bf16_peak_tflops(
        size=gemm_size, warmup=warmup, iters=iters, device=device
    )
    bw = measure_hbm_bandwidth_tb_s(
        nbytes=copy_bytes, warmup=warmup, iters=iters, device=device
    )
    info.update(
        {
            "peak_bf16_tflops": round(peak, 1),
            "hbm_tb_s": round(bw, 3),
            "ridge_point_flop_per_byte": round(peak / bw, 1),
            "microbench": {
                "gemm_size": gemm_size,
                "copy_bytes": copy_bytes,
                "warmup": warmup,
                "iters": iters,
            },
        }
    )
    return info


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--gemm-size", type=int, default=8192)
    ap.add_argument("--copy-bytes", type=int, default=4 * 1024**3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    info = measure_roofline(
        device=args.device,
        gemm_size=args.gemm_size,
        copy_bytes=args.copy_bytes,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(json.dumps(info, indent=2))
    print(
        f"\n{info['name']}: BF16 peak {info['peak_bf16_tflops']} TFLOPS, "
        f"HBM {info['hbm_tb_s']} TB/s, ridge "
        f"{info['ridge_point_flop_per_byte']} FLOP/byte"
    )


if __name__ == "__main__":
    _main()
