from __future__ import annotations

import argparse
import json
import re

import torch


def device_slug(name: str) -> str:
    """Filesystem-safe lowercase slug for a device name (for output filenames)."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "gpu"


def probe_device(index: int = 0) -> dict:
    """Static device facts via the CUDA runtime — name, SMs, memory, clocks."""
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


def _sync() -> None:
    torch.cuda.synchronize()


def measure_bf16_peak_tflops(
    *,
    size: int = 8192,
    warmup: int = 10,
    iters: int = 50,
    device: int = 0,
) -> float:
    """Peak sustained BF16 dense GEMM throughput (TFLOPS), best of ``iters``.

    A square ``size × size`` matmul does ``2·size³`` FLOP. Large ``size`` keeps
    the GEMM compute-bound (well past the roofline ridge) so the result tracks
    the device's BF16 tensor-core peak, not memory traffic.
    """
    dev = torch.device(f"cuda:{device}")
    a = torch.randn(size, size, dtype=torch.bfloat16, device=dev)
    b = torch.randn(size, size, dtype=torch.bfloat16, device=dev)
    for _ in range(warmup):
        torch.mm(a, b)
    _sync()

    flop = 2.0 * size**3
    best_s = float("inf")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        torch.mm(a, b)
        end.record()
        _sync()
        best_s = min(best_s, start.elapsed_time(end) / 1e3)
    return flop / best_s / 1e12


def measure_hbm_bandwidth_tb_s(
    *,
    nbytes: int = 4 * 1024**3,
    warmup: int = 10,
    iters: int = 50,
    device: int = 0,
) -> float:
    """Peak HBM bandwidth (TB/s) from a large D2D copy, best of ``iters``.

    A copy of ``nbytes`` reads ``nbytes`` and writes ``nbytes``, so the traffic
    is ``2·nbytes``. Uses a bf16 buffer sized to ``nbytes``.
    """
    dev = torch.device(f"cuda:{device}")
    n_elems = nbytes // 2  # bf16 = 2 bytes
    src = torch.empty(n_elems, dtype=torch.bfloat16, device=dev)
    dst = torch.empty_like(src)
    for _ in range(warmup):
        dst.copy_(src)
    _sync()

    moved = 2.0 * src.numel() * src.element_size()  # read + write
    best_s = float("inf")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        dst.copy_(src)
        end.record()
        _sync()
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
    """Detected device facts + measured peak compute, bandwidth, ridge point."""
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
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    info = measure_roofline(
        device=args.device,
        gemm_size=args.gemm_size,
        copy_bytes=args.copy_bytes,
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
