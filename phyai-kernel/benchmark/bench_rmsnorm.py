# Benchmark scaffolding adapted from SGLang
# (https://github.com/sgl-project/sglang), Copyright 2023-2024 SGLang Team,
# licensed under the Apache License, Version 2.0:
#     http://www.apache.org/licenses/LICENSE-2.0
"""Benchmark phyai_kernel.rmsnorm against ``torch.nn.RMSNorm`` and the HF eager fallback.

Inspired by SGLang's ``sgl-kernel`` RMSNorm benchmark. Run::

    python benchmark/bench_rmsnorm.py
    python benchmark/bench_rmsnorm.py --use_residual --dtype bf16
    python benchmark/bench_rmsnorm.py --variant gemma

Prints latency in microseconds (median across many iters) and the achieved
GB/s based on the read+write traffic each variant performs. ``torch.nn.RMSNorm``
serves as the reference baseline.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from typing import Callable, List, Optional, Tuple

import torch
import triton

import phyai_kernel

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_VARIANTS = ("rmsnorm", "rmsnorm_hf", "gemma", "fused_add", "gemma_fused_add")


# --------------------------------------------------------------------------- #
# HF eager reference (matches sglang/HF semantics).                           #
# --------------------------------------------------------------------------- #


class HFRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        orig = x.dtype
        xf = x.to(torch.float32)
        if residual is not None:
            xf = xf + residual.to(torch.float32)
            residual = xf.to(orig)
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        out = xf.to(orig) * self.weight
        if residual is None:
            return out
        return out, residual


# --------------------------------------------------------------------------- #
# Closures for each provider × variant.                                       #
# --------------------------------------------------------------------------- #


def _build_phyai_fn(
    variant: str,
) -> Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float], None]:
    if variant == "rmsnorm":
        return lambda x, w, r, eps: phyai_kernel.rmsnorm(x, w, eps)
    if variant == "rmsnorm_hf":
        return lambda x, w, r, eps: phyai_kernel.rmsnorm_hf(x, w, eps)
    if variant == "gemma":
        return lambda x, w, r, eps: phyai_kernel.gemma_rmsnorm(x, w, eps)
    if variant == "fused_add":
        return lambda x, w, r, eps: phyai_kernel.fused_add_rmsnorm(
            x.clone(), r.clone(), w, eps
        )
    if variant == "gemma_fused_add":
        return lambda x, w, r, eps: phyai_kernel.gemma_fused_add_rmsnorm(
            x.clone(), r.clone(), w, eps
        )
    raise ValueError(variant)


def _build_torch_nn_fn(
    variant: str, hidden_size: int, dtype: torch.dtype
) -> Optional[Callable]:
    """Wrap ``torch.nn.RMSNorm`` for the variants it can express directly."""
    if variant == "rmsnorm":
        m = torch.nn.RMSNorm(hidden_size, eps=1e-6, device="cuda", dtype=dtype)
        return lambda x, w, r, eps: m(x)
    # torch.nn.RMSNorm has no residual fusion or `(1+w)` form; skip.
    return None


def _build_hf_fn(variant: str, hidden_size: int, dtype: torch.dtype) -> Callable:
    m = HFRMSNorm(hidden_size).to(device="cuda", dtype=dtype)
    use_residual = variant.startswith("fused_add") or variant.startswith("gemma_fused")
    if variant == "gemma" or variant == "gemma_fused_add":
        # HFRMSNorm doesn't model `1+w`; emulate by precomputing weight = 1+w.
        # (caller passes the same weight, so we just leave it alone — close enough
        # for a perf comparison where the multiply count is identical.)
        pass
    if use_residual:
        return lambda x, w, r, eps: m(x.clone(), r.clone())
    return lambda x, w, r, eps: m(x)


# --------------------------------------------------------------------------- #
# Bench driver.                                                               #
# --------------------------------------------------------------------------- #


def _bytes_per_token(variant: str, n_cols: int, dtype: torch.dtype) -> int:
    """HBM traffic per token. Weight reads are dropped: with reuse across
    tokens they hit L2 cache and don't show up at the memory controller."""
    bw = torch.tensor([], dtype=dtype).element_size()
    if variant in ("rmsnorm", "rmsnorm_hf", "gemma"):
        # read x + write out
        return 2 * n_cols * bw
    # fused-add: read x + read residual + write residual + write x
    return 4 * n_cols * bw


def _bench_one(
    fn: Callable,
    args: Tuple,
    warmup: int = 25,
    iters: int = 100,
) -> float:
    """Return median latency in microseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(lambda: fn(*args), warmup=10, rep=iters)
    return ms * 1000.0


def _make_inputs(
    n_rows: int, n_cols: int, dtype: torch.dtype, with_residual: bool
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=dtype)
    w = torch.randn(n_cols, device="cuda", dtype=dtype) * 0.1 + 1.0
    r = torch.randn_like(x) if with_residual else None
    return x, w, r


def main() -> int:
    parser = argparse.ArgumentParser("phyai-kernel RMSNorm benchmark")
    parser.add_argument("--variant", choices=_VARIANTS, default="rmsnorm")
    parser.add_argument(
        "--dtype",
        choices=tuple(_DTYPES),
        default="bf16",
        help="activation dtype (default bf16)",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=str,
        default="896,2048,3584,4096,4608,8192",
        help="comma-separated hidden sizes (Qwen/Gemma typical)",
    )
    parser.add_argument(
        "--num_tokens",
        type=str,
        default="64,512,4096",
        help="comma-separated num_tokens (= batch * seq_len)",
    )
    parser.add_argument(
        "--use_residual",
        action="store_true",
        help="run with a residual input (fused_add* only)",
    )
    args = parser.parse_args()

    dtype = _DTYPES[args.dtype]
    hidden_sizes: List[int] = [int(x) for x in args.hidden_sizes.split(",") if x]
    token_counts: List[int] = [int(x) for x in args.num_tokens.split(",") if x]

    variant = args.variant
    needs_residual = variant in ("fused_add", "gemma_fused_add") or args.use_residual
    if variant in ("rmsnorm", "rmsnorm_hf", "gemma") and args.use_residual:
        # In sglang, residual fusion is a separate variant, not a flag — so
        # silently coerce to the fused variant for clarity.
        variant = (
            "fused_add"
            if variant == "rmsnorm"
            else ("gemma_fused_add" if variant == "gemma" else variant)
        )
        needs_residual = True

    print(
        f"variant={variant}  dtype={args.dtype}  use_residual={needs_residual}\n"
        f"--------------------------------------------------------------------"
    )
    header = (
        f"{'tokens':>7} {'hidden':>7} | "
        f"{'phyai (µs)':>11} {'torch (µs)':>11} {'HF (µs)':>10} | "
        f"{'phyai TB/s':>10} {'torch TB/s':>10} | "
        f"{'speedup vs torch':>17}"
    )
    print(header)
    print("-" * len(header))

    phyai_fn = _build_phyai_fn(variant)

    for tokens, hidden in itertools.product(token_counts, hidden_sizes):
        x, w, r = _make_inputs(tokens, hidden, dtype, needs_residual)

        torch_fn = _build_torch_nn_fn(variant, hidden, dtype)
        hf_fn = _build_hf_fn(variant, hidden, dtype)

        t_phyai = _bench_one(phyai_fn, (x, w, r, 1e-6))
        t_torch = _bench_one(torch_fn, (x, w, r, 1e-6)) if torch_fn else float("nan")
        t_hf = _bench_one(hf_fn, (x, w, r, 1e-6))

        bytes_total = _bytes_per_token(variant, hidden, dtype) * tokens
        # bytes / (microseconds * 1e-6 s/µs) / 1e12 B/TB
        tbps_phyai = bytes_total / (t_phyai * 1e6)
        tbps_torch = bytes_total / (t_torch * 1e6) if torch_fn else float("nan")

        speedup = t_torch / t_phyai if torch_fn else float("nan")
        print(
            f"{tokens:>7d} {hidden:>7d} | "
            f"{t_phyai:>10.2f}  {t_torch:>10.2f}  {t_hf:>9.2f} | "
            f"{tbps_phyai:>9.2f}  {tbps_torch:>9.2f} | "
            f"{speedup:>16.2f}x"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
