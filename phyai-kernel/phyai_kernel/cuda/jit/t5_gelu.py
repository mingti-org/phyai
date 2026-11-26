"""CuTe CUDA T5 GELU with the rounding of the eager half-precision expression."""

from __future__ import annotations

import torch

from phyai_kernel.jit_utils import cache_once, load_cuda_jit


@cache_once
def _kernel():
    from flashinfer.jit.env import CUTLASS_INCLUDE_DIRS

    return load_cuda_jit(
        "t5_gelu",
        cuda_files=["t5_gelu.cuh"],
        cuda_wrappers=[("run", "phyai_kernel::t5_gelu")],
        extra_include_paths=[str(path) for path in CUTLASS_INCLUDE_DIRS],
        extra_cuda_cxx_flags=["--fmad=false", "-lineinfo"],
    )


def _run(x: torch.Tensor, out: torch.Tensor) -> None:
    _kernel().run(x, out)


_LIBRARY = torch.library.Library("phyai_kernel", "FRAGMENT")
_LIBRARY.define("t5_gelu_out(Tensor x, Tensor(a!) out) -> ()")
_LIBRARY.impl("t5_gelu_out", _run, "CUDA")


@torch.library.register_fake("phyai_kernel::t5_gelu_out")
def _fake(x: torch.Tensor, out: torch.Tensor) -> None:
    return None


def t5_gelu(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Apply Wan/T5's tanh GELU to a CUDA float16 or bfloat16 tensor.

    Contiguous tensors and strided views are read directly, including slices,
    transposes, storage offsets, and zero strides. The output is contiguous.
    An optional ``out`` must match x's shape, dtype and device, be contiguous,
    and not overlap x. Warm up once before CUDA graph capture to finish JIT
    compilation. CUTLASS headers come from the installed FlashInfer package.
    """
    if out is None:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
    torch.ops.phyai_kernel.t5_gelu_out(x, out)
    return out


__all__ = ["t5_gelu"]
