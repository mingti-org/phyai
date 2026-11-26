"""CUDA kernels compiled on first use through TVM-FFI."""

from phyai_kernel.cuda.jit.t5_gelu import t5_gelu

__all__ = ["t5_gelu"]
