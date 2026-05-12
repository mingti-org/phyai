"""phyai-kernel — JIT-compiled CPU/CUDA kernels for phyai via tvm-ffi."""

from phyai_kernel import jit_utils
from phyai_kernel.jit_utils import jit
from phyai_kernel.triton import (
    fused_add_rmsnorm,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
    rmsnorm_hf,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "fused_add_rmsnorm",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "jit",
    "jit_utils",
    "rmsnorm",
    "rmsnorm_hf",
]
