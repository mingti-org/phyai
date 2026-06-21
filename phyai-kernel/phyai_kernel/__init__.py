"""phyai-kernel — JIT-compiled CPU/CUDA kernels for phyai via tvm-ffi."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from phyai_kernel import jit_utils
from phyai_kernel.jit_utils import jit
from phyai_kernel.triton import (
    adarmsnorm,
    create_paged_kv_indices,
    fused_add_rmsnorm,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    layernorm,
    masked_embedding_lookup,
    rmsnorm,
    rmsnorm_hf,
)

try:
    __version__ = _pkg_version("phyai-kernel")
except PackageNotFoundError:  # raw source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "adarmsnorm",
    "create_paged_kv_indices",
    "fused_add_rmsnorm",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "jit",
    "jit_utils",
    "layernorm",
    "masked_embedding_lookup",
    "rmsnorm",
    "rmsnorm_hf",
]
