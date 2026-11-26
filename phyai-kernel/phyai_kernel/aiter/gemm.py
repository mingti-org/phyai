"""aiter bf16 GEMM with tuned shapes shipped beside this module.

``aiter_bf16_tuned_gemm.csv`` holds solutions tuned for PI0.5 on MI300X
(gfx942, 304 CUs) with aiter ``285579b621``; other shapes, devices and aiter
versions fall back to Torch inside ``tgemm``.
"""

from __future__ import annotations

import os
import threading
from importlib import resources

import torch

_ENV = "AITER_CONFIG_GEMM_BF16"
_lock = threading.Lock()
_configured = False


def ensure_tuned_config() -> None:
    """Add the shipped tuned shapes to aiter's bf16 GEMM config, once per process.

    aiter merges every CSV listed in ``AITER_CONFIG_GEMM_BF16`` and memoizes
    the merged file and per-shape lookups on first use, so the caches are
    cleared after the variable is extended. A failure restores the variable
    and leaves the next call to retry.
    """

    global _configured
    with _lock:
        if _configured:
            return
        import aiter.jit.core as aiter_core
        from aiter import tuned_gemm

        tuned = str(resources.files(__package__) / "aiter_bf16_tuned_gemm.csv")
        previous = os.environ.get(_ENV)
        configured = aiter_core.AITER_CONFIG_GEMM_BF16 if previous is None else previous
        if tuned not in configured.split(os.pathsep):
            try:
                os.environ[_ENV] = os.pathsep.join((configured, tuned))
                type(aiter_core.AITER_CONFIGS).get_config_file.cache_clear()
                tuned_gemm.get_GEMM_A16W16_config_.cache_clear()
                tuned_gemm.get_GEMM_A16W16_config.cache_clear()
            except BaseException:
                if previous is None:
                    os.environ.pop(_ENV, None)
                else:
                    os.environ[_ENV] = previous
                raise
        _configured = True


def mm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Return ``x @ weight.T + bias`` for bf16 ``x`` and ``weight (N, K)``."""

    if x.numel() == 0:
        return x.new_empty((*x.shape[:-1], weight.shape[0]))
    from aiter import tuned_gemm

    return tuned_gemm.tgemm.mm(x, weight, bias)


__all__ = ["ensure_tuned_config", "mm"]
