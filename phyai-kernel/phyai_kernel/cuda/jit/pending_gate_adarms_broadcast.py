"""Thor BF16 gated residual/AdaRMS with caller-owned Graph workspace.

The residual is rounded to BF16 before the FP32 RMS reduction and affine.
This module does not select itself or change the model's default dispatch.
"""

from __future__ import annotations

from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch


def row_span_supported(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    pointer: int,
    storage_bytes: int,
    offset: int,
    width: int,
) -> bool:
    return (
        len(shape) == len(stride) == 2
        and 1 <= shape[0] <= 4096
        and shape[1] == width
        and stride[1] == 1
        and width <= stride[0] <= 2**24
        and pointer > 0
        and pointer % 2 == 0
        and offset >= 0
        and 2 * (offset + (shape[0] - 1) * stride[0] + width) <= storage_bytes
    )


@lru_cache(maxsize=16)
def entrypoint(directory: str | None = None) -> Callable[..., None]:
    from phyai_kernel.jit_utils.compile import jit

    @jit(
        device="cuda",
        func_name="pending_gate_adarms",
        cuda_files=["pending_gate_adarms_broadcast.cuh"],
        cpp_wrappers=[],
        cuda_wrappers=[("pending_gate_adarms", "pending_gate_adarms")],
        extra_cuda_cxx_flags=["-gencode=arch=compute_110,code=sm_110"],
        build_directory=directory,
    )
    def invoke(module, a, b, g, m, h, y, eps, stream):
        module.pending_gate_adarms(a, b, g, m, h, y, float(eps), int(stream))

    return invoke


def validate(
    a: torch.Tensor,
    b: torch.Tensor,
    g: torch.Tensor,
    m: torch.Tensor,
    h: torch.Tensor,
    y: torch.Tensor,
) -> None:
    import torch

    tensors = (a, b, g, m, h, y)
    if not a.is_cuda or any(
        t.dtype != torch.bfloat16 or t.device != a.device or t.requires_grad
        for t in tensors
    ):
        raise ValueError("CUDA BF16 inference tensors required")
    spans = []
    for t in tensors:
        width = 3072 if t is m else 1024
        if not row_span_supported(
            t.shape,
            t.stride(),
            t.data_ptr(),
            t.untyped_storage().nbytes(),
            t.storage_offset(),
            width,
        ):
            raise ValueError("unsupported actual view storage/stride")
        n = (t.shape[0] - 1) * t.stride(0) + width
        spans.append((t.data_ptr(), t.data_ptr() + 2 * n))
    if (
        b.shape != a.shape
        or h.shape != a.shape
        or y.shape != a.shape
        or a.shape[0] % g.shape[0]
        or a.shape[0] % m.shape[0]
        or not h.is_contiguous()
        or not y.is_contiguous()
    ):
        raise ValueError("unsupported output or broadcast rows")
    for i in (4, 5):
        if any(
            i != j and spans[i][0] < spans[j][1] and spans[j][0] < spans[i][1]
            for j in range(6)
        ):
            raise ValueError(
                "output alias forbidden; readonly gate/modulation alias allowed"
            )


def prepare(
    directory: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
) -> Callable[..., None]:
    """Compile and warm the four launch variants before Graph capture.

    Omit ``directory`` to use the normal PhyAI JIT cache. The returned callable
    is used by :func:`run`; it owns no tensor workspace. Retain the callable,
    inputs and both output buffers until every captured Graph is released.
    """
    import torch

    target = torch.device("cuda" if device is None else device)
    if target.type != "cuda":
        raise ValueError("CUDA device required")
    with torch.cuda.device(target):
        return _prepare_current_device(directory)


def _prepare_current_device(directory: str | Path | None) -> Callable[..., None]:
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare in capture forbidden")
    if torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("strict Thor target required")
    call = entrypoint(None if directory is None else str(directory))
    a, b, h, y = [
        torch.zeros((1, 1024), device="cuda", dtype=torch.bfloat16) for _ in range(4)
    ]
    g = torch.zeros((1, 1024), device="cuda", dtype=torch.bfloat16)
    m = torch.zeros((1, 3072), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream().cuda_stream
    call(a, b, g, m, h, y, 1e-6, stream)
    odd_gate = torch.zeros(1025, device="cuda", dtype=torch.bfloat16)[1:].view(1, 1024)
    call(a, b, odd_gate, m, h, y, 1e-6, stream)
    # Warm both general-row specializations before any possible Graph capture.
    a, b, h, y = [
        torch.zeros((2, 1024), device="cuda", dtype=torch.bfloat16) for _ in range(4)
    ]
    g = torch.zeros((2, 1024), device="cuda", dtype=torch.bfloat16)
    m = torch.zeros((2, 3072), device="cuda", dtype=torch.bfloat16)
    call(a, b, g, m, h, y, 1e-6, stream)
    odd_gate = torch.zeros(
        2 * 1025 + 1, device="cuda", dtype=torch.bfloat16
    ).as_strided((2, 1024), (1025, 1), 1)
    call(a, b, odd_gate, m, h, y, 1e-6, stream)
    torch.cuda.synchronize()
    return call


def run(
    a: torch.Tensor,
    b: torch.Tensor,
    g: torch.Tensor,
    m: torch.Tensor,
    h: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    call: Callable[..., None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write the rounded residual to ``h`` and modulated RMSNorm to ``y``.

    ``a``, ``b``, ``h`` and ``y`` are ``(rows, 1024)`` CUDA BF16 tensors.
    Gate ``g`` and modulation ``m`` have widths 1024 and 3072; each row is
    broadcast across a consecutive group of input rows. Read-only inputs may
    alias, including a gate view into modulation. Outputs must be contiguous,
    mutually disjoint and disjoint from all inputs. No workspace is allocated.
    ``call`` must have been returned by :func:`prepare` for this device.
    """
    import torch

    epsilon = float(eps)
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    validate(a, b, g, m, h, y)
    with torch.cuda.device(a.device):
        call(a, b, g, m, h, y, epsilon, torch.cuda.current_stream().cuda_stream)
    return h, y
