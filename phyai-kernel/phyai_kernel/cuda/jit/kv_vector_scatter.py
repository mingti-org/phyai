"""Thor BF16 vector K/V writes for a canonical Pi0.5 Graph cache.

This is not a replacement for generic indexed assignment: slot zero is an
unconsumed padding sentinel, and positive slot indices must be unique and less
than 1019. Dispatch remains unchanged unless a caller explicitly uses this module.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch


def span_supported(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    pointer: int,
    offset: int,
    storage_bytes: int,
    *,
    cache: bool = False,
) -> bool:
    if cache:
        layout = tuple(shape) == (1019, 1, 1, 256) and tuple(stride) == (
            256,
            256,
            256,
            1,
        )
        rows = 1019
    else:
        layout = (
            len(shape) == len(stride) == 3
            and 1 <= shape[0] <= 4096
            and tuple(shape[1:]) == (1, 256)
            and tuple(stride[1:]) == (256, 1)
            and 256 <= stride[0] <= 2**24
            and stride[0] % 8 == 0
        )
        rows = shape[0] if layout else 0
    return bool(
        layout
        and pointer > 0
        and pointer % 16 == 0
        and offset >= 0
        and 2 * (offset + (rows - 1) * stride[0] + 256) <= storage_bytes
    )


def slot_span_supported(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    pointer: int,
    offset: int,
    storage_bytes: int,
) -> bool:
    return (
        len(shape) == 1
        and 1 <= shape[0] <= 4096
        and tuple(stride) == (1,)
        and pointer > 0
        and pointer % 8 == 0
        and offset >= 0
        and 8 * (offset + shape[0]) <= storage_bytes
    )


def validate_slots(slots: torch.Tensor) -> None:
    """Check a CPU slot map before copying it to the captured device buffer.

    Repeated zeros are allowed. This function performs no device transfer or
    synchronization and must be called again when the host slot values change.
    The caller owns the matching device copy and its stream ordering.
    """
    import torch

    if (
        slots.device.type != "cpu"
        or slots.dtype != torch.int64
        or not slot_span_supported(
            slots.shape,
            slots.stride(),
            slots.data_ptr(),
            slots.storage_offset(),
            slots.untyped_storage().nbytes(),
        )
    ):
        raise ValueError("contiguous CPU int64 slot map required")
    values = slots.tolist()
    if any(value < 0 or value >= 1019 for value in values):
        raise ValueError("slot index outside [0, 1019)")
    positive = [value for value in values if value]
    if len(set(positive)) != len(positive):
        raise ValueError("positive slot indices must be unique")


def validate(
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    import torch

    spans = []
    for index, t in enumerate((k, v, kc, vc)):
        if (
            not t.is_cuda
            or t.device != k.device
            or t.dtype != torch.bfloat16
            or t.requires_grad
            or not span_supported(
                t.shape,
                t.stride(),
                t.data_ptr(),
                t.storage_offset(),
                t.untyped_storage().nbytes(),
                cache=index >= 2,
            )
        ):
            raise ValueError("unsupported actual K/V vector view")
        spans.append(
            (t.data_ptr(), t.data_ptr() + 2 * ((t.shape[0] - 1) * t.stride(0) + 256))
        )
    if k.shape != v.shape or slots.shape != (k.shape[0],):
        raise ValueError("source/slot row mismatch")
    if (
        slots.device != k.device
        or slots.dtype != torch.int64
        or not slot_span_supported(
            slots.shape,
            slots.stride(),
            slots.data_ptr(),
            slots.storage_offset(),
            slots.untyped_storage().nbytes(),
        )
    ):
        raise ValueError("aligned same-device int64 slot vector required")
    spans.append((slots.data_ptr(), slots.data_ptr() + slots.numel() * 8))
    for i in (2, 3):
        if any(
            i != j and spans[i][0] < spans[j][1] and spans[j][0] < spans[i][1]
            for j in range(5)
        ):
            raise ValueError("cache aliases another input/output")


@lru_cache(maxsize=16)
def entrypoint(directory: str | None = None) -> Callable[..., None]:
    from phyai_kernel.jit_utils.compile import jit

    @jit(
        device="cuda",
        func_name="kv_vector_scatter",
        cuda_files=["kv_vector_scatter.cuh"],
        cpp_wrappers=[],
        cuda_wrappers=[("kv_vector_scatter", "kv_vector_scatter")],
        extra_cuda_cxx_flags=["-gencode=arch=compute_110,code=sm_110"],
        build_directory=directory,
    )
    def invoke(module, k, v, kc, vc, slots, stream):
        module.kv_vector_scatter(k, v, kc, vc, slots, int(stream))

    return invoke


def prepare(
    directory: str | Path | None = None,
    *,
    device: torch.device | str | None = None,
) -> Callable[..., None]:
    """Warm the strict SM110 kernel outside capture using the normal JIT cache."""
    import torch

    target = torch.device("cuda" if device is None else device)
    if target.type != "cuda":
        raise ValueError("CUDA device required")
    with torch.cuda.device(target):
        return _prepare_current_device(directory)


def _prepare_current_device(directory: str | Path | None) -> Callable[..., None]:
    import torch

    if (
        torch.cuda.is_current_stream_capturing()
        or torch.cuda.get_device_capability() != (11, 0)
    ):
        raise RuntimeError("prepare strict Thor JIT before capture")
    call = entrypoint(None if directory is None else str(directory))
    k, v = [
        torch.zeros((1, 1, 256), device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    kc, vc = [
        torch.zeros((1019, 1, 1, 256), device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    slots = torch.ones(1, device="cuda", dtype=torch.int64)
    call(k, v, kc, vc, slots, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return call


def run(
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    slots: torch.Tensor,
    call: Callable[..., None],
) -> None:
    """Write K/V rows to the two caches on the current stream of their device.

    Device slot values must match a host map accepted by :func:`validate_slots`,
    or have the same guarantees from their scheduler owner. This function only
    checks metadata, storage spans and aliases; it does not copy slot values to
    the CPU. Cache row zero is preserved. Retain the callable and all tensors
    until every captured Graph is released; their addresses must stay stable.
    ``call`` comes from :func:`prepare` for the same device. There is no
    allocation or fallback.
    """
    import torch

    validate(k, v, kc, vc, slots)
    with torch.cuda.device(k.device):
        call(k, v, kc, vc, slots, torch.cuda.current_stream().cuda_stream)
