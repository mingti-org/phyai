"""SM110 NVFP4 Gate+Up+GeGLU kernel for the pi0.5 language MLP."""

from __future__ import annotations

import os
import re
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch

from phyai_kernel.jit_utils import PHYAI_KERNEL_CUDA_CSRC_DIR, jit


SUPPORTED_M = frozenset({784, 816, 880, 968})
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 16384
MERGED_SIZE = 2 * INTERMEDIATE_SIZE
_MIN_CUTLASS_VERSION = (4, 4, 2)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def nvfp4_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    """Return FlashInfer/CUTLASS 128x4 scale storage for an NVFP4 matrix."""
    if rows <= 0 or cols <= 0 or cols % 16 != 0:
        raise ValueError("rows must be positive and cols must be divisible by 16")
    return ((rows + 127) // 128 * 128, (cols // 16 + 3) // 4 * 4)


def _cutlass_version(root: Path) -> tuple[int, int, int] | None:
    version_header = root / "include" / "cutlass" / "version.h"
    if not version_header.is_file():
        return None
    text = version_header.read_text(encoding="utf-8")
    values = []
    for field in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"^#define CUTLASS_{field}\s+(\d+)$", text, re.MULTILINE)
        if match is None:
            return None
        values.append(int(match.group(1)))
    return tuple(values)  # type: ignore[return-value]


def _resolve_cutlass_root() -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("PHYAI_CUTLASS_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(_REPOSITORY_ROOT / "third_party" / "mirage" / "deps" / "cutlass")

    flashinfer_spec = find_spec("flashinfer")
    if flashinfer_spec is not None and flashinfer_spec.submodule_search_locations:
        candidates.extend(
            Path(location) / "data" / "cutlass"
            for location in flashinfer_spec.submodule_search_locations
        )

    for candidate in candidates:
        resolved = candidate.resolve()
        version = _cutlass_version(resolved)
        if version is not None and version >= _MIN_CUTLASS_VERSION:
            return resolved
    return candidates[0].resolve()


_CUTLASS_ROOT = _resolve_cutlass_root()
_CUTLASS_INCLUDE_PATHS = (
    _CUTLASS_ROOT / "include",
    _CUTLASS_ROOT / "tools" / "util" / "include",
)


@jit(
    device="cuda",
    func_name="phyai_pi05_thor_nvfp4_geglu",
    cuda_files=["pi05_thor_nvfp4_geglu.cuh"],
    cpp_wrappers=[],
    cuda_wrappers=[("phyai_pi05_thor_nvfp4_geglu", "phyai_pi05_thor_nvfp4_geglu")],
    extra_cuda_cxx_flags=[
        "-DNDEBUG",
        "--use_fast_math",
        "-gencode=arch=compute_110a,code=sm_110a",
    ],
    extra_include_paths=[str(path) for path in _CUTLASS_INCLUDE_PATHS],
    build_directory=os.environ.get("PHYAI_PI05_THOR_NVFP4_BUILD_DIR"),
)
def _launch(
    compiled_module: Any,
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    interleaved_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = activation.shape[0]
    stream = int(torch.cuda.current_stream(activation.device).cuda_stream)
    return_code = compiled_module.phyai_pi05_thor_nvfp4_geglu(
        activation,
        activation_scale,
        interleaved_weight,
        weight_scale,
        workspace,
        output,
        output_scale,
        m,
        stream,
    )
    if return_code != 0:
        raise RuntimeError(
            f"pi0.5 Thor NVFP4 GeGLU failed with return code {return_code}"
        )
    return output, output_scale


def _contract_error(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    interleaved_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> str | None:
    tensors = (
        activation,
        activation_scale,
        interleaved_weight,
        weight_scale,
        workspace,
        output,
        output_scale,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        return "all tensors must be CUDA tensors"
    if any(tensor.requires_grad for tensor in tensors):
        return "the kernel is inference-only"
    if any(tensor.dtype != torch.uint8 for tensor in tensors):
        return "all packed values and scales must use uint8 storage"
    if any(not tensor.is_contiguous() for tensor in tensors):
        return "all tensors must be contiguous"
    if any(tensor.device != activation.device for tensor in tensors[1:]):
        return "all tensors must be on the same device"
    if activation.ndim != 2:
        return "activation must be a 2-D packed NVFP4 tensor"

    m = activation.shape[0]
    expected_shapes = {
        "activation": (m, HIDDEN_SIZE // 2),
        "activation_scale": nvfp4_scale_shape(m, HIDDEN_SIZE),
        "interleaved_weight": (MERGED_SIZE, HIDDEN_SIZE // 2),
        "weight_scale": nvfp4_scale_shape(MERGED_SIZE, HIDDEN_SIZE),
        "workspace": (m, INTERMEDIATE_SIZE),
        "output": (m, INTERMEDIATE_SIZE // 2),
        "output_scale": nvfp4_scale_shape(m, INTERMEDIATE_SIZE),
    }
    actual = {
        "activation": tuple(activation.shape),
        "activation_scale": tuple(activation_scale.shape),
        "interleaved_weight": tuple(interleaved_weight.shape),
        "weight_scale": tuple(weight_scale.shape),
        "workspace": tuple(workspace.shape),
        "output": tuple(output.shape),
        "output_scale": tuple(output_scale.shape),
    }
    if m not in SUPPORTED_M:
        return f"unsupported token count M={m}"
    for name, expected in expected_shapes.items():
        if actual[name] != expected:
            return f"{name} must have shape {expected}, got {actual[name]}"
    if torch.cuda.get_device_capability(activation.device) != (11, 0):
        return "the kernel requires SM110"
    version = _cutlass_version(_CUTLASS_ROOT)
    if version is None or version < _MIN_CUTLASS_VERSION:
        return "CUTLASS 4.4.2 or newer is required"
    return None


def supports_pi05_thor_nvfp4_geglu(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    interleaved_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy the fixed pi0.5 SM110 NVFP4 contract."""
    return (
        _contract_error(
            activation,
            activation_scale,
            interleaved_weight,
            weight_scale,
            workspace,
            output,
            output_scale,
        )
        is None
    )


def pi05_thor_nvfp4_geglu(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    interleaved_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fixed SM110 schedule into caller-owned NVFP4 output buffers.

    Inputs and weights must use CUTLASS/FlashInfer 128x4 scale storage and a
    global scale of one. Weight rows are pairwise interleaved as
    ``gate0, up0, gate1, up1, ...``. Caller-owned buffers keep allocations out
    of CUDA Graph capture.
    """
    error = _contract_error(
        activation,
        activation_scale,
        interleaved_weight,
        weight_scale,
        workspace,
        output,
        output_scale,
    )
    if error is not None:
        raise ValueError(error)
    return _launch(
        activation,
        activation_scale,
        interleaved_weight,
        weight_scale,
        workspace,
        output,
        output_scale,
    )


__all__ = [
    "HIDDEN_SIZE",
    "INTERMEDIATE_SIZE",
    "MERGED_SIZE",
    "SUPPORTED_M",
    "nvfp4_scale_shape",
    "pi05_thor_nvfp4_geglu",
    "supports_pi05_thor_nvfp4_geglu",
]
