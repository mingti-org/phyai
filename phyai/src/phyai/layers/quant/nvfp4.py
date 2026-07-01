"""Nvfp4Spec — packed E2M1 FP4 weight with FP8 block scales.

NVFP4 stores two E2M1 values per byte and uses one FP8-E4M3 scale for
every 16 values along the K dimension. The logical weight is still the
linear ``(N, K)`` matrix, but the stored parameter is ``(N, K // 2)``.

Two scale layouts are useful:

* ``"linear"`` — test/reference layout, ``(N, K // 16)``.
* ``"128x4"`` — FlashInfer / Blackwell GEMM layout, padded to
  ``(ceil(N, 128), ceil(K // 16, 4))``.

The activation path is handled by the backend because FlashInfer's FP4
GEMM needs both the per-block activation scales and a separate global
scale scalar, which does not fit the current :class:`ActivationView`
shape used by FP8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn

from phyai.layers.quant.base import AllocationRequest


_NVFP4_BLOCK_SIZE = 16
_FP8_E4M3_AMAX = 448.0
_NVFP4_MAX = 6.0
_E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _scale_shape(
    out_per_rank: int,
    in_per_rank: int,
    scale_layout: Literal["linear", "128x4"],
) -> tuple[int, int]:
    k_blocks = in_per_rank // _NVFP4_BLOCK_SIZE
    if scale_layout == "linear":
        return out_per_rank, k_blocks
    if scale_layout == "128x4":
        return _round_up(out_per_rank, 128), _round_up(k_blocks, 4)
    raise ValueError(
        f"Nvfp4Spec scale_layout must be 'linear' or '128x4', got {scale_layout!r}"
    )


def _per_tensor_amax_to_scale(amax: torch.Tensor) -> torch.Tensor:
    """Convert tensor amax to the TorchAO-style NVFP4 per-tensor scale."""
    return amax.float() / (_FP8_E4M3_AMAX * _NVFP4_MAX)


def _quantize_e2m1(data: torch.Tensor) -> torch.Tensor:
    thresholds = torch.tensor(_E2M1_THRESHOLDS, dtype=torch.float32, device=data.device)
    mag = torch.bucketize(data.abs(), thresholds).to(torch.uint8)
    sign = (data < 0).to(torch.uint8) << 3
    codes = (mag | sign).contiguous()
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def _quantize_nvfp4_linear(
    weight: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_tensor_scale = _per_tensor_amax_to_scale(weight.abs().amax()).clamp_min(1e-12)
    src = weight.float().reshape(weight.shape[0], -1, block_size)
    block_scale = src.abs().amax(dim=-1) / _NVFP4_MAX
    block_scale = (block_scale / per_tensor_scale).clamp(
        torch.finfo(torch.float8_e4m3fn).tiny,
        _FP8_E4M3_AMAX,
    )
    block_scale_fp8 = block_scale.to(torch.float8_e4m3fn)
    reciprocal = (1.0 / per_tensor_scale) / block_scale_fp8.float()
    scaled = (src * reciprocal.unsqueeze(-1)).clamp(-_NVFP4_MAX, _NVFP4_MAX)
    packed = _quantize_e2m1(scaled.reshape(weight.shape))
    return packed, block_scale_fp8, per_tensor_scale.reshape(1)


@dataclass
class Nvfp4Spec:
    """NVFP4 Linear weight spec.

    ``scale_layout="128x4"`` is the production layout consumed by
    FlashInfer's ``mm_fp4`` backend. ``"linear"`` is kept for the Torch
    reference path and small CPU tests.
    """

    scale_layout: Literal["linear", "128x4"] = "128x4"
    weight_dtype: torch.dtype = torch.uint8
    block_size: int = _NVFP4_BLOCK_SIZE

    def __post_init__(self) -> None:
        if self.block_size != _NVFP4_BLOCK_SIZE:
            raise ValueError("Nvfp4Spec only supports block_size=16")
        if self.scale_layout not in ("linear", "128x4"):
            raise ValueError(
                f"Nvfp4Spec scale_layout must be 'linear' or '128x4', "
                f"got {self.scale_layout!r}"
            )

    @property
    def spec_id(self) -> str:
        return f"nvfp4_block_{self.block_size}_{self.scale_layout}"

    def allocate(self, layer: nn.Module, request: AllocationRequest) -> None:
        if len(request.weight_shape) != 2:
            raise ValueError(
                f"Nvfp4Spec.allocate expects a 2-D weight_shape (N, K), "
                f"got {request.weight_shape!r}"
            )
        out_per_rank, in_per_rank = request.weight_shape
        if in_per_rank % self.block_size != 0:
            raise ValueError(
                f"Nvfp4Spec: in_per_rank={in_per_rank} not divisible by "
                f"block_size={self.block_size}"
            )
        device = request.device

        layer.weight = nn.Parameter(
            torch.empty(
                out_per_rank,
                in_per_rank // 2,
                dtype=self.weight_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        layer.weight_scale = nn.Parameter(
            torch.ones(
                _scale_shape(out_per_rank, in_per_rank, self.scale_layout),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        layer.weight_global_scale = nn.Parameter(
            torch.ones(1, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        layer._nvfp4_pending_weight = None
        layer.logical_widths = list(request.logical_widths)

    def process_after_loading(self, layer: nn.Module) -> None:
        pending = getattr(layer, "_nvfp4_pending_weight", None)
        if pending is not None:
            print("Nvfp4Spec: quantizing loaded weight to packed NVFP4")
            self.quantize_loaded_weight(layer, pending)
            layer._nvfp4_pending_weight = None
        return None

    def load_weight(
        self,
        layer: nn.Module,
        loaded: torch.Tensor,
        shard_id: "int | str | None",
        default_loader: Callable[[nn.Parameter, torch.Tensor, object], None],
    ) -> None:
        logical_shape = (layer.weight.shape[0], layer.weight.shape[1] * 2)
        if loaded.dtype in (torch.float16, torch.bfloat16, torch.float32):
            pending = getattr(layer, "_nvfp4_pending_weight", None)
            if pending is None:
                pending = torch.empty(
                    logical_shape,
                    dtype=loaded.dtype,
                    device=loaded.device,
                )
                layer._nvfp4_pending_weight = pending
            default_loader(pending, loaded, shard_id)
            return

        default_loader(layer.weight, loaded, shard_id)

    def quantize_loaded_weight(self, layer: nn.Module, weight: torch.Tensor) -> None:
        src = weight.detach().to(device=layer.weight.device).contiguous()
        if src.dtype not in (torch.bfloat16, torch.float32, torch.float16):
            src = src.to(torch.bfloat16)

        if self.scale_layout == "linear":
            packed, scale, global_scale = _quantize_nvfp4_linear(src, self.block_size)
        elif self.scale_layout == "128x4":
            if not src.is_cuda:
                raise RuntimeError(
                    "Nvfp4Spec(scale_layout='128x4') needs CUDA FlashInfer to "
                    "quantize high-precision weights on load. Use "
                    "scale_layout='linear' for CPU reference quantization or load "
                    "a pre-quantized 128x4 checkpoint."
                )
            from flashinfer.quantization import SfLayout, nvfp4_quantize

            flashinfer_global_scale = (_FP8_E4M3_AMAX * _NVFP4_MAX) / (
                src.float().abs().amax().clamp_min(1e-12)
            )
            flashinfer_global_scale = flashinfer_global_scale.reshape(1)
            if src.dtype == torch.float32:
                src = src.to(torch.bfloat16)
            packed, scale = nvfp4_quantize(
                src,
                flashinfer_global_scale,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
                enable_pdl=False,
            )
            global_scale = 1.0 / flashinfer_global_scale
        else:
            raise RuntimeError(
                f"unhandled Nvfp4Spec scale_layout={self.scale_layout!r}"
            )

        layer.weight.data.copy_(packed.to(device=layer.weight.device).view(torch.uint8))
        layer.weight_scale.data.copy_(
            scale.to(device=layer.weight_scale.device).view(layer.weight_scale.dtype)
        )
        layer.weight_global_scale.data.copy_(
            global_scale.to(device=layer.weight_global_scale.device)
        )


__all__ = ["Nvfp4Spec"]
