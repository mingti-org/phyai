"""Load-time quantization and packed storage for Humming linear kernels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn

from phyai.kernel.types import PhysicalSignature
from phyai.layers.quant.base import AllocationRequest
from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.scheme import QDType, QuantScheme


_WEIGHT_DTYPES = {
    QDType.INT4: "int4",
    QDType.INT8: "int8",
    QDType.FP8_E4M3: "float8e4m3",
    QDType.FP8_E5M2: "float8e5m2",
    QDType.NVFP4: "float4e2m1",
    QDType.MXFP4: "float4e2m1",
}
_ACTIVATION_PAIRS = {
    QDType.INT4: {QDType.INT8, QDType.FP8_E4M3},
    QDType.INT8: {QDType.INT8},
    QDType.FP8_E4M3: {QDType.FP8_E4M3},
    QDType.FP8_E5M2: {QDType.FP8_E5M2},
}
_DENSE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True)
class HummingSpec:
    """Quantize complete local weight matrices after checkpoint loading.

    The existing weight loader assembles TP shards and fused projections into
    a CPU staging tensor. Only a complete matrix is transferred and packed.
    """

    scheme: QuantScheme
    weight_dtype: torch.dtype = field(default=torch.int32, init=False)

    def __post_init__(self) -> None:
        weight = self.scheme.weight
        if not self.scheme.online:
            raise ValueError("HummingSpec requires online=True for load-time PTQ")
        if weight.dtype not in _WEIGHT_DTYPES:
            raise ValueError(f"Humming does not support weight dtype {weight.dtype!r}")
        if not weight.symmetric or weight.dynamic:
            raise ValueError(
                "Humming PTQ requires symmetric, static weight quantization"
            )
        if weight.granularity not in (
            Granularity.PER_CHANNEL,
            Granularity.PER_TENSOR,
            Granularity.BLOCK,
        ):
            raise ValueError(f"Unsupported Humming granularity {weight.granularity!r}")
        block = weight.block_shape
        if block is not None and (
            not isinstance(block, (tuple, list))
            or len(block) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in block
            )
        ):
            raise ValueError("Humming block_shape must contain two integer dimensions")
        if weight.dtype in (QDType.NVFP4, QDType.MXFP4):
            expected = (1, 16 if weight.dtype is QDType.NVFP4 else 32)
            if (
                weight.granularity is not Granularity.BLOCK
                or weight.block_shape
                not in (
                    None,
                    expected,
                )
            ):
                raise ValueError(
                    f"{weight.dtype.value} requires block_shape={expected}"
                )
        elif weight.micro_scaled:
            raise ValueError("micro_scaled is supported only for NVFP4 and MXFP4")
        elif weight.granularity is Granularity.BLOCK:
            if block is None:
                raise ValueError(
                    "Humming block weight granularity requires block_shape"
                )
            block_n = block[0]
            if weight.dtype in (QDType.FP8_E4M3, QDType.FP8_E5M2):
                if block_n != 1 and (block_n < 64 or block_n & (block_n - 1)):
                    raise ValueError(
                        "Humming FP8 block N must be 1 or a power of two >= 64"
                    )
            elif block_n != 1:
                raise ValueError(
                    "Humming grouped weights require block_shape=(1, group_size)"
                )
        elif weight.block_shape is not None:
            raise ValueError("block_shape requires block weight granularity")

        activation = self.scheme.input
        if activation is not None:
            if activation.dtype not in _ACTIVATION_PAIRS.get(weight.dtype, set()):
                raise ValueError(
                    f"Unsupported Humming weight/activation pair: "
                    f"{weight.dtype.value}/{activation.dtype.value}"
                )
            if (
                not activation.dynamic
                or not activation.symmetric
                or activation.micro_scaled
                or activation.granularity is not Granularity.PER_CHANNEL
                or activation.block_shape is not None
            ):
                raise ValueError(
                    "Humming A8 requires symmetric dynamic per-token activations"
                )

        group_size = self.group_size
        minimum = 32 if activation is not None else 16
        if weight.granularity is Granularity.BLOCK and (
            isinstance(group_size, bool)
            or not isinstance(group_size, int)
            or group_size < minimum
            or group_size & (group_size - 1)
        ):
            raise ValueError(f"Humming group_size must be a power of two >= {minimum}")

    @property
    def group_size(self) -> int:
        if self.scheme.weight.dtype is QDType.NVFP4:
            return 16
        if self.scheme.weight.dtype is QDType.MXFP4:
            return 32
        block = self.scheme.weight.block_shape
        return block[1] if block is not None else 0

    @property
    def activation(self) -> str:
        return "a16" if self.scheme.input is None else self.scheme.input.dtype.value

    @property
    def block_shape(self) -> tuple[int, int] | None:
        if not self.group_size:
            return None
        block = self.scheme.weight.block_shape
        return tuple(block) if block is not None else (1, self.group_size)

    @property
    def physical_signature(self) -> PhysicalSignature:
        weight = self.scheme.weight
        block = self.block_shape
        scale_type = "channel"
        scale_dtype = None
        if weight.granularity is Granularity.PER_TENSOR:
            scale_type, scale_dtype = "tensor", "fp32"
        elif block is not None:
            scale_type = "block" if block[0] > 1 else "group"
            if block[0] > 1:
                scale_dtype = "fp32"
        if weight.dtype is QDType.NVFP4:
            scale_dtype = "fp8_e4m3"
        elif weight.dtype is QDType.MXFP4:
            scale_dtype = "fp8_e8m0"
        return PhysicalSignature(
            format=weight.dtype.value,
            layout="humming",
            granularity=weight.granularity.value,
            block_shape=block,
            scale_dtype=scale_dtype,
            storage_dtype="int32",
            fields={
                "activation": self.activation,
                "scale_type": scale_type,
                "scale_2_type": "tensor" if weight.dtype is QDType.NVFP4 else "none",
            },
        )

    @property
    def spec_id(self) -> str:
        block = self.block_shape
        granularity = self.scheme.weight.granularity.value
        suffix = f"{block[0]}x{block[1]}" if block is not None else granularity
        return f"humming_{self.scheme.weight.dtype.value}_{self.activation}_{suffix}"

    def allocate(self, layer: nn.Module, request: AllocationRequest) -> None:
        if len(request.weight_shape) != 2 or request.fused_dim != 0:
            raise ValueError(
                "HummingSpec requires a 2-D linear weight fused along dimension 0"
            )
        shape_n, shape_k = request.weight_shape
        if shape_n <= 0 or shape_k <= 0 or shape_n % 64:
            raise ValueError(
                "Humming requires positive local N/K and local N divisible by 64"
            )
        k_alignment = 32 if self.scheme.input is None else 64
        if shape_k % k_alignment or (self.group_size and shape_k % self.group_size):
            raise ValueError(
                f"Humming local K={shape_k} must be divisible by {k_alignment} "
                f"and group_size={self.group_size or 'channel'}"
            )
        if self.block_shape is not None and shape_n % self.block_shape[0]:
            raise ValueError("Humming local N must be divisible by block_shape[0]")
        if request.params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Humming requires FP16 or BF16 layer precision")
        if (
            self.scheme.weight.dtype is QDType.MXFP4
            and request.params_dtype is not torch.bfloat16
        ):
            raise ValueError(
                "Humming MXFP4 requires BF16 layer precision with humming-kernels 0.1.16"
            )
        if (
            self.scheme.input is not None
            and self.scheme.input.dtype is QDType.FP8_E5M2
            and request.params_dtype is not torch.bfloat16
        ):
            raise ValueError(
                "Humming FP8 E5M2 activations require BF16 layer precision"
            )
        if not request.logical_widths or any(
            width <= 0 for width in request.logical_widths
        ):
            raise ValueError("Humming logical_widths must contain positive widths")
        if sum(request.logical_widths) != shape_n:
            raise ValueError("Humming logical_widths must sum to local N")

        layer.weight = nn.Parameter(
            torch.empty(0, dtype=self.weight_dtype, device=request.device),
            requires_grad=False,
        )
        layer.weight.logical_shape = request.weight_shape
        # The loader stages dense weights without casting to packed storage dtype.
        layer.weight.loader_preserves_dtype = True
        layer.logical_shape = request.weight_shape
        layer.logical_widths = list(request.logical_widths)
        layer._humming_params_dtype = request.params_dtype
        layer._humming_pending_weight = None
        layer._humming_loaded_shards = set()
        layer._humming_prepared = False

    def load_weight(
        self,
        layer: nn.Module,
        loaded: torch.Tensor,
        shard_id: int | str | None,
        default_loader: Callable[[torch.Tensor, torch.Tensor, Any], None],
    ) -> None:
        if layer._humming_prepared:
            raise RuntimeError(
                "Humming packed weights cannot be updated; construct and reload a new layer"
            )
        if loaded.dtype not in _DENSE_DTYPES:
            raise ValueError(
                "Humming load-time PTQ requires unquantized FP16/BF16/FP32 checkpoint weights"
            )
        if loaded.ndim != 2 or loaded.shape[1] < layer.logical_shape[1]:
            raise ValueError(
                "Humming checkpoint weights must be 2-D and cover the local K dimension"
            )
        expected = self._expected_shards(layer)
        if shard_id not in expected:
            raise ValueError(
                f"Unexpected Humming weight shard {shard_id!r}; expected {expected!r}"
            )
        if shard_id in layer._humming_loaded_shards:
            raise RuntimeError(f"Humming weight shard {shard_id!r} was already loaded")
        keys = getattr(layer.weight, "hf_keys", None)
        local_rows = layer.logical_shape[0]
        if keys and len(keys) == len(layer.logical_widths):
            local_rows = next(
                width
                for (_, key_shard), width in zip(keys, layer.logical_widths)
                if key_shard == shard_id
            )
        if loaded.shape[0] < local_rows:
            raise ValueError(
                "Humming checkpoint weight does not cover the local output rows"
            )

        source = loaded.detach().to(device="cpu")
        pending = layer._humming_pending_weight
        if pending is None:
            pending = torch.empty(layer.logical_shape, dtype=source.dtype, device="cpu")
        elif pending.dtype != source.dtype:
            pending = pending.to(torch.promote_types(pending.dtype, source.dtype))
        default_loader(pending, source, shard_id)
        layer._humming_pending_weight = pending
        layer._humming_loaded_shards.add(shard_id)

    @staticmethod
    def _expected_shards(layer: nn.Module) -> set[int | str | None]:
        keys = getattr(layer.weight, "hf_keys", None)
        return {shard_id for _, shard_id in keys} if keys else {None}

    def process_after_loading(self, layer: nn.Module) -> None:
        if layer._humming_prepared:
            return
        missing = self._expected_shards(layer) - layer._humming_loaded_shards
        pending = layer._humming_pending_weight
        if missing or pending is None:
            raise RuntimeError(
                "Humming cannot quantize an incomplete weight; "
                f"missing checkpoint shards: {sorted(missing, key=repr)!r}"
            )

        config, tensors = self._prepare_weights(layer, pending)
        parameters = {
            name: nn.Parameter(tensor, requires_grad=False)
            for name, tensor in tensors.items()
        }
        parameters["weight"].__dict__.update(layer.weight.__dict__)
        for name, parameter in parameters.items():
            setattr(layer, name, parameter)
        layer.humming_config = config
        layer._humming_pending_weight = None
        layer._humming_prepared = True

    def _prepare_weights(
        self, layer: nn.Module, weight: torch.Tensor
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        device = layer.weight.device
        if device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError("Humming PTQ requires an NVIDIA CUDA device")
        try:
            from humming.schema import HummingInputSchema, HummingWeightSchema
            from humming.transform import (
                prepare_layer_config,
                transform_humming_tensors,
            )
            from humming.tune import get_heuristics_config
        except ImportError as exc:
            raise ImportError(
                "Humming is unavailable; install project dependencies with `uv sync`"
            ) from exc

        dtype = layer._humming_params_dtype
        with torch.cuda.device(device):
            major, minor = torch.cuda.get_device_capability(device)
            sm_version = major * 10 + minor
            minimum_sm = 80 if dtype is torch.bfloat16 else 75
            if self.scheme.input is not None and self.scheme.input.dtype in (
                QDType.FP8_E4M3,
                QDType.FP8_E5M2,
            ):
                minimum_sm = max(minimum_sm, 89)
            if sm_version < minimum_sm:
                raise RuntimeError(
                    f"Humming {self.spec_id} requires SM{minimum_sm} or newer"
                )

            from phyai.kernel.bootstrap import get_kernel_selector
            from phyai.kernel.types import KernelQuery

            get_kernel_selector().select(
                KernelQuery.build(
                    "gemm",
                    device=device,
                    role=getattr(layer, "kernel_role", ""),
                    dtype={
                        "input": dtype,
                        "output": dtype,
                        "weight": self.weight_dtype,
                    },
                    quant=self.physical_signature,
                    shape={"N": layer.logical_shape[0], "K": layer.logical_shape[1]},
                )
            )

            weight_quant = self.scheme.weight
            schema_kwargs: dict[str, Any] = {
                "b_dtype": _WEIGHT_DTYPES[weight_quant.dtype],
                "weight_scale_group_size": self.group_size,
                "has_zero_point": False,
            }
            if self.block_shape is not None and self.block_shape[0] > 1:
                schema_kwargs["weight_scale_group_size_n"] = self.block_shape[0]
            if weight_quant.granularity is Granularity.PER_TENSOR:
                schema_kwargs["weight_scale_type"] = "tensor"
            if weight_quant.dtype is QDType.MXFP4:
                schema_kwargs["bs_dtype"] = "float8e8m0"
            elif weight_quant.dtype is QDType.NVFP4:
                schema_kwargs.update(
                    bs_dtype="float8e4m3", weight_scale_2_type="tensor"
                )
            weight_schema = HummingWeightSchema(**schema_kwargs)
            input_schema = HummingInputSchema()
            if self.scheme.input is not None:
                input_schema = HummingInputSchema(
                    a_dtype=_WEIGHT_DTYPES[self.scheme.input.dtype],
                    input_quant_mode="dynamic_token",
                )
            config = prepare_layer_config(
                shape_n=layer.logical_shape[0],
                shape_k=layer.logical_shape[1],
                weight_schema=weight_schema,
                input_schema=input_schema,
                torch_dtype=dtype,
                device=device,
                has_bias=False,
            )
            tuning = get_heuristics_config(config, device=device)
            for _, _, candidate in tuning:
                _, tile_n, tile_k = candidate["block_shape"]
                if config.shape_n % tile_n or config.shape_k % tile_k:
                    raise ValueError(
                        f"Humming has no aligned default kernel for local weight {layer.logical_shape}; "
                        f"selected tile requires N%{tile_n}=0 and K%{tile_k}=0"
                    )

            source = weight.to(device=device).contiguous()
            if weight_quant.dtype is QDType.NVFP4:
                from phyai.layers.quant.nvfp4 import _quantize_nvfp4_linear

                packed, scales, global_scale = _quantize_nvfp4_linear(source, 16)
                tensors = {
                    "weight": packed.contiguous().view(torch.int32),
                    "weight_scale": scales,
                    "weight_scale_2": global_scale,
                }
            else:
                tensors = HummingWeightSchema.quant_tensor(source, weight_schema, dtype)
            return config, transform_humming_tensors(config, tensors)


__all__ = ["HummingSpec"]
