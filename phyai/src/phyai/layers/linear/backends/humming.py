"""Execute Humming GEMMs over weights prepared by HummingSpec."""

from __future__ import annotations

from functools import partial
from typing import Mapping

import torch


def _workspace(layer: torch.nn.Module, device: torch.device) -> torch.Tensor:
    if torch.cuda.is_current_stream_capturing():
        # Each graph owns and initializes its locks, even when several graphs
        # are captured on the same stream and replayed in a different order.
        return torch.zeros(1024, dtype=torch.int32, device=device)
    # Stream-K reductions mutate locks. Calls on different streams must own
    # separate storage, including capture streams and green-context streams.
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    workspaces = getattr(layer, "_humming_workspaces", None)
    if workspaces is None:
        workspaces = {}
        layer._humming_workspaces = workspaces
    locks = workspaces.get(key)
    if locks is None:
        locks = torch.zeros(1024, dtype=torch.int32, device=device)
        workspaces[key] = locks
    return locks


def gemm_humming(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    compute_config: dict | None = None,
    tuning_config: dict | list | None = None,
) -> torch.Tensor:
    if not getattr(layer, "_humming_prepared", False):
        raise RuntimeError("Humming weights must finish post_load() before execution")
    if x.device != layer.weight.device:
        raise ValueError("Humming input and prepared weight must use the same device")
    if x.dtype != layer._humming_params_dtype:
        raise ValueError("Humming input dtype must match the prepared compute dtype")
    n, k = layer.logical_shape
    if x.ndim < 1 or x.shape[-1] != k:
        raise ValueError(f"Humming input must have last dimension {k}")
    output_shape = (*x.shape[:-1], n)
    if x.numel() == 0:
        return x.new_empty(output_shape)

    from humming.forward import humming_forward

    inputs = x.reshape(-1, k).contiguous()
    locks = _workspace(layer, x.device)
    result = humming_forward(
        layer.humming_config,
        inputs=inputs,
        weight=layer.weight,
        weight_scale=getattr(layer, "weight_scale", None),
        weight_scale_2=getattr(layer, "weight_scale_2", None),
        zero_point=getattr(layer, "zero_point", None),
        locks=locks,
        compute_config=compute_config,
        tuning_config=tuning_config,
    )
    if bias is not None:
        result = result + bias
    return result.reshape(output_shape)


def prepare(facts, params):
    """Bind execution-only tuning, leaving the prepared weight layout fixed."""
    unknown = set(params) - {"compute_config", "tuning_config"}
    if unknown:
        raise ValueError(f"unsupported Humming parameters: {sorted(unknown)}")
    compute = params.get("compute_config")
    if compute is not None:
        if not isinstance(compute, Mapping):
            raise TypeError("Humming compute_config must be a mapping")
        allowed = {
            "gemm_type",
            "use_f16_accum",
            "use_batch_invariant",
            "use_m_major_input_scale",
        }
        if set(compute) - allowed:
            raise ValueError("Humming compute_config contains unsupported fields")
        if compute.get("gemm_type", "dense") != "dense":
            raise ValueError("Humming Linear supports only dense GEMM")
        for name, value in compute.items():
            if name != "gemm_type" and not isinstance(value, bool):
                raise TypeError(f"Humming {name} must be a boolean")
        compute = dict(compute)
    tuning = params.get("tuning_config")
    if tuning is not None and not isinstance(tuning, (dict, list)):
        raise TypeError("Humming tuning_config must be a mapping or list")
    return partial(gemm_humming, compute_config=compute, tuning_config=tuning)


__all__ = ["gemm_humming"]
