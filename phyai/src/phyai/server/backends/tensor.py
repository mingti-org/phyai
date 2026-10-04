"""Bridge named protocol tensors to a model's Engine request type."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import time

import grpc
import numpy as np
import torch

from phyai.server.lifecycle import EngineUnavailableError
from phyai.server.serving import load_callable
from phyai_gateway.bindings import model_inference_pb2 as pb

TENSOR_DTYPES = {
    pb.DATA_TYPE_UINT8: (torch.uint8, np.dtype("u1")),
    pb.DATA_TYPE_INT32: (torch.int32, np.dtype("<i4")),
    pb.DATA_TYPE_INT64: (torch.int64, np.dtype("<i8")),
    pb.DATA_TYPE_FLOAT16: (torch.float16, np.dtype("<f2")),
    pb.DATA_TYPE_BFLOAT16: (torch.bfloat16, np.dtype("<u2")),
    pb.DATA_TYPE_FLOAT32: (torch.float32, np.dtype("<f4")),
    pb.DATA_TYPE_FLOAT64: (torch.float64, np.dtype("<f8")),
    pb.DATA_TYPE_BOOL: (torch.bool, np.dtype("?")),
}
WIRE_DTYPES = {
    dtype: (wire, numpy_dtype) for wire, (dtype, numpy_dtype) in TENSOR_DTYPES.items()
}


def decode_tensor(tensor: pb.Tensor) -> torch.Tensor:
    try:
        dtype, numpy_dtype = TENSOR_DTYPES[tensor.dtype]
    except KeyError as error:
        raise ValueError(f"unsupported input tensor dtype: {tensor.dtype}") from error
    shape = tuple(tensor.shape)
    if len(tensor.data) != math.prod(shape) * numpy_dtype.itemsize:
        raise ValueError("input tensor data length does not match shape and dtype")
    array = (
        np.frombuffer(tensor.data, dtype=numpy_dtype)
        .astype(numpy_dtype.newbyteorder("="), copy=True)
        .reshape(shape)
    )
    value = torch.from_numpy(array)
    return value.view(torch.bfloat16) if dtype == torch.bfloat16 else value


def encode_tensor(tensor: torch.Tensor) -> pb.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("model outputs must be torch tensors")
    try:
        dtype, numpy_dtype = WIRE_DTYPES[tensor.dtype]
    except KeyError as error:
        raise TypeError(f"unsupported output tensor dtype: {tensor.dtype}") from error
    value = tensor.detach().cpu().resolve_neg().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    array = value.numpy().astype(numpy_dtype, copy=False)
    return pb.Tensor(data=array.tobytes(), shape=list(tensor.shape), dtype=dtype)


class TensorBackend:
    def __init__(
        self,
        *,
        engine_args,
        deployment=None,
        request_type: str | None = None,
        input_device: str = "cpu",
        output_name: str = "output",
    ):
        if not isinstance(output_name, str) or not output_name.strip():
            raise ValueError("output_name must be a nonempty string")
        self.request_type = load_callable(request_type) if request_type else None
        if self.request_type is not None and not isinstance(self.request_type, type):
            raise TypeError("request_type must name a Python class")
        self.input_device = torch.device(input_device)
        self.output_name = output_name
        self._healthy = False
        self.engine = None

        from phyai.engine import Engine

        self.engine = Engine(engine_args, deployment=deployment)
        try:
            self.engine.setup()
        except BaseException:
            self.close()
            raise
        self._healthy = True

    @property
    def healthy(self) -> bool:
        return self._healthy and self.engine is not None

    def infer(self, request, context):
        try:
            extensions = json.loads(request.extensions_json or "{}")
            if not isinstance(extensions, dict):
                raise ValueError("extensions_json must be a JSON object")
            parameters = extensions.get("parameters", {})
            if not isinstance(parameters, dict):
                raise ValueError("extensions_json.parameters must be an object")
            if parameters.keys() & request.inputs.keys():
                raise ValueError("input names must not overlap parameters")
            values = dict(parameters)
            for name, tensor in request.inputs.items():
                values[name] = decode_tensor(tensor).to(self.input_device)
            payload = self.request_type(**values) if self.request_type else values
        except (ValueError, TypeError, OverflowError) as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

        start = time.perf_counter_ns()
        try:
            result = self.engine.step(payload)
        except EngineUnavailableError:
            self._healthy = False
            context.abort(grpc.StatusCode.UNAVAILABLE, "model engine is unavailable")
        elapsed_us = (time.perf_counter_ns() - start) // 1000
        if isinstance(result, torch.Tensor):
            result = {self.output_name: result}
        if not isinstance(result, Mapping) or any(
            not isinstance(name, str) or not name for name in result
        ):
            raise TypeError("model must return a tensor or a mapping of named tensors")
        return pb.InferenceResponse(
            request_id=request.request_id,
            outputs={name: encode_tensor(tensor) for name, tensor in result.items()},
            inference_time_us=elapsed_us,
        )

    def close(self):
        self._healthy = False
        engine, self.engine = self.engine, None
        if engine is not None:
            engine.close()
