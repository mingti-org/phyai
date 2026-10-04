import json
import math
import time
import uuid
from collections.abc import Mapping

import msgpack
import numpy as np

from phyai_gateway.bindings import model_inference_pb2


NUMPY_DTYPES = {
    model_inference_pb2.DATA_TYPE_UINT8: np.dtype("u1"),
    model_inference_pb2.DATA_TYPE_INT32: np.dtype("<i4"),
    model_inference_pb2.DATA_TYPE_INT64: np.dtype("<i8"),
    model_inference_pb2.DATA_TYPE_FLOAT16: np.dtype("<f2"),
    model_inference_pb2.DATA_TYPE_FLOAT32: np.dtype("<f4"),
    model_inference_pb2.DATA_TYPE_FLOAT64: np.dtype("<f8"),
    model_inference_pb2.DATA_TYPE_BOOL: np.dtype("?"),
}


class RLinfWireError(ValueError):
    pass


class RLinfPayloadError(ValueError):
    pass


class RLinfBackendResponseError(RuntimeError):
    pass


def _get_field(value, name):
    return value.get(name, value.get(name.encode()))


def unpack_numpy(value):
    if not isinstance(value, dict):
        return value
    is_array = _get_field(value, "__ndarray__")
    is_scalar = _get_field(value, "__npgeneric__")
    if not is_array and not is_scalar:
        return value
    try:
        dtype = np.dtype(_get_field(value, "dtype"))
    except (TypeError, ValueError) as error:
        raise RLinfWireError("invalid NumPy dtype") from error
    if dtype.kind in {"O", "V", "c"}:
        raise RLinfWireError(f"unsupported NumPy dtype: {dtype}")
    data = _get_field(value, "data")
    if is_scalar:
        try:
            return dtype.type(data)
        except (TypeError, ValueError, OverflowError) as error:
            raise RLinfWireError("invalid NumPy scalar") from error
    shape = _get_field(value, "shape")
    if not isinstance(data, bytes) or not isinstance(shape, (list, tuple)):
        raise RLinfWireError("invalid NumPy array payload")
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
        for dimension in shape
    ):
        raise RLinfWireError("invalid NumPy array shape")
    if len(data) != math.prod(shape) * dtype.itemsize:
        raise RLinfWireError("NumPy array shape does not match its data length")
    try:
        return np.frombuffer(data, dtype=dtype).reshape(tuple(shape))
    except (TypeError, ValueError, OverflowError) as error:
        raise RLinfWireError("invalid NumPy array payload") from error


def pack_numpy(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"}:
            raise TypeError(f"unsupported NumPy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(order="C"),
            b"dtype": array.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot encode {type(value).__name__}")


class RLinfAdapter:
    def __init__(self, model_client, default_model: str | None = None):
        self._model_client = model_client
        self._default_model = default_model

    def infer_msgpack(self, body: bytes) -> bytes:
        try:
            payload = msgpack.unpackb(body, raw=False, object_hook=unpack_numpy)
        except RLinfWireError:
            raise
        except (msgpack.UnpackException, ValueError, TypeError) as error:
            raise RLinfWireError("invalid MessagePack request") from error

        if isinstance(payload, Mapping) and "input" in payload:
            return self._infer_envelope(payload)
        data = self._validate_payload(payload, self._default_model)
        request = self._build_request(data, time.time_ns())
        response = self._model_client.infer(request, data["model_name"])
        batch = self._decode_actions(
            response,
            request.request_id,
            data["batch_size"],
            data["horizon"],
        )
        return msgpack.packb({"actions": batch}, default=pack_numpy, use_bin_type=True)

    @staticmethod
    def _model_name(payload, default_model=None):
        model_name = payload.get("model_name")
        model_alias = payload.get("model")
        if model_name is None:
            model_name = model_alias
        elif model_alias is not None and model_alias != model_name:
            raise RLinfPayloadError("model and model_name must match")
        if model_name is None:
            model_name = default_model
        if not isinstance(model_name, str) or not model_name.strip():
            raise RLinfPayloadError(
                "model_name is required unless the gateway has a default model"
            )
        return model_name.strip()

    @staticmethod
    def _validate_payload(payload, default_model=None):
        if not isinstance(payload, Mapping):
            raise RLinfPayloadError("request payload must be an object")
        model_name = RLinfAdapter._model_name(payload, default_model)
        observation = payload.get("observation")
        metadata = payload.get("metadata", {})
        if not isinstance(observation, Mapping):
            raise RLinfPayloadError("observation must be an object")
        if not isinstance(metadata, Mapping):
            raise RLinfPayloadError("metadata must be an object")
        batch_size = metadata.get("batch_size")
        if batch_size is None:
            for key in ("states", "main_images", "wrist_images", "extra_view_images"):
                value = observation.get(key)
                if isinstance(value, np.ndarray) and value.ndim:
                    batch_size = value.shape[0]
                    break
            if batch_size is None:
                tasks = observation.get("task_descriptions")
                if isinstance(tasks, (list, tuple)):
                    batch_size = len(tasks)
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise RLinfPayloadError("metadata.batch_size must be a positive integer")

        images = []
        for name in ("main_images", "wrist_images", "extra_view_images"):
            value = observation.get(name)
            if value is not None:
                images.extend(RLinfAdapter._validate_images(value, name, batch_size))
        states = observation.get("states")
        if states is not None:
            if not isinstance(states, np.ndarray) or states.ndim < 2:
                raise RLinfPayloadError(
                    "observation.states must be a batched NumPy array"
                )
            if states.shape[0] != batch_size or min(states.shape[1:]) <= 0:
                raise RLinfPayloadError(
                    "observation.states shape does not match the batch"
                )
            if states.dtype.kind != "f" or states.dtype.itemsize not in (2, 4, 8):
                raise RLinfPayloadError(
                    "observation.states must use a floating-point dtype"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                states = np.ascontiguousarray(states, dtype="<f4")
            if not np.isfinite(states).all():
                raise RLinfPayloadError(
                    "observation.states must contain finite float32 values"
                )

        tasks = observation.get("task_descriptions")
        if tasks is not None:
            if not isinstance(tasks, (list, tuple)) or len(tasks) != batch_size:
                raise RLinfPayloadError(
                    "observation.task_descriptions must match the batch size"
                )
            if not all(isinstance(task, str) for task in tasks):
                raise RLinfPayloadError(
                    "observation.task_descriptions must contain strings"
                )

        horizon = payload.get("requested_action_horizon", 0)
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or not 0 <= horizon <= 2**32 - 1
        ):
            raise RLinfPayloadError("requested_action_horizon must be a uint32")
        extensions = {"source": "rlinf", **metadata, "batch_size": batch_size}
        if tasks is not None:
            extensions["instructions"] = list(tasks)
        try:
            extensions_json = json.dumps(
                extensions, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as error:
            raise RLinfPayloadError("metadata must contain JSON values") from error

        return {
            "model_name": model_name.strip(),
            "batch_size": batch_size,
            "images": images,
            "states": states,
            "tasks": tasks,
            "horizon": horizon,
            "extensions_json": extensions_json,
        }

    @staticmethod
    def _validate_images(images, name, batch_size):
        if not isinstance(images, np.ndarray) or images.ndim not in (4, 5):
            raise RLinfPayloadError(
                f"observation.{name} must be a rank-4 or rank-5 NumPy array"
            )
        if images.dtype != np.uint8:
            raise RLinfPayloadError(f"observation.{name} must use uint8")
        if images.shape[0] != batch_size or min(images.shape[1:]) <= 0:
            raise RLinfPayloadError(
                f"observation.{name} shape does not match the batch"
            )
        if images.shape[-1] == 3:
            layout = model_inference_pb2.IMAGE_LAYOUT_HWC
        elif images.shape[-3] == 3:
            layout = model_inference_pb2.IMAGE_LAYOUT_CHW
        else:
            raise RLinfPayloadError(
                f"observation.{name} must contain HWC or CHW RGB images"
            )
        if images.ndim == 4:
            return [(name, images, layout)]
        return [
            (f"{name}.{index}", images[:, index], layout)
            for index in range(images.shape[1])
        ]

    @staticmethod
    def _build_request(data, received_ns):
        request_id = f"rlinf-{uuid.uuid4().hex}"
        images = []
        for image_name, batch, layout in data["images"]:
            image = np.ascontiguousarray(batch)
            images.append(
                model_inference_pb2.Image(
                    name=image_name,
                    data=image.tobytes(order="C"),
                    shape=list(image.shape),
                    dtype=model_inference_pb2.DATA_TYPE_UINT8,
                    encoding=model_inference_pb2.IMAGE_ENCODING_RAW,
                    layout=layout,
                )
            )

        request = model_inference_pb2.InferenceRequest(
            request_id=request_id,
            timestamp_ns=received_ns,
            images=images,
            instruction=data["tasks"][0] if data["tasks"] else "",
            requested_action_horizon=data["horizon"],
            extensions_json=data["extensions_json"],
        )
        state = data["states"]
        if state is not None:
            request.robot_state.CopyFrom(
                model_inference_pb2.Tensor(
                    data=state.tobytes(order="C"),
                    shape=list(state.shape),
                    dtype=model_inference_pb2.DATA_TYPE_FLOAT32,
                )
            )
        return request

    def _infer_envelope(self, payload):
        model_name = self._model_name(payload, self._default_model)
        if not isinstance(payload["input"], Mapping):
            raise RLinfPayloadError("input must be an object")
        for field in ("parameters", "runtime"):
            if field in payload and not isinstance(payload[field], Mapping):
                raise RLinfPayloadError(f"{field} must be an object")

        request = model_inference_pb2.InferenceRequest(
            request_id=f"rlinf-{uuid.uuid4().hex}",
            timestamp_ns=time.time_ns(),
        )
        input_metadata = dict(payload["input"])
        observation = input_metadata.get("observation")
        fields = [input_metadata]
        if isinstance(observation, Mapping):
            input_metadata["observation"] = dict(observation)
            fields.append(input_metadata["observation"])
        for values in fields:
            for name, value in list(values.items()):
                if not isinstance(value, np.ndarray):
                    continue
                if not isinstance(name, str) or not name:
                    raise RLinfPayloadError(
                        "tensor input names must be non-empty strings"
                    )
                if name in request.inputs:
                    raise RLinfPayloadError(f"duplicate tensor input name: {name}")
                dtype = value.dtype.newbyteorder("<")
                proto_dtype = next(
                    (key for key, item in NUMPY_DTYPES.items() if item == dtype), None
                )
                if proto_dtype is None:
                    raise RLinfPayloadError(
                        f"unsupported tensor dtype for {name}: {value.dtype}"
                    )
                array = np.ascontiguousarray(value, dtype=dtype)
                if dtype.kind == "f" and not np.isfinite(array).all():
                    raise RLinfPayloadError(
                        f"tensor input {name} contains non-finite values"
                    )
                request.inputs[name].CopyFrom(
                    model_inference_pb2.Tensor(
                        data=array.tobytes(order="C"),
                        shape=value.shape,
                        dtype=proto_dtype,
                    )
                )
                del values[name]

        extensions = {"source": "rlinf", **payload, "input": input_metadata}
        try:
            request.extensions_json = json.dumps(
                extensions, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as error:
            raise RLinfPayloadError(
                "envelope metadata must contain JSON values"
            ) from error
        response = self._model_client.infer(request, model_name)
        actions = self._decode_actions(response, request.request_id)
        envelope = {
            "data": [
                {
                    "action": {
                        "values": actions,
                        "shape": list(actions.shape),
                        "dtype": actions.dtype.name,
                    }
                }
            ]
        }
        return msgpack.packb(envelope, default=pack_numpy, use_bin_type=True)

    @staticmethod
    def _decode_actions(response, request_id, batch_size=None, horizon=0):
        if response.request_id != request_id:
            raise RLinfBackendResponseError(
                "Model Server response request_id does not match request"
            )
        tensor = response.actions
        dtype = NUMPY_DTYPES.get(tensor.dtype)
        if dtype is None or dtype.kind != "f":
            raise RLinfBackendResponseError(
                "Model Server actions must use a floating-point dtype"
            )
        if not tensor.shape or any(dimension <= 0 for dimension in tensor.shape):
            raise RLinfBackendResponseError(
                "Model Server actions must have a non-empty shape"
            )
        if batch_size is not None and (
            len(tensor.shape) != 3
            or tensor.shape[0] != batch_size
            or tensor.shape[1] <= 0
            or (horizon and tensor.shape[1] != horizon)
            or tensor.shape[2] <= 0
        ):
            raise RLinfBackendResponseError(
                "Model Server actions must have shape [batch, horizon, action_dim] matching the request"
            )
        if len(tensor.data) != math.prod(tensor.shape) * dtype.itemsize:
            raise RLinfBackendResponseError(
                "Model Server action shape does not match its data length"
            )
        try:
            actions = np.frombuffer(tensor.data, dtype=dtype).reshape(
                tuple(tensor.shape)
            )
        except (ValueError, OverflowError) as error:
            raise RLinfBackendResponseError(
                "Model Server actions have an invalid shape"
            ) from error
        if not np.isfinite(actions).all():
            raise RLinfBackendResponseError(
                "Model Server actions contain non-finite values"
            )
        return actions
