import json
import math
import time
import uuid
from collections.abc import Mapping

import msgpack
import numpy as np

from phyai_gateway.bindings import model_inference_pb2


class RLinfWireError(ValueError):
    pass


class RLinfPayloadError(ValueError):
    pass


class RLinfBackendResponseError(RuntimeError):
    pass


def _get_field(value, name):
    return value.get(name, value.get(name.encode()))


def _unpack_numpy(value):
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
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 0
        for dimension in shape
    ):
        raise RLinfWireError("invalid NumPy array shape")
    if len(data) != math.prod(shape) * dtype.itemsize:
        raise RLinfWireError("NumPy array shape does not match its data length")
    return np.frombuffer(data, dtype=dtype).reshape(tuple(shape))


def _pack_numpy(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"}:
            raise TypeError(f"unsupported NumPy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(order="C"),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot encode {type(value).__name__}")


class RLinfAdapter:
    def __init__(self, model_client):
        self._model_client = model_client

    def infer_msgpack(self, body: bytes) -> bytes:
        try:
            payload = msgpack.unpackb(body, raw=False, object_hook=_unpack_numpy)
        except RLinfWireError:
            raise
        except (msgpack.UnpackException, ValueError, TypeError) as error:
            raise RLinfWireError("invalid MessagePack request") from error

        data = self._validate_payload(payload)
        request = self._build_request(data, time.time_ns())
        response, _ = self._model_client.infer(request, data["model_name"])
        batch = self._decode_actions(
            response,
            request.request_id,
            data["batch_size"],
            data["horizon"],
        )
        return msgpack.packb(
            {"actions": batch}, default=_pack_numpy, use_bin_type=True
        )

    @staticmethod
    def _validate_payload(payload):
        if not isinstance(payload, Mapping):
            raise RLinfPayloadError("request payload must be an object")
        model_name = payload.get("model_name")
        model_alias = payload.get("model")
        if model_name is None:
            model_name = model_alias
        elif model_alias is not None and model_alias != model_name:
            raise RLinfPayloadError("model and model_name must match")
        if not isinstance(model_name, str) or not model_name.strip():
            raise RLinfPayloadError("model_name is required")

        observation = payload.get("observation")
        metadata = payload.get("metadata")
        if not isinstance(observation, Mapping):
            raise RLinfPayloadError("observation must be an object")
        if not isinstance(metadata, Mapping):
            raise RLinfPayloadError("metadata must be an object")
        batch_size = metadata.get("batch_size")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise RLinfPayloadError("metadata.batch_size must be a positive integer")

        main_images = RLinfAdapter._validate_images(
            observation.get("main_images"), "observation.main_images", batch_size
        )
        wrist_images = RLinfAdapter._validate_images(
            observation.get("wrist_images"), "observation.wrist_images", batch_size
        )
        states = observation.get("states")
        if not isinstance(states, np.ndarray) or states.ndim != 2:
            raise RLinfPayloadError("observation.states must be a rank-2 NumPy array")
        if states.shape[0] != batch_size or states.shape[1] <= 0:
            raise RLinfPayloadError("observation.states shape does not match the batch")
        if states.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise RLinfPayloadError("observation.states must use float32 or float64")
        if not np.isfinite(states).all():
            raise RLinfPayloadError("observation.states contains non-finite values")

        tasks = observation.get("task_descriptions")
        if not isinstance(tasks, (list, tuple)) or len(tasks) != batch_size:
            raise RLinfPayloadError(
                "observation.task_descriptions must match the batch size"
            )
        if not all(isinstance(task, str) and task.strip() for task in tasks):
            raise RLinfPayloadError(
                "observation.task_descriptions must contain non-empty strings"
            )

        horizon = payload.get("requested_action_horizon")
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or not 1 <= horizon <= 2**32 - 1
        ):
            raise RLinfPayloadError("requested_action_horizon must be a positive uint32")
        mode = metadata.get("mode")
        stage_id = metadata.get("stage_id")
        reset = metadata.get("reset")
        if mode != "eval":
            raise RLinfPayloadError("metadata.mode must be eval")
        if isinstance(stage_id, bool) or not isinstance(stage_id, int):
            raise RLinfPayloadError("metadata.stage_id must be an integer")
        if not isinstance(reset, bool):
            raise RLinfPayloadError("metadata.reset must be a boolean")

        return {
            "model_name": model_name.strip(),
            "batch_size": batch_size,
            "main_images": main_images,
            "wrist_images": wrist_images,
            "states": states,
            "tasks": tasks,
            "horizon": horizon,
            "metadata": {
                "source": "rlinf",
                "mode": mode,
                "batch_size": batch_size,
                "stage_id": stage_id,
                "reset": reset,
            },
        }

    @staticmethod
    def _validate_images(images, name, batch_size):
        if not isinstance(images, np.ndarray) or images.ndim != 4:
            raise RLinfPayloadError(f"{name} must be a rank-4 NumPy array")
        if images.dtype != np.uint8:
            raise RLinfPayloadError(f"{name} must use uint8")
        if (
            images.shape[0] != batch_size
            or min(images.shape[1:3]) <= 0
            or images.shape[3] != 3
        ):
            raise RLinfPayloadError(
                f"{name} must have shape [batch, height, width, 3]"
            )
        return images

    @staticmethod
    def _build_request(data, received_ns):
        request_id = f"rlinf-{uuid.uuid4().hex}"
        images = []
        for image_name, batch in (
            ("agentview", data["main_images"]),
            ("robot0_eye_in_hand", data["wrist_images"]),
        ):
            image = np.ascontiguousarray(batch)
            images.append(
                model_inference_pb2.Image(
                    name=image_name,
                    data=image.tobytes(order="C"),
                    shape=list(image.shape),
                    dtype=model_inference_pb2.DATA_TYPE_UINT8,
                    encoding=model_inference_pb2.IMAGE_ENCODING_RAW,
                    layout=model_inference_pb2.IMAGE_LAYOUT_HWC,
                )
            )

        state = np.ascontiguousarray(data["states"], dtype="<f4")
        extensions = {**data["metadata"], "instructions": list(data["tasks"])}
        return model_inference_pb2.InferenceRequest(
            request_id=request_id,
            timestamp_ns=received_ns,
            images=images,
            robot_state=model_inference_pb2.Tensor(
                data=state.tobytes(order="C"),
                shape=list(state.shape),
                dtype=model_inference_pb2.DATA_TYPE_FLOAT32,
            ),
            instruction=data["tasks"][0],
            requested_action_horizon=data["horizon"],
            extensions_json=json.dumps(extensions, separators=(",", ":")),
        )

    @staticmethod
    def _decode_actions(response, request_id, batch_size, horizon):
        if response.request_id != request_id:
            raise RLinfBackendResponseError(
                "Model Server response request_id does not match request"
            )
        tensor = response.actions
        if tensor.dtype != model_inference_pb2.DATA_TYPE_FLOAT32:
            raise RLinfBackendResponseError("Model Server actions must use FLOAT32")
        if (
            len(tensor.shape) != 3
            or tensor.shape[0] != batch_size
            or tensor.shape[1] != horizon
            or tensor.shape[2] <= 0
        ):
            raise RLinfBackendResponseError(
                "Model Server actions must have shape [batch, requested_horizon, action_dim]"
            )
        if len(tensor.data) != math.prod(tensor.shape) * 4:
            raise RLinfBackendResponseError(
                "Model Server action shape does not match its data length"
            )
        actions = np.frombuffer(tensor.data, dtype="<f4").reshape(tuple(tensor.shape))
        if not np.isfinite(actions).all():
            raise RLinfBackendResponseError(
                "Model Server actions contain non-finite values"
            )
        return actions
