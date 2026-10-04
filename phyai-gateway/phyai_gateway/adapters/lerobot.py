import itertools
import json
import math
import operator
import pickle
import threading
from dataclasses import dataclass, field

import grpc
import numpy as np
import torch
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction
from lerobot.transport import services_pb2, services_pb2_grpc

from phyai_gateway.bindings import model_inference_pb2
from phyai_gateway.clients.model_inference import abort_for_backend_error

MAX_OBSERVATION_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class _PolicySetup:
    model_name: str
    actions_per_chunk: int
    state_names: tuple
    cameras: tuple
    extensions_json: str


@dataclass
class _SessionState:
    condition: threading.Condition = field(default_factory=threading.Condition)
    pending_observation: object = None
    setup: _PolicySetup | None = None


@dataclass(frozen=True)
class _DecodedObservation:
    timestamp: float
    timestep: int
    task: str
    state: np.ndarray
    images: dict


class LeRobotAdapter(services_pb2_grpc.AsyncInferenceServicer):
    """Route stock LeRobot clients using their checkpoint and robot features."""

    def __init__(self, model_client, *, fps=30):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("LeRobot fps must be positive and finite")
        self._model_client = model_client
        self._environment_dt = 1.0 / fps
        self._sessions = {}
        self._sessions_lock = threading.Lock()
        self._request_ids = itertools.count(1)

    def Ready(self, request, context):
        session_id = self._session_id(context)
        with self._sessions_lock:
            self._sessions[session_id] = _SessionState()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):
        session = self._lookup_session(context)
        try:
            config = pickle.loads(request.data)
            if not isinstance(config, RemotePolicyConfig):
                raise ValueError("PolicySetup is not a RemotePolicyConfig")
            model_name = config.pretrained_name_or_path
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("pretrained_name_or_path must name a registered model")
            actions_per_chunk = config.actions_per_chunk
            if isinstance(actions_per_chunk, bool):
                raise ValueError("actions_per_chunk must be an integer not boolean")
            try:
                actions_per_chunk = operator.index(actions_per_chunk)
            except TypeError as error:
                raise ValueError("actions_per_chunk must be an integer") from error
            if not 1 <= actions_per_chunk < 2**32:
                raise ValueError("actions_per_chunk must be a positive uint32")
            if not isinstance(config.lerobot_features, dict):
                raise ValueError("lerobot_features must be a dict")
            state_feature = config.lerobot_features.get("observation.state")
            if not isinstance(state_feature, dict):
                raise ValueError("observation.state dict feature is required")
            state_names = state_feature.get("names")
            if (
                not isinstance(state_names, (list, tuple))
                or not state_names
                or any(not isinstance(name, str) or not name for name in state_names)
                or len(set(state_names)) != len(state_names)
            ):
                raise ValueError(
                    "observation.state names must be unique nonempty strings"
                )
            state_names = tuple(state_names)
            if tuple(state_feature.get("shape", ())) != (len(state_names),):
                raise ValueError("observation.state shape must match its names")
            rename_map = config.rename_map
            if not isinstance(rename_map, dict) or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                or not value
                for key, value in rename_map.items()
            ):
                raise ValueError("rename_map must map nonempty feature names")
            cameras = []
            camera_names = set()
            for name, feature in config.lerobot_features.items():
                if not name.startswith("observation.images."):
                    continue
                raw_name = name.removeprefix("observation.images.")
                output_name = rename_map.get(name, name).removeprefix(
                    "observation.images."
                )
                if not raw_name or not output_name or output_name in camera_names:
                    raise ValueError("Camera names must be nonempty and unique")
                if not isinstance(feature, dict) or feature.get("dtype") not in (
                    "image",
                    "video",
                ):
                    raise ValueError(f"{name} must be an image or video feature")
                shape = tuple(feature.get("shape", ()))
                if len(shape) != 3 or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in shape
                ):
                    raise ValueError(f"{name} must declare a nonempty HWC shape")
                cameras.append((raw_name, output_name, shape))
                camera_names.add(output_name)
            setup = _PolicySetup(
                model_name=model_name,
                actions_per_chunk=actions_per_chunk,
                state_names=state_names,
                cameras=tuple(cameras),
                extensions_json=json.dumps(
                    {
                        "lerobot": {
                            "policy_type": config.policy_type,
                            "pretrained_name_or_path": model_name,
                            "device": config.device,
                            "lerobot_features": config.lerobot_features,
                            "rename_map": rename_map,
                        }
                    },
                    allow_nan=False,
                ),
            )
        except Exception as error:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Failed to decode LeRobot PolicySetup: {error}",
            )

        with session.condition:
            session.setup = setup
            session.pending_observation = None
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):
        payload = bytearray()
        started = False
        completed = False
        for chunk in request_iterator:
            if completed:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Received data after TRANSFER_END",
                )

            state = chunk.transfer_state
            if state == services_pb2.TRANSFER_BEGIN:
                if started:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "Unexpected TRANSFER_BEGIN",
                    )
                started = True
            elif state == services_pb2.TRANSFER_MIDDLE:
                if not started:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "TRANSFER_MIDDLE received before TRANSFER_BEGIN",
                    )
            elif state == services_pb2.TRANSFER_END:
                completed = True
            else:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Unknown observation transfer state",
                )

            if len(payload) + len(chunk.data) > MAX_OBSERVATION_BYTES:
                context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    "Observation exceeds 100 MiB",
                )
            payload.extend(chunk.data)

        if not completed:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Observation stream ended before TRANSFER_END",
            )
        if not payload:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Observation payload is empty",
            )

        session = self._lookup_session(context)
        with session.condition:
            setup = session.setup
        if setup is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "SendPolicyInstructions must precede observations",
            )
        try:
            observation = self._decode_observation(bytes(payload), setup)
        except Exception as error:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Failed to decode LeRobot observation: {error}",
            )

        with session.condition:
            if session.setup is not setup:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "PolicySetup changed while decoding the observation",
                )
            session.pending_observation = (observation, setup)
            session.condition.notify()
        return services_pb2.Empty()

    def GetActions(self, request, context):
        session = self._lookup_session(context)
        with session.condition:
            ready = session.condition.wait_for(
                lambda: session.pending_observation is not None,
                timeout=2.0,
            )
            if not ready:
                return services_pb2.Actions()
            observation, setup = session.pending_observation
            horizon = setup.actions_per_chunk
            session.pending_observation = None

        request_id = f"lerobot-gateway-{next(self._request_ids)}"
        inference_request = self._build_request(observation, request_id, setup)
        try:
            response = self._model_client.infer(inference_request, setup.model_name)
        except Exception as error:
            abort_for_backend_error(context, error)

        actions = self._decode_actions(response, request_id, horizon, context)
        if observation.timestep > 2**63 - len(actions):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "Action chunk timestep is invalid"
            )
        action_tensor = torch.from_numpy(actions.copy())
        action_chunk = [
            TimedAction(
                timestamp=observation.timestamp + index * self._environment_dt,
                timestep=observation.timestep + index,
                action=action,
            )
            for index, action in enumerate(action_tensor)
        ]
        return services_pb2.Actions(data=pickle.dumps(action_chunk))

    @staticmethod
    def _decode_observation(payload, setup):
        decoded = pickle.loads(payload)
        try:
            timestamp = float(decoded.timestamp)
            if not 0 <= timestamp < 2**63 / 1_000_000_000:
                raise ValueError(
                    "Observation timestamp must fit signed 64-bit nanoseconds"
                )
            if isinstance(decoded.timestep, bool):
                raise ValueError("timestep must be a nonnegative signed 64-bit integer")
            try:
                timestep = operator.index(decoded.timestep)
            except TypeError as error:
                raise ValueError("timestep must be a signed 64-bit integer") from error
            if not 0 <= timestep < 2**63:
                raise ValueError("timestep must be a nonnegative signed 64-bit integer")
            observation = decoded.observation
        except AttributeError as error:
            raise ValueError(
                "TimedObservation is missing required attributes"
            ) from error

        if not isinstance(observation, dict):
            raise ValueError("TimedObservation.observation is not a dict")
        task = observation.get("task", "")
        if not isinstance(task, str):
            raise ValueError("Observation task must be a string")

        state_value = observation.get("observation.state")
        state_names = setup.state_names
        raw_state = None
        if state_value is None or any(name in observation for name in state_names):
            try:
                raw_state = LeRobotAdapter._state_array(
                    np.asarray([observation[name] for name in state_names]),
                    len(state_names),
                )
            except KeyError as error:
                raise ValueError(
                    f"Observation state field is missing: {error.args[0]}"
                ) from error
        state_value = (
            raw_state
            if state_value is None
            else LeRobotAdapter._state_array(state_value, len(state_names))
        )
        if raw_state is not None and not np.array_equal(state_value, raw_state):
            raise ValueError(
                "Packed observation.state differs from its named state fields"
            )

        images = {}
        for key, output_name, shape in setup.cameras:
            value = observation.get(key)
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Image '{key}' must be a NumPy array")
            if value.dtype != np.uint8:
                raise ValueError(f"Image '{key}' dtype must be uint8")
            if value.shape != shape:
                raise ValueError(f"Image '{key}' shape must match its declared {shape}")
            data = value.tobytes(order="C")
            images[output_name] = (shape, data)
        return _DecodedObservation(
            timestamp=timestamp,
            timestep=timestep,
            task=task,
            state=state_value,
            images=images,
        )

    @staticmethod
    def _state_array(value, state_dim):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("Observation state tensor must be on CPU")
            value = value.detach()
            if value.dtype == torch.bfloat16:
                value = value.float()
            value = value.numpy()
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "iuf":
            raise ValueError("Observation state must be a numeric array or CPU tensor")
        if value.shape == (1, state_dim):
            value = value[0]
        if value.shape != (state_dim,):
            raise ValueError(
                f"Observation state shape must be [{state_dim}] or [1,{state_dim}]"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            value = value.astype("<f4", copy=False)
        if not np.isfinite(value).all():
            raise ValueError("Observation state must contain finite float32 values")
        return value

    @staticmethod
    def _build_request(observation, request_id, setup):
        timestamp_ns = int(observation.timestamp * 1_000_000_000)
        images = []
        for image_name, (shape, image_data) in observation.images.items():
            images.append(
                model_inference_pb2.Image(
                    name=image_name,
                    data=image_data,
                    shape=shape,
                    dtype=model_inference_pb2.DATA_TYPE_UINT8,
                    encoding=model_inference_pb2.IMAGE_ENCODING_RAW,
                    layout=model_inference_pb2.IMAGE_LAYOUT_HWC,
                )
            )

        return model_inference_pb2.InferenceRequest(
            model_name=setup.model_name,
            request_id=request_id,
            timestamp_ns=timestamp_ns,
            images=images,
            robot_state=model_inference_pb2.Tensor(
                data=observation.state.tobytes(),
                shape=observation.state.shape,
                dtype=model_inference_pb2.DATA_TYPE_FLOAT32,
            ),
            instruction=observation.task,
            requested_action_horizon=setup.actions_per_chunk,
            extensions_json=setup.extensions_json,
        )

    @staticmethod
    def _decode_actions(response, request_id, horizon, context):
        if response.request_id != request_id:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server returned a mismatched request_id",
            )
        actions = response.actions
        dtypes = {
            model_inference_pb2.DATA_TYPE_FLOAT16: "<f2",
            model_inference_pb2.DATA_TYPE_BFLOAT16: "<u2",
            model_inference_pb2.DATA_TYPE_FLOAT32: "<f4",
            model_inference_pb2.DATA_TYPE_FLOAT64: "<f8",
        }
        dtype = dtypes.get(actions.dtype)
        if dtype is None:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server actions must use a floating point dtype",
            )
        if len(actions.shape) != 2 or not 1 <= actions.shape[0] <= horizon:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server actions must have shape [1..requested_horizon, action_dim]",
            )
        action_count = actions.shape[0]
        action_dim = actions.shape[1]
        if action_dim <= 0:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server action_dim must be positive",
            )
        if len(actions.data) != action_count * action_dim * np.dtype(dtype).itemsize:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server action shape does not match its data length",
            )
        values = np.frombuffer(actions.data, dtype=dtype).reshape(
            action_count, action_dim
        )
        if actions.dtype == model_inference_pb2.DATA_TYPE_BFLOAT16:
            values = (values.astype("<u4") << 16).view("<f4")
        if not np.isfinite(values).all():
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server actions contain non-finite values",
            )
        return values

    @staticmethod
    def _session_id(context):
        for item in context.invocation_metadata():
            if item.key == "x-session-id" and item.value:
                return item.value
        return context.peer()

    def _lookup_session(self, context):
        session_id = self._session_id(context)
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"{session_id} session not found",
            )
        return session
