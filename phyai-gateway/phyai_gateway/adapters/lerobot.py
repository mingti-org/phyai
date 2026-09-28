import itertools
import math
import operator
import pickle
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import grpc
import numpy as np
import torch
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction
from lerobot.transport import services_pb2, services_pb2_grpc

from phyai_gateway.bindings import model_inference_pb2
from phyai_gateway.clients.model_inference import abort_for_backend_error

MAX_OBSERVATION_BYTES = 100 * 1024 * 1024
MAX_ACTIONS_PER_CHUNK = 50
MODEL_NAME = "pi05"
LATENCY_HEADER = (
    "pickle_encode_ms,model_request_infer_ms,"
    "inference_time_ms,total_gateway_ms\n"
)


@dataclass
class _SessionState:
    condition: threading.Condition = field(default_factory=threading.Condition)
    pending_observation: object = None
    actions_per_chunk: int = 1
    state_names: tuple = ()


@dataclass(frozen=True)
class _DecodedObservation:
    timestamp: float
    timestep: int
    must_go: bool
    task: str
    state: tuple
    images: dict


class LeRobotAdapter(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(self, model_client):
        self._model_client = model_client
        self._sessions = {}
        self._sessions_lock = threading.Lock()
        self._request_ids = itertools.count(1)
        self._latency_lock = threading.Lock()
        self._latency_path = (
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "latencies"
            / "Latencies.csv"
        )
        self._latency_enabled = True
        try:
            self._latency_path.parent.mkdir(parents=True, exist_ok=True)
            with self._latency_path.open(
                "w", encoding="utf-8", newline=""
            ) as file:
                file.write(LATENCY_HEADER)
        except OSError:
            self._latency_enabled = False

    def Ready(self, request, context):
        print(type(context))
        print(context)
        session_id = self._session_id(context)
        with self._sessions_lock:
            self._sessions[session_id] = _SessionState()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):
        decode_error = None
        try:
            config = pickle.loads(request.data)
            if not isinstance(config, RemotePolicyConfig):
                raise ValueError("PolicySetup is not a RemotePolicyConfig")
            actions_per_chunk = config.actions_per_chunk
            if isinstance(actions_per_chunk, bool):
                raise ValueError("actions_per_chunk must be an integer not boolean")
            try:
                actions_per_chunk = operator.index(actions_per_chunk)
            except TypeError as error:
                raise ValueError(
                    "actions_per_chunk must be an integer"
                ) from error
            if not 1 <= actions_per_chunk <= MAX_ACTIONS_PER_CHUNK:
                raise ValueError("actions_per_chunk must be in range 1..50")
            state_feature = config.lerobot_features.get("observation.state")
            if not isinstance(state_feature, dict):
                raise ValueError("observation.state dict feature is required")
            state_names = state_feature.get("names")
            if (
                not isinstance(state_names, (list, tuple))
                or any(not isinstance(name, str) or not name for name in state_names)
                or len(set(state_names)) != len(state_names)
            ):
                raise ValueError(
                    "observation.state feature error"
                )
            state_names = tuple(state_names)
        except Exception as error:
            decode_error = str(error)
        if decode_error is not None:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Failed to decode LeRobot PolicySetup: {decode_error}",
            )

        session = self._lookup_session(context)
        with session.condition:
            session.actions_per_chunk = actions_per_chunk
            session.state_names = state_names
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

        decode_error = None
        session = self._lookup_session(context)
        try:
            observation = self._decode_observation(
                bytes(payload), session.state_names
            )
        except Exception as error:
            decode_error = str(error)
        if decode_error is not None:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Failed to decode LeRobot observation: {decode_error}",
            )

        with session.condition:
            session.pending_observation = observation
            session.condition.notify()
        return services_pb2.Empty()

    def GetActions(self, request, context):
        total_start = time.perf_counter()
        session = self._lookup_session(context)
        with session.condition:
            ready = session.condition.wait_for(
                lambda: session.pending_observation is not None,
                timeout=2.0,
            )
            if not ready:
                return services_pb2.Actions()
            observation = session.pending_observation
            horizon = session.actions_per_chunk
            session.pending_observation = None

        request_id = f"lerobot-gateway-{next(self._request_ids)}"
        inference_request = self._build_request(
            observation, request_id, horizon, context
        )
        try:
            response, model_request_infer_ms = self._model_client.infer(
                inference_request, MODEL_NAME
            )
        except grpc.RpcError as error:
            abort_for_backend_error(context, error)
        except Exception as error:
            abort_for_backend_error(context, error)

        self._validate_response(response, request_id, horizon, context)

        encode_start = time.perf_counter()
        action_dim = response.actions.shape[1]
        action_array = np.frombuffer(
            response.actions.data,
            dtype="<f4",
        ).reshape(horizon, action_dim).copy()
        action_tensor = torch.from_numpy(action_array)
        if observation.timestep < 0 or observation.timestep > 2**63 - horizon:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Action chunk timestep is invalid",
            )
        timestamp = model_request_infer_ms / 1000.0
        action_chunk = []
        for index in range(horizon):
            action_chunk.append(
                TimedAction(
                    timestamp=timestamp,
                    timestep=observation.timestep + index,
                    action=action_tensor[index],
                )
            )
        action_data = pickle.dumps(action_chunk)
        pickle_encode_ms = (time.perf_counter() - encode_start) * 1000.0

        result = services_pb2.Actions(data=action_data)
        total_gateway_ms = (time.perf_counter() - total_start) * 1000.0
        with self._latency_lock:
            if self._latency_enabled:
                try:
                    with self._latency_path.open(
                        "a", encoding="utf-8", newline=""
                    ) as file:
                        file.write(
                            ",".join((
                                str(pickle_encode_ms),
                                str(model_request_infer_ms),
                                str(response.inference_time_us / 1000.0),
                                str(total_gateway_ms),
                            )) + chr(10)
                        )
                except OSError:
                    self._latency_enabled = False
        return result

    @staticmethod
    def _decode_observation(payload, state_names):
        decoded = pickle.loads(payload)
        try:
            timestamp = float(decoded.timestamp)
            try:
                timestep = operator.index(decoded.timestep)
            except TypeError as error:
                raise ValueError(
                    "timestep must be a signed 64-bit integer"
                ) from error
            if not -(2**63) <= timestep < 2**63:
                raise ValueError("timestep must be a signed 64-bit integer")
            must_go = bool(decoded.must_go)
            observation = decoded.observation
        except AttributeError as error:
            raise ValueError(
                "TimedObservation is missing required attributes"
            ) from error

        if not isinstance(observation, dict):
            raise ValueError("TimedObservation.observation is not a dict")
        task = observation.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("Observation task must be a nonempty string")

        state_value = observation.get("observation.state")


        """
        The current setup is not sufficiently generic: a LeRobot client using a
        different robot may not be directly compatible. Adding a separate adapter
        for every robot would also be undesirable. Although PolicySetup provides
        some dynamic registration, one Gateway may still support only one robot
        configuration at a time.

        Possible solution:
            Maintain a table of robot state layouts and match against it, for example:
            {"Libero-robot": {libero-features}, ...}
        """
        if state_value is None:
            if len(state_names) != 8:
                raise ValueError("Session state feature is not configured")
            try:
                state_value = np.asarray(
                    [observation[name] for name in state_names],
                    dtype=np.float32,
                )
            except KeyError as error:
                raise ValueError(
                    f"Observation state field is missing: {error.args[0]}"
                ) from error
        if isinstance(state_value, torch.Tensor):
            if state_value.device.type != "cpu":
                raise ValueError("Observation state tensor must be on CPU")
            if state_value.dtype != torch.float32:
                raise ValueError("Observation state dtype must be float32")
            state_value = state_value.detach().numpy()
        if not isinstance(state_value, np.ndarray):
            raise ValueError(
                "Observation state must be a NumPy array or CPU torch.Tensor"
            )
        if state_value.dtype != np.float32:
            raise ValueError("Observation state dtype must be float32")
        if state_value.shape == (1, 8):
            state_value = state_value[0]
        if state_value.shape != (8,):
            raise ValueError("Observation state shape must be [8] or [1,8]")
        if not np.isfinite(state_value).all():
            raise ValueError("Observation state must contain finite values")
        state = tuple(
            state_value.astype("<f4", copy=False).tolist()
        )

        images = {}
        for key in ("agentview", "robot0_eye_in_hand"):
            value = observation.get(key)
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Image '{key}' must be a NumPy array")
            if value.dtype != np.uint8:
                raise ValueError(f"Image '{key}' dtype must be uint8")
            if value.ndim != 3 or value.shape[2] != 3:
                raise ValueError(
                    f"Image '{key}' shape must be nonempty HWC with 3 channels"
                )
            if any(dimension <= 0 for dimension in value.shape):
                raise ValueError(f"Image '{key}' must be nonempty")
            data = value.tobytes(order="C")
            expected_bytes = math.prod(value.shape)
            if len(data) != expected_bytes:
                raise ValueError(
                    f"Image '{key}' byte length does not match shape"
                )
            images[key] = (tuple(int(x) for x in value.shape), data)
        return _DecodedObservation(
            timestamp=timestamp,
            timestep=timestep,
            must_go=must_go,
            task=task,
            state=tuple(state),
            images=images,
        )

    @staticmethod
    def _build_request(observation, request_id, horizon, context):
        if (
            not math.isfinite(observation.timestamp)
            or observation.timestamp < 0.0
        ):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "LeRobot observation timestamp is invalid",
            )
        timestamp_ns = int(observation.timestamp * 1_000_000_000)
        if timestamp_ns > 2**63 - 1:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "LeRobot observation timestamp is invalid",
            )
        images = []
        for image_name in ("agentview", "robot0_eye_in_hand"):
            shape, image_data = observation.images[image_name]
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
            request_id=request_id,
            timestamp_ns=timestamp_ns,
            images=images,
            robot_state=model_inference_pb2.Tensor(
                data=struct.pack("<8f", *observation.state),
                shape=[8],
                dtype=model_inference_pb2.DATA_TYPE_FLOAT32,
            ),
            instruction=observation.task,
            requested_action_horizon=horizon,
        )

    @staticmethod
    def _validate_response(response, request_id, horizon, context):
        if response.request_id != request_id:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server returned a mismatched request_id",
            )
        actions = response.actions
        if actions.dtype != model_inference_pb2.DATA_TYPE_FLOAT32:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server actions must use FLOAT32",
            )
        if len(actions.shape) != 2 or actions.shape[0] != horizon:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server actions must have shape [requested_horizon, action_dim]",
            )
        action_dim = actions.shape[1]
        if action_dim <= 0:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server action_dim must be positive",
            )
        if len(actions.data) != horizon * action_dim * 4:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server action shape does not match its data length",
            )

    @staticmethod
    def _session_id(context):
        for item in context.invocation_metadata():
            if item.key == "x-session-id":
                value = item.value
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="strict")
                if value:
                    return value
                break
        context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            "x-session-id is required",
        )

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
