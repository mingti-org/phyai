import json
import math
import time
import uuid
import threading

import grpc

from phyai_gateway.bindings import model_inference_pb2, model_inference_pb2_grpc
from phyai_gateway.services.model_registry import ModelRegistryService

INFERENCE_TIMEOUT_SECONDS = 600
MAX_MESSAGE_BYTES = 100 * 1024 * 1024


class NoHealthyModelServerError(Exception):
    pass


class InferenceCancelledError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.CANCELLED

    def details(self):
        return "request was cancelled"

    def trailing_metadata(self):
        return ()


class InvalidBatchedRequest(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.INVALID_ARGUMENT

    def details(self):
        return "batched request cannot be split safely"

    def trailing_metadata(self):
        return ()


class InvalidBatchedResponse(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.DATA_LOSS

    def details(self):
        return "backend returned invalid batched actions"

    def trailing_metadata(self):
        return ()


class InferenceDeadlineExceeded(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.DEADLINE_EXCEEDED

    def details(self):
        return "model inference deadline exceeded"

    def trailing_metadata(self):
        return ()


def _split_batch_request(request, max_batch_size):
    extensions = None
    if request.extensions_json:
        try:
            extensions = json.loads(request.extensions_json)
        except ValueError:
            extensions = None
    batch_images = [image for image in request.images if len(image.shape) == 4]
    declared_batch = (
        extensions.get("batch_size") if isinstance(extensions, dict) else None
    )
    if not batch_images:
        if type(declared_batch) is int and declared_batch > max_batch_size:
            raise InvalidBatchedRequest()
        return None
    batch_size = batch_images[0].shape[0]
    if batch_size <= max_batch_size:
        if type(declared_batch) is int and declared_batch > max_batch_size:
            raise InvalidBatchedRequest()
        return None
    if not request.request_id:
        raise InvalidBatchedRequest()
    if not isinstance(extensions, dict):
        raise InvalidBatchedRequest()
    instructions = list(request.instructions)
    if (
        request.inputs
        or len(batch_images) != len(request.images)
        or any(
            image.shape[0] != batch_size
            or image.encoding != model_inference_pb2.IMAGE_ENCODING_RAW
            or image.dtype != model_inference_pb2.DATA_TYPE_UINT8
            or image.layout
            not in (
                model_inference_pb2.IMAGE_LAYOUT_HWC,
                model_inference_pb2.IMAGE_LAYOUT_CHW,
            )
            or any(dimension <= 0 for dimension in image.shape)
            or len(image.data) != math.prod(image.shape)
            for image in request.images
        )
        or list(request.robot_state.shape)[:1] != [batch_size]
        or len(request.robot_state.shape) != 2
        or any(dimension <= 0 for dimension in request.robot_state.shape)
        or request.robot_state.dtype != model_inference_pb2.DATA_TYPE_FLOAT32
        or len(request.robot_state.data) != math.prod(request.robot_state.shape) * 4
        or type(extensions.get("batch_size")) is not int
        or extensions["batch_size"] != batch_size
        or len(instructions) != batch_size
        or any(
            not isinstance(value, str) or not value.strip() for value in instructions
        )
    ):
        raise InvalidBatchedRequest()
    chunks = []
    for start in range(0, batch_size, max_batch_size):
        end = min(start + max_batch_size, batch_size)
        chunk_size = end - start
        chunk_extensions = dict(extensions)
        chunk_extensions["batch_size"] = chunk_size
        chunk_extensions.pop("instructions", None)
        chunk = model_inference_pb2.InferenceRequest(
            request_id=f"{request.request_id}-{uuid.uuid4().hex}",
            timestamp_ns=request.timestamp_ns,
            instructions=instructions[start:end],
            requested_action_horizon=request.requested_action_horizon,
            extensions_json=json.dumps(chunk_extensions, separators=(",", ":")),
            model_name=request.model_name,
        )
        for image in request.images:
            stride = math.prod(image.shape[1:])
            chunk.images.add(
                name=image.name,
                data=image.data[start * stride : end * stride],
                shape=[chunk_size, *image.shape[1:]],
                dtype=image.dtype,
                encoding=image.encoding,
                layout=image.layout,
            )
        stride = math.prod(request.robot_state.shape[1:]) * 4
        chunk.robot_state.CopyFrom(
            model_inference_pb2.Tensor(
                data=request.robot_state.data[start * stride : end * stride],
                shape=[chunk_size, *request.robot_state.shape[1:]],
                dtype=request.robot_state.dtype,
            )
        )
        chunks.append(chunk)
    return chunks


def _validate_batch_response(request, chunk, response, action_shape):
    shape = tuple(response.actions.shape)
    if (
        response.request_id != chunk.request_id
        or response.outputs
        or response.extensions_json
        or response.actions.dtype != model_inference_pb2.DATA_TYPE_FLOAT32
        or len(shape) != 3
        or shape[0] != chunk.images[0].shape[0]
        or min(shape) <= 0
        or (
            request.requested_action_horizon
            and shape[1] != request.requested_action_horizon
        )
        or len(response.actions.data) != math.prod(shape) * 4
        or (action_shape is not None and shape[1:] != action_shape)
    ):
        raise InvalidBatchedResponse()
    return shape[1:]


def _merge_batch_responses(request, chunks, responses):
    return model_inference_pb2.InferenceResponse(
        request_id=request.request_id,
        inference_time_us=sum(response.inference_time_us for response in responses),
        actions=model_inference_pb2.Tensor(
            data=b"".join(response.actions.data for response in responses),
            shape=[
                sum(chunk.images[0].shape[0] for chunk in chunks),
                *responses[0].actions.shape[1:],
            ],
            dtype=model_inference_pb2.DATA_TYPE_FLOAT32,
        ),
    )


class ModelInferenceClient:
    def __init__(self, registry: ModelRegistryService):
        self._registry = registry
        self._stubs_lock = threading.Lock()
        self._channels: dict[str, grpc.Channel] = {}
        self._stubs: dict[str, model_inference_pb2_grpc.ModelInferenceStub] = {}
        self._closed = False

    def infer(
        self,
        request: model_inference_pb2.InferenceRequest,
        model_name: str,
        *,
        timeout: float | None = None,
        context: grpc.ServicerContext | None = None,
        cancel_event: threading.Event | None = None,
    ) -> model_inference_pb2.InferenceResponse:
        if (context is not None and not context.is_active()) or (
            cancel_event is not None and cancel_event.is_set()
        ):
            raise InferenceCancelledError()
        model_name = model_name.strip()
        if request.model_name and request.model_name.strip() != model_name:
            raise ValueError("request model_name does not match the selected route")
        if request.model_name != model_name:
            forwarded = model_inference_pb2.InferenceRequest()
            forwarded.CopyFrom(request)
            forwarded.model_name = model_name
            request = forwarded
        selected = self._registry.acquire_server(model_name)
        if selected is None:
            raise NoHealthyModelServerError(
                f"no healthy model server registered for model {model_name}"
            )

        try:
            with self._stubs_lock:
                if self._closed:
                    raise RuntimeError("ModelInferenceClient is closed")
                for endpoint in self._channels.keys() - self._registry.endpoints():
                    self._channels.pop(endpoint).close()
                    self._stubs.pop(endpoint)
                stub = self._stubs.get(selected.endpoint)
                if stub is None:
                    channel = grpc.insecure_channel(
                        selected.endpoint,
                        options=[
                            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
                            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
                        ],
                    )
                    stub = model_inference_pb2_grpc.ModelInferenceStub(channel)
                    self._channels[selected.endpoint] = channel
                    self._stubs[selected.endpoint] = stub

            deadline = INFERENCE_TIMEOUT_SECONDS
            if timeout is not None:
                deadline = min(deadline, timeout)
            if context is not None:
                remaining = context.time_remaining()
                if remaining is not None:
                    deadline = min(deadline, remaining)
            expires_at = time.monotonic() + deadline
            chunks = (
                _split_batch_request(request, selected.max_batch_size)
                if selected.max_batch_size is not None
                else None
            )
            requests = chunks if chunks is not None else [request]
            responses = []
            action_shape = None
            for chunk in requests:
                if (context is not None and not context.is_active()) or (
                    cancel_event is not None and cancel_event.is_set()
                ):
                    raise InferenceCancelledError()
                remaining = expires_at - time.monotonic()
                if remaining <= 0:
                    raise InferenceDeadlineExceeded()
                pending = stub.Infer.future(chunk, timeout=remaining)
                try:
                    if context is None and cancel_event is None:
                        response = pending.result()
                    else:
                        # Poll to respect cancellations without retaining callbacks.
                        while (context is None or context.is_active()) and (
                            cancel_event is None or not cancel_event.is_set()
                        ):
                            try:
                                response = pending.result(timeout=0.1)
                                break
                            except grpc.FutureTimeoutError:
                                pass
                        else:
                            raise InferenceCancelledError()
                finally:
                    pending.cancel()
                if chunks is not None:
                    action_shape = _validate_batch_response(
                        request, chunk, response, action_shape
                    )
                responses.append(response)
            if chunks is None:
                return responses[0]
            return _merge_batch_responses(request, chunks, responses)
        finally:
            self._registry.release_server(selected.server_id)

    def has_healthy_server(self, model_name: str) -> bool:
        with self._stubs_lock:
            if self._closed:
                return False
        return self._registry.has_healthy_server(model_name)

    def close(self) -> None:
        with self._stubs_lock:
            self._closed = True
            channels = list(self._channels.values())
            self._channels.clear()
            self._stubs.clear()
        for channel in channels:
            channel.close()


def abort_for_backend_error(context: grpc.ServicerContext, error: Exception) -> None:
    if isinstance(error, NoHealthyModelServerError):
        context.abort(grpc.StatusCode.UNAVAILABLE, str(error))
    if isinstance(error, grpc.FutureCancelledError):
        context.abort(grpc.StatusCode.CANCELLED, "request was cancelled")
    if isinstance(error, grpc.RpcError):
        trailing_metadata = error.trailing_metadata()
        if trailing_metadata:
            context.set_trailing_metadata(trailing_metadata)
        context.abort(error.code(), error.details())
    context.abort(grpc.StatusCode.INTERNAL, str(error))
