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
            pending = stub.Infer.future(request, timeout=deadline)
            try:
                if context is None and cancel_event is None:
                    return pending.result()
                # A RobotInference stream reuses its context for many requests.
                # Polling avoids retaining a cancellation callback for each frame.
                while (context is None or context.is_active()) and (
                    cancel_event is None or not cancel_event.is_set()
                ):
                    try:
                        return pending.result(timeout=0.1)
                    except grpc.FutureTimeoutError:
                        pass
                raise InferenceCancelledError()
            finally:
                pending.cancel()
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
