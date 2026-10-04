import threading

import grpc

from phyai_gateway.bindings import model_inference_pb2, model_inference_pb2_grpc
from phyai_gateway.services.model_registry import ModelRegistryService

INFERENCE_TIMEOUT_SECONDS = 600
MAX_MESSAGE_BYTES = 100 * 1024 * 1024


class NoHealthyModelServerError(Exception):
    pass


class ModelInferenceClient:
    def __init__(self, registry: ModelRegistryService):
        self._registry = registry
        self._stubs_lock = threading.Lock()
        self._channels: dict[str, grpc.Channel] = {}
        self._stubs: dict[str, model_inference_pb2_grpc.ModelInferenceStub] = {}
        self._closed = False

    def infer(
        self, request: model_inference_pb2.InferenceRequest, model_name: str
    ) -> model_inference_pb2.InferenceResponse:
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

            return stub.Infer(request, timeout=INFERENCE_TIMEOUT_SECONDS)
        finally:
            self._registry.release_server(selected.server_id)

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
    if isinstance(error, grpc.RpcError):
        trailing_metadata = error.trailing_metadata()
        if trailing_metadata:
            context.set_trailing_metadata(trailing_metadata)
        context.abort(error.code(), error.details())
    context.abort(grpc.StatusCode.INTERNAL, str(error))
