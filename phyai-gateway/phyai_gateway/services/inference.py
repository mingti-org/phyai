import grpc

from phyai_gateway.bindings import model_inference_pb2, model_inference_pb2_grpc
from phyai_gateway.clients.model_inference import (
    ModelInferenceClient,
    abort_for_backend_error,
)


class ModelInferenceService(model_inference_pb2_grpc.ModelInferenceServicer):
    def __init__(self, model_client: ModelInferenceClient):
        self._model_client = model_client

    def CheckHealth(self, request, context):
        model_name = request.model_name.strip()
        if not model_name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model_name is required")
        return model_inference_pb2.HealthResponse(
            healthy=self._model_client.has_healthy_server(model_name)
        )

    def Infer(self, request, context):
        model_name = request.model_name.strip()
        if not model_name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "model_name is required")
        if not request.request_id.strip():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id is required")

        try:
            response = self._model_client.infer(request, model_name, context=context)
        except Exception as error:
            abort_for_backend_error(context, error)

        if response.request_id != request.request_id:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server response request_id does not match request",
            )
        return response
