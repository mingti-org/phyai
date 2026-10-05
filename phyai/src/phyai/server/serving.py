"""Serve model backends through the gateway inference protocol."""

from __future__ import annotations

from concurrent import futures
from functools import partial
import importlib
import os
import signal
import threading
from typing import TYPE_CHECKING

import grpc

from phyai.utils import configure_logging, get_logger
from phyai_gateway.bindings import model_inference_pb2 as pb
from phyai_gateway.bindings import model_inference_pb2_grpc as rpc

if TYPE_CHECKING:
    from phyai.server.config import ServerConfig

logger = get_logger(__name__)


def load_callable(reference: str):
    module_name, separator, name = reference.partition(":")
    if not separator or not module_name or not name or ":" in name:
        raise ValueError("Python callable references must use module:name")
    value = importlib.import_module(module_name)
    for attribute in name.split("."):
        value = getattr(value, attribute)
    if not callable(value):
        raise TypeError(f"{reference!r} is not callable")
    return value


class ModelInferenceServicer(rpc.ModelInferenceServicer):
    def __init__(self, backend, model_name: str, inference_executor):
        self.backend = backend
        self.model_name = model_name
        self.stopping = False
        self.Infer = partial(self.Infer)
        self.Infer.experimental_thread_pool = inference_executor

    def CheckHealth(self, request, context):
        route_matches = (
            not request.model_name or request.model_name.strip() == self.model_name
        )
        return pb.HealthResponse(
            healthy=route_matches and not self.stopping and self.backend.healthy
        )

    def Infer(self, request, context):
        if request.model_name and request.model_name.strip() != self.model_name:
            context.abort(grpc.StatusCode.NOT_FOUND, "model_name is not served here")
        if not request.request_id.strip():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id is required")
        if self.stopping or not self.backend.healthy:
            context.abort(grpc.StatusCode.UNAVAILABLE, "model backend is unavailable")
        if not context.is_active():
            context.abort(grpc.StatusCode.CANCELLED, "request was cancelled")
        try:
            response = self.backend.infer(request, context)
        except grpc.RpcError as error:
            if error.trailing_metadata():
                context.set_trailing_metadata(error.trailing_metadata())
            context.abort(error.code(), error.details())
        except Exception:
            if context.code() not in (None, grpc.StatusCode.OK):
                raise
            logger.exception("Inference failed for request %s", request.request_id)
            context.abort(grpc.StatusCode.INTERNAL, "model inference failed")
        if not isinstance(response, pb.InferenceResponse):
            context.abort(
                grpc.StatusCode.INTERNAL, "backend returned an invalid response"
            )
        if response.request_id != request.request_id:
            context.abort(
                grpc.StatusCode.DATA_LOSS, "backend response request_id mismatch"
            )
        return response


def serve(config: ServerConfig) -> None:
    configure_logging()
    options = config.server
    if config.deployment is not None and config.deployment.mode == "external":
        engine_config = config.engine_args.config
        world_size = (
            engine_config.parallel.infer_replica_world_size()
            if engine_config is not None
            else 1
        )
        if world_size > 1 or int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError(
                "external multi-rank serving requires a request broadcaster; "
                "use local deployment for managed model workers"
            )
    host = options.host.removeprefix("[").removesuffix("]")
    host = f"[{host}]" if ":" in host else host
    address = f"{host}:{options.port}"
    main_thread = threading.current_thread() is threading.main_thread()
    previous_sigterm = signal.getsignal(signal.SIGTERM) if main_thread else None
    if main_thread:
        signal.signal(signal.SIGTERM, signal.default_int_handler)

    try:
        factory = load_callable(config.adapter_factory)
        backend = factory(
            engine_args=config.engine_args,
            deployment=config.deployment,
            **config.adapter_args,
        )
        try:
            with (
                futures.ThreadPoolExecutor(
                    max_workers=options.workers
                ) as inference_executor,
                futures.ThreadPoolExecutor(max_workers=1) as health_executor,
            ):
                service = ModelInferenceServicer(
                    backend, options.model_name, inference_executor
                )
                server = grpc.server(
                    health_executor,
                    options=[
                        ("grpc.max_receive_message_length", options.max_message_bytes),
                        ("grpc.max_send_message_length", options.max_message_bytes),
                        ("grpc.so_reuseport", 0),
                    ],
                )
                try:
                    rpc.add_ModelInferenceServicer_to_server(service, server)
                    if not server.add_insecure_port(address):
                        raise RuntimeError(f"failed to bind model server to {address}")
                    server.start()
                    logger.info("Serving model %s on %s", options.model_name, address)
                    server.wait_for_termination()
                finally:
                    service.stopping = True
                    server.stop(options.grace_seconds).wait()
        finally:
            backend.close()
    except KeyboardInterrupt:
        logger.info("Model server stopped")
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, previous_sigterm)
