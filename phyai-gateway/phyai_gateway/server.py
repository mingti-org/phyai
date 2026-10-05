import argparse
import logging
import math
import signal
from concurrent import futures
from functools import partial

import grpc
import uvicorn

from phyai_gateway.adapters.robot import RobotAdapter
from phyai_gateway.bindings import model_inference_pb2_grpc, robot_pb2_grpc
from phyai_gateway.clients.model_inference import ModelInferenceClient
from phyai_gateway.http_server import MAX_MESSAGE_BYTES, create_http_app
from phyai_gateway.services.health import BackendHealthMonitor
from phyai_gateway.services.inference import ModelInferenceService
from phyai_gateway.services.model_registry import ModelRegistryService

SHUTDOWN_GRACE_SECONDS = 5
logger = logging.getLogger(__name__)


def _exit_on_sigterm(signum, frame):
    raise SystemExit(0)


def _set_executor(service, methods, executor):
    for name in methods:
        handler = partial(getattr(service, name))
        handler.experimental_thread_pool = executor
        setattr(service, name, handler)
    return service


def main():
    parser = argparse.ArgumentParser(
        description="Route inference requests to registered model servers."
    )
    parser.add_argument("--gateway", default="0.0.0.0:50111")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=30000)
    parser.add_argument(
        "--backend",
        nargs=2,
        action="append",
        metavar=("MODEL", "ENDPOINT"),
        default=[],
        help="add a backend to probe for health; repeat for multiple backends",
    )
    parser.add_argument(
        "--http-model", help="default model for HTTP clients that omit a model name"
    )
    parser.add_argument(
        "--robot-model",
        help="enable RobotInference for a backend returning joint targets and a gripper opening",
    )
    parser.add_argument(
        "--lerobot",
        action="store_true",
        help="enable LeRobot's pickle protocol for trusted clients (requires the lerobot extra)",
    )
    parser.add_argument("--lerobot-fps", type=float, default=30)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if not 1 <= args.http_port <= 65535:
        parser.error("--http-port must be in range 1..65535")
    if not math.isfinite(args.lerobot_fps) or args.lerobot_fps <= 0:
        parser.error("--lerobot-fps must be positive and finite")

    registry = ModelRegistryService()
    try:
        for model_name, endpoint in args.backend:
            registry.add_backend(model_name, endpoint)
    except ValueError as error:
        parser.error(str(error))
    if args.lerobot:
        try:
            from lerobot.transport import services_pb2_grpc
            from phyai_gateway.adapters.lerobot import LeRobotAdapter
        except ImportError as error:
            parser.error(f"LeRobot adapter requires phyai-gateway[lerobot]: {error}")

    logging.basicConfig(level=logging.INFO)
    monitor = BackendHealthMonitor(registry)
    model_client = ModelInferenceClient(registry)
    with (
        futures.ThreadPoolExecutor(
            max_workers=args.threads, thread_name_prefix="gateway-inference"
        ) as inference_executor,
        futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="gateway-control"
        ) as control_executor,
    ):
        server = grpc.server(
            control_executor,
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
                ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
                ("grpc.so_reuseport", 0),
            ],
        )
        previous_sigterm = signal.signal(signal.SIGTERM, _exit_on_sigterm)
        try:
            model_inference_pb2_grpc.add_ModelRegistryServicer_to_server(
                registry, server
            )
            model_inference_pb2_grpc.add_ModelInferenceServicer_to_server(
                _set_executor(
                    ModelInferenceService(model_client), ("Infer",), inference_executor
                ),
                server,
            )
            if args.robot_model:
                robot_pb2_grpc.add_RobotInferenceServicer_to_server(
                    _set_executor(
                        RobotAdapter(model_client, model_name=args.robot_model),
                        ("Communicate",),
                        inference_executor,
                    ),
                    server,
                )
            if args.lerobot:
                services_pb2_grpc.add_AsyncInferenceServicer_to_server(
                    _set_executor(
                        LeRobotAdapter(model_client, fps=args.lerobot_fps),
                        (
                            "Ready",
                            "SendPolicyInstructions",
                            "SendObservations",
                            "GetActions",
                        ),
                        inference_executor,
                    ),
                    server,
                )
            if server.add_insecure_port(args.gateway) == 0:
                raise RuntimeError(f"failed to bind gateway to {args.gateway}")
            server.start()
            monitor.start()
            logger.info("Gateway gRPC listening on %s", args.gateway)
            uvicorn.run(
                create_http_app(
                    model_client, default_model=args.http_model, registry=registry
                ),
                host=args.http_host,
                port=args.http_port,
                timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS,
            )
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            server.stop(SHUTDOWN_GRACE_SECONDS).wait()
            monitor.close()
            model_client.close()


if __name__ == "__main__":
    main()
