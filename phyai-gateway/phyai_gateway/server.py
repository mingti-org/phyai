import argparse
import signal
import threading
from concurrent import futures

import grpc
from lerobot.transport import services_pb2_grpc

from phyai_gateway.adapters.lerobot import LeRobotAdapter
from phyai_gateway.adapters.robot import RobotAdapter
from phyai_gateway.bindings import model_inference_pb2_grpc, robot_pb2_grpc
from phyai_gateway.clients.model_inference import ModelInferenceClient
from phyai_gateway.http_server import HTTPServer
from phyai_gateway.services.model_registry import ModelRegistryService

MAX_MESSAGE_BYTES = 100 * 1024 * 1024
INFERENCE_SHUTDOWN_GRACE_SECONDS = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="0.0.0.0:50111")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=30000)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.threads),
        options=[("grpc.max_receive_message_length", MAX_MESSAGE_BYTES)],
    )
    registry = ModelRegistryService()
    model_client = ModelInferenceClient(registry)
    robot_adapter = RobotAdapter(model_client)
    lerobot_adapter = LeRobotAdapter(model_client)

    model_inference_pb2_grpc.add_ModelRegistryServicer_to_server(
        registry, server
    )
    robot_pb2_grpc.add_RobotInferenceServicer_to_server(
        robot_adapter, server
    )
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(
        lerobot_adapter, server
    )

    bound_port = server.add_insecure_port(args.gateway)
    if bound_port == 0:
        model_client.close()
        raise RuntimeError(f"failed to bind Gateway to {args.gateway}")

    http_server = HTTPServer(args.http_host, args.http_port, model_client)
    http_server.start()

    shutdown_lock = threading.Lock()
    stopped = False

    def shutdown(signum=None, frame=None):
        nonlocal stopped
        with shutdown_lock:
            if stopped:
                return
            stopped = True
        http_server.stop()
        server.stop(INFERENCE_SHUTDOWN_GRACE_SECONDS).wait()
        model_client.close()

    server.start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"Python Gateway gRPC listening on {args.gateway}", flush=True)
    print(
        f"Python Gateway HTTP listening on {args.http_host}:{args.http_port}",
        flush=True,
    )
    try:
        server.wait_for_termination()
    finally:
        shutdown()

