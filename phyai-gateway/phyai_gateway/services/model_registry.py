import threading
import time
from dataclasses import dataclass

from phyai_gateway.bindings import (
    model_inference_pb2,
    model_inference_pb2_grpc,
)

HEARTBEAT_INTERVAL_SECONDS = 5
HEARTBEAT_TIMEOUT_SECONDS = 15


@dataclass
class RegisteredModelServer:
    server_id: str
    endpoint: str
    model_name: str
    last_heartbeat: float
    in_flight_requests: int = 0


@dataclass(frozen=True)
class SelectedModelServer:
    server_id: str
    endpoint: str


class ModelRegistryService(model_inference_pb2_grpc.ModelRegistryServicer):
    def __init__(self):
        self._lock = threading.Lock()
        self._servers = {}
        self._next_server_id = 1

    def Register(self, request, context):
        if not request.endpoint:
            return model_inference_pb2.RegisterResponse(
                accepted=False,
                message="endpoint is required",
            )
        if not request.model_name:
            return model_inference_pb2.RegisterResponse(
                accepted=False,
                message="model_name is required",
            )

        with self._lock:
            server_id = f"model-server-{self._next_server_id}"
            self._next_server_id += 1
            self._servers[server_id] = RegisteredModelServer(
                server_id=server_id,
                endpoint=request.endpoint,
                model_name=request.model_name,
                last_heartbeat=time.monotonic(),
            )

        print(
            f"[Model Registry] Registered {server_id} "
            f"model={request.model_name} endpoint={request.endpoint}",
            flush=True,
        )
        return model_inference_pb2.RegisterResponse(
            accepted=True,
            server_id=server_id,
            heartbeat_interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
        )

    def Heartbeat(self, request, context):
        if not request.server_id:
            return model_inference_pb2.HeartbeatResponse(
                registered=False,
                message="server_id is required",
            )

        with self._lock:
            server = self._servers.get(request.server_id)
            if server is None:
                return model_inference_pb2.HeartbeatResponse(
                    registered=False,
                    message="server_id is not registered",
                )
            server.last_heartbeat = time.monotonic()

        return model_inference_pb2.HeartbeatResponse(registered=True)

    def acquire_server(self, model_name):
        now = time.monotonic()
        with self._lock:
            selected = None
            for server in self._servers.values():
                if server.model_name != model_name:
                    continue
                if now - server.last_heartbeat > HEARTBEAT_TIMEOUT_SECONDS:
                    continue
                if (
                    selected is None
                    or server.in_flight_requests < selected.in_flight_requests
                ):
                    selected = server

            if selected is None:
                return None

            selected.in_flight_requests += 1
            return SelectedModelServer(
                server_id=selected.server_id,
                endpoint=selected.endpoint,
            )

    def release_server(self, server_id):
        with self._lock:
            server = self._servers.get(server_id)
            if server is not None and server.in_flight_requests > 0:
                server.in_flight_requests -= 1
