import logging
import threading
import time
import uuid
from dataclasses import dataclass

from phyai_gateway.bindings import (
    model_inference_pb2,
    model_inference_pb2_grpc,
)

HEARTBEAT_INTERVAL_SECONDS = 5
HEARTBEAT_TIMEOUT_SECONDS = 15
logger = logging.getLogger(__name__)


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

    def Register(self, request, context):
        endpoint = request.endpoint.strip()
        model_name = request.model_name.strip()
        if not endpoint:
            return model_inference_pb2.RegisterResponse(
                accepted=False,
                message="endpoint is required",
            )
        if not model_name:
            return model_inference_pb2.RegisterResponse(
                accepted=False,
                message="model_name is required",
            )

        with self._lock:
            now = time.monotonic()
            self._remove_expired(now)
            server = next(
                (
                    server
                    for server in self._servers.values()
                    if server.endpoint == endpoint and server.model_name == model_name
                ),
                None,
            )
            if server is None:
                server_id = uuid.uuid4().hex
                server = RegisteredModelServer(server_id, endpoint, model_name, now)
                self._servers[server_id] = server
            else:
                server_id = server.server_id
                server.last_heartbeat = now

        logger.info(
            "Registered %s model=%s endpoint=%s", server_id, model_name, endpoint
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
            self._remove_expired(now)
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

    def endpoints(self) -> set[str]:
        with self._lock:
            self._remove_expired(time.monotonic())
            return {server.endpoint for server in self._servers.values()}

    def _remove_expired(self, now):
        expired = [
            server_id
            for server_id, server in self._servers.items()
            if now - server.last_heartbeat > HEARTBEAT_TIMEOUT_SECONDS
            and server.in_flight_requests == 0
        ]
        for server_id in expired:
            del self._servers[server_id]
