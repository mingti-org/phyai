import logging
import threading
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit

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
    managed: bool = False
    healthy: bool = True
    health_failures: int = 0
    removed: bool = False


@dataclass(frozen=True)
class SelectedModelServer:
    server_id: str
    endpoint: str
    model_name: str


def validate_backend(model_name, endpoint):
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name is required")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint is required")
    endpoint = endpoint.strip().removeprefix("grpc://")
    try:
        address = urlsplit(f"//{endpoint}")
        valid = (
            address.hostname
            and address.port is not None
            and 1 <= address.port <= 65535
            and address.username is None
            and not (address.path or address.query or address.fragment)
            and not any(char in endpoint for char in "?#\\")
            and not any(
                char.isspace() or ord(char) < 32 or ord(char) == 127
                for char in endpoint
            )
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("endpoint must be HOST:PORT or [IPv6]:PORT with port 1..65535")
    return model_name.strip(), endpoint


class ModelRegistryService(model_inference_pb2_grpc.ModelRegistryServicer):
    def __init__(self):
        self._lock = threading.Lock()
        self._servers = {}

    def Register(self, request, context):
        try:
            server = self._register(request.model_name, request.endpoint, managed=False)
        except ValueError as error:
            return model_inference_pb2.RegisterResponse(
                accepted=False, message=str(error)
            )
        return model_inference_pb2.RegisterResponse(
            accepted=True,
            server_id=server["server_id"],
            heartbeat_interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
        )

    def add_backend(self, model_name, endpoint):
        model_name, endpoint = validate_backend(model_name, endpoint)
        return self._register(model_name, endpoint, managed=True)

    def _register(self, model_name, endpoint, *, managed):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name is required")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint is required")
        model_name, endpoint = model_name.strip(), endpoint.strip()
        with self._lock:
            now = time.monotonic()
            self._remove_expired(now)
            server = next(
                (
                    server
                    for server in self._servers.values()
                    if not server.removed
                    and server.endpoint == endpoint
                    and server.model_name == model_name
                ),
                None,
            )
            if server is None:
                server_id = uuid.uuid4().hex
                server = RegisteredModelServer(
                    server_id,
                    endpoint,
                    model_name,
                    now,
                    managed=managed,
                    healthy=not managed,
                )
                self._servers[server_id] = server
            else:
                server_id = server.server_id
                server.last_heartbeat = now
                if managed and not server.managed:
                    server.managed = True
                    server.healthy = False
            result = self._describe(server, now)

        logger.info(
            "Registered %s model=%s endpoint=%s", server_id, model_name, endpoint
        )
        return result

    def Heartbeat(self, request, context):
        if not request.server_id:
            return model_inference_pb2.HeartbeatResponse(
                registered=False,
                message="server_id is required",
            )

        with self._lock:
            now = time.monotonic()
            self._remove_expired(now)
            server = self._servers.get(request.server_id)
            if (
                server is None
                or server.removed
                or (not server.managed and not self._healthy(server, now))
            ):
                return model_inference_pb2.HeartbeatResponse(
                    registered=False,
                    message="server_id is not registered",
                )
            server.last_heartbeat = now

        return model_inference_pb2.HeartbeatResponse(registered=True)

    def list_backends(self):
        with self._lock:
            now = time.monotonic()
            self._remove_expired(now)
            return [
                self._describe(server, now)
                for server in self._servers.values()
                if not server.removed
            ]

    def remove_backend(self, server_id):
        with self._lock:
            server = self._servers.get(server_id)
            if server is None or server.removed:
                return False
            server.removed = True
            self._remove_expired(time.monotonic())
            return True

    def health_targets(self):
        with self._lock:
            return [
                SelectedModelServer(
                    server.server_id, server.endpoint, server.model_name
                )
                for server in self._servers.values()
                if server.managed and not server.removed
            ]

    def record_health(self, server_id, healthy, *, failure_threshold=3):
        with self._lock:
            server = self._servers.get(server_id)
            if server is None or server.removed or not server.managed:
                return
            was_healthy = server.healthy
            if healthy:
                server.health_failures = 0
                server.healthy = True
            else:
                server.health_failures += 1
                if server.health_failures >= failure_threshold:
                    server.healthy = False
            if was_healthy != server.healthy:
                logger.info(
                    "Backend %s model=%s healthy=%s",
                    server.endpoint,
                    server.model_name,
                    server.healthy,
                )

    def has_healthy_server(self, model_name: str) -> bool:
        with self._lock:
            now = time.monotonic()
            self._remove_expired(now)
            return any(
                server.model_name == model_name
                and not server.removed
                and self._healthy(server, now)
                for server in self._servers.values()
            )

    def acquire_server(self, model_name):
        with self._lock:
            now = time.monotonic()
            self._remove_expired(now)
            selected = None
            for server in self._servers.values():
                if server.model_name != model_name:
                    continue
                if server.removed or not self._healthy(server, now):
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
                model_name=selected.model_name,
            )

    def release_server(self, server_id):
        with self._lock:
            server = self._servers.get(server_id)
            if server is not None and server.in_flight_requests > 0:
                server.in_flight_requests -= 1
            self._remove_expired(time.monotonic())

    def endpoints(self) -> set[str]:
        with self._lock:
            self._remove_expired(time.monotonic())
            return {server.endpoint for server in self._servers.values()}

    def _remove_expired(self, now):
        expired = [
            server_id
            for server_id, server in self._servers.items()
            if (
                server.removed
                or (
                    not server.managed
                    and now - server.last_heartbeat > HEARTBEAT_TIMEOUT_SECONDS
                )
            )
            and server.in_flight_requests == 0
        ]
        for server_id in expired:
            del self._servers[server_id]

    @staticmethod
    def _healthy(server, now):
        if server.managed:
            return server.healthy
        return now - server.last_heartbeat <= HEARTBEAT_TIMEOUT_SECONDS

    def _describe(self, server, now):
        return {
            "server_id": server.server_id,
            "model_name": server.model_name,
            "endpoint": server.endpoint,
            "healthy": self._healthy(server, now),
            "in_flight_requests": server.in_flight_requests,
        }
