import logging
import math
import threading

import grpc

from phyai_gateway.bindings import model_inference_pb2, model_inference_pb2_grpc

logger = logging.getLogger(__name__)


class BackendHealthMonitor:
    def __init__(self, registry, *, interval=5.0, timeout=2.0, failure_threshold=3):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("health interval must be positive and finite")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("health timeout must be positive and finite")
        if (
            isinstance(failure_threshold, bool)
            or not isinstance(failure_threshold, int)
            or failure_threshold < 1
        ):
            raise ValueError("health failure threshold must be a positive integer")
        self._registry = registry
        self._interval = interval
        self._timeout = timeout
        self._failure_threshold = failure_threshold
        self._channels = {}
        self._stubs = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="backend-health", daemon=True
        )

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.check_once()
            except Exception:
                logger.exception("Backend health check failed")
            self._stop.wait(self._interval)

    def check_once(self):
        targets = self._registry.health_targets()
        endpoints = {target.endpoint for target in targets}
        for endpoint in self._channels.keys() - endpoints:
            self._channels.pop(endpoint).close()
            self._stubs.pop(endpoint)
        pending = []
        for target in targets:
            if self._stop.is_set():
                break
            stub = self._stubs.get(target.endpoint)
            if stub is None:
                channel = grpc.insecure_channel(target.endpoint)
                stub = model_inference_pb2_grpc.ModelInferenceStub(channel)
                self._channels[target.endpoint] = channel
                self._stubs[target.endpoint] = stub
            pending.append(
                (
                    target.server_id,
                    stub.CheckHealth.future(
                        model_inference_pb2.HealthRequest(model_name=target.model_name),
                        timeout=self._timeout,
                    ),
                )
            )
        for server_id, response in pending:
            if self._stop.is_set():
                response.cancel()
                continue
            try:
                healthy = response.result().healthy
            except grpc.RpcError:
                healthy = False
            self._registry.record_health(
                server_id, healthy, failure_threshold=self._failure_threshold
            )

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()
        self._stubs.clear()
