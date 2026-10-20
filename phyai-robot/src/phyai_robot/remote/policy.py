"""Synchronous model RPC with a resident predictor and leased run ownership.

Architecture:
    This module implements Protocol v2 of the phyai-robot remote inference protocol:
    1. Zero Robot Imports: The model server (`PolicyServer`) operates exclusively on named
       model arrays (such as 'image.<camera>' and 'state' -> 'actions'). It has zero dependencies
       on robot hardware, ROS 2, or robot-specific layout code.
    2. Zero Memory Copies: Multipart ZMQ messages carry raw contiguous C-order bytes.
       Data is packed from and unpacked directly into NumPy array buffers without serialization copies.
    3. Socket Lifecycle & Thread Handoff: `RemotePolicy` reuses a single ZMQ REQ socket protected
       by a mutual-exclusion Lock. The lock acquisition provides the memory fence required by ZMQ
       to safely hand off the underlying socket across thread boundaries (e.g. from the warmup
       thread to the deployment background inference worker thread).
    4. Autonomous Lease Expiration: `PolicyServer` enforces an idle timeout (`lease_timeout_s`,
       default 60s). If an active client process terminates abruptly or loses connectivity without
       calling "end", the server automatically frees its run reservation, avoiding permanent lockup.
    5. No Automatic Retries: Hardware safety dictates that failed RPC exchanges are never replayed,
       preventing duplicated noise draws, trajectory desynchronization, or stale command emission.
"""

from __future__ import annotations

import math
import time
import uuid
import logging
from typing import Any, Protocol
from threading import Lock, Event
from collections.abc import Mapping, Callable

from numpy.typing import NDArray

from ._wire import (
    MAX_BYTES,
    PolicySpec,
    pack,
    unpack,
    read_values,
    values_payload,
)

logger = logging.getLogger(__name__)

Arrays = Mapping[str, NDArray[Any]]


class RemotePolicyError(RuntimeError):
    """Exception raised when the policy server rejects a request or returns mismatched run IDs."""


def _positive(name: str, value: float) -> None:
    """Verify that a timeout parameter is a finite, positive float."""
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


class RemotePolicy:
    """Synchronous ZMQ client exposing a Policy interface over a remote network socket.

    Lifecycle:
        1. Query model contract: `spec = client.info()`
        2. Reserve server run:   `client.begin_run()`
        3. Execute warmup:       `client.warmup(inputs)` (discard output)
        4. Cyclic inference:     `outputs = client.predict(inputs)`
        5. Terminate run:        `client.close()`

    Concurrency & Thread Safety:
        Calls to info, begin_run, warmup, predict, and close are serialized under `_lock`.
        If close() is called while a background prediction is in-flight on another thread,
        the socket is not destroyed underneath it; the owning thread releases the run upon return.
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        *,
        timeout_s: float = 2.0,
        warmup_timeout_s: float = 120.0,
    ) -> None:
        """Initialize RemotePolicy client.

        Args:
            endpoint: ZMQ socket endpoint URL (e.g. 'tcp://127.0.0.1:5555').
            timeout_s: Per-exchange timeout budget in seconds for regular calls (info, begin, predict, end).
            warmup_timeout_s: Extended timeout budget for warmup (allows lazy JIT & CUDA graph capture).
        """
        _positive("timeout_s", timeout_s)
        _positive("warmup_timeout_s", warmup_timeout_s)
        self.endpoint = endpoint
        self.timeout_s, self.warmup_timeout_s = timeout_s, warmup_timeout_s

        self._lock = (
            Lock()
        )  # Serializes RPC calls and provides cross-thread socket handoff fence.
        self._closing = Event()  # Signals client termination to active in-flight calls.
        self._context: Any = None
        self._socket: Any = None
        self._run_id: str | None = None
        self._request_id = 0
        self._spec: PolicySpec | None = None
        self._failed = False
        self._warmed = False

    @property
    def spec(self) -> PolicySpec | None:
        """Return the PolicySpec returned by begin_run, or None before run reservation."""
        return self._spec

    def _connect(self) -> Any:
        """Lazily initialize ZMQ context and REQ socket with low-latency options."""
        import zmq

        if self._context is None:
            self._context = zmq.Context()
        if self._socket is None:
            socket = self._context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(
                zmq.IMMEDIATE, 1
            )  # Drop message if server endpoint is not connected.
            socket.setsockopt(zmq.MAXMSGSIZE, MAX_BYTES)
            socket.connect(self.endpoint)
            self._socket = socket
        return self._socket

    def _disconnect(self) -> None:
        """Safely close and destroy the active REQ socket."""
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def _rpc(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        arrays: Any = (),
        *,
        timeout_s: float | None = None,
    ) -> tuple[dict[str, Any], list[NDArray[Any]]]:
        """Perform a single synchronous request-reply exchange over ZMQ.

        Must be called while holding self._lock. Monotonic deadline covers socket readiness,
        message transmission, and server reply reception.
        """
        import zmq

        self._request_id += 1
        request_id, run_id = self._request_id, self._run_id
        frames = pack(
            dict(
                payload or {}, operation=operation, request_id=request_id, run_id=run_id
            ),
            arrays,
        )
        deadline = time.monotonic() + (
            self.timeout_s if timeout_s is None else timeout_s
        )
        socket = self._connect()

        def wait(event: int) -> None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not socket.poll(
                max(1, math.ceil(remaining * 1000)), event
            ):
                raise TimeoutError(
                    f"policy {operation} timed out; request was not retried"
                )

        try:
            wait(zmq.POLLOUT)
            socket.send_multipart(frames, flags=zmq.DONTWAIT, copy=False)
            wait(zmq.POLLIN)
            response, values = unpack(socket.recv_multipart(flags=zmq.DONTWAIT))
        except BaseException:
            # When an exchange times out or fails, REQ socket state becomes invalid.
            # Destroy the socket so subsequent calls don't consume late response frames.
            self._disconnect()
            raise

        if response.get("request_id") != request_id:
            raise RemotePolicyError("policy response request ID mismatch")
        if operation != "begin" and response.get("run_id") != run_id:
            raise RemotePolicyError("policy response run ID mismatch")
        if not response.get("ok"):
            raise RemotePolicyError(response.get("error", "policy request failed"))
        return response, values

    def _ensure_open(self) -> None:
        """Verify that the client has not encountered an unrecoverable failure or closed."""
        if self._closing.is_set() or self._failed:
            raise RuntimeError("RemotePolicy is closed or failed; create a new client")

    def info(self) -> PolicySpec:
        """Query model specifications and input/output schemas without reserving a run."""
        with self._lock:
            self._ensure_open()
            result, _ = self._rpc("info")
            return PolicySpec.from_dict(result["spec"])

    def begin_run(self) -> PolicySpec:
        """Reserve server run, reset RNG noise stream, and receive validated PolicySpec."""
        with self._lock:
            self._ensure_open()
            if self._run_id is not None:
                raise RuntimeError("this client already owns a run")
            try:
                response, _ = self._rpc("begin")
                run_id = response.get("run_id")
                if not isinstance(run_id, str) or not run_id:
                    raise RemotePolicyError("missing server run ID")
                self._run_id = run_id
                self._spec = PolicySpec.from_dict(response["spec"])
                return self._spec
            except BaseException:
                self._failed = True
                raise

    def warmup(self, inputs: Arrays) -> dict[str, NDArray[Any]]:
        """Trigger initial model warmup inference; output must be discarded by caller."""
        return self._predict(inputs, warmup=True)

    def predict(self, request: Arrays) -> dict[str, NDArray[Any]]:
        """Execute one synchronous prediction cycle on model input arrays."""
        return self._predict(request, warmup=False)

    def _predict(self, inputs: Arrays, *, warmup: bool) -> dict[str, NDArray[Any]]:
        with self._lock:
            self._ensure_open()
            if self._run_id is None or self._spec is None:
                raise RuntimeError("begin_run is required before prediction")
            if warmup == self._warmed:
                raise RuntimeError("warmup is required exactly once before predict")
            try:
                meta, arrays = values_payload(inputs, self._spec.inputs)
                response, values = self._rpc(
                    "warmup" if warmup else "predict",
                    meta,
                    arrays,
                    timeout_s=self.warmup_timeout_s if warmup else self.timeout_s,
                )
                outputs = read_values(response, values, self._spec.outputs)
                self._warmed = True
                return outputs
            except BaseException:
                self._failed = True
                raise
            finally:
                if self._closing.is_set():
                    self._shutdown()

    def _shutdown(self) -> None:
        """End the server run and terminate ZMQ sockets; caller must hold self._lock."""
        try:
            if self._run_id is not None:
                try:
                    self._rpc("end")
                except Exception as error:  # noqa: BLE001 - lease will expire if unreachable
                    logger.warning("Could not release run %s: %s", self._run_id, error)
                finally:
                    self._run_id = None
        finally:
            self._disconnect()
            if self._context is not None:
                self._context.term()
                self._context = None

    def close(self) -> None:
        """Initiate client shutdown and release remote run.

        Safe to invoke while background predictions are in-flight; ownership cleanup
        is safely deferred to the executing thread without destroying sockets.
        """
        self._closing.set()
        if self._lock.acquire(blocking=False):
            try:
                self._shutdown()
            finally:
                self._lock.release()


class ModelPolicy(Protocol):
    """Protocol for server-side policy predictors operating on named model arrays."""

    def predict(self, inputs: Arrays) -> Arrays:
        """Run inference forward pass on preprocessed model input arrays."""
        ...


class PolicyServer:
    """ZMQ server hosting a resident ModelPolicy and managing sequential robot runs.

    Features:
    - 1-at-a-time Exclusivity: Only one client run can hold reservation at any time.
    - Automatic Lease Recovery: Automatically frees reservation if client is silent
      for `lease_timeout_s` (default 60s).
    - Unchanged Model State: Serves multiple robot episodes without reloading model weights.
    """

    def __init__(
        self,
        *,
        policy: ModelPolicy,
        spec: PolicySpec,
        reset_run: Callable[[], None],
        endpoint: str = "tcp://127.0.0.1:5555",
        lease_timeout_s: float = 60.0,
    ) -> None:
        """Initialize PolicyServer.

        Args:
            policy: Server-side model predictor.
            spec: Declared PolicySpec defining input/output schemas and metadata.
            reset_run: Callback to reset the initial diffusion noise generator.
            endpoint: ZMQ bind URL (e.g. 'tcp://127.0.0.1:5555').
            lease_timeout_s: Inactive run lease expiration duration in seconds.
        """
        spec.to_dict()
        _positive("lease_timeout_s", lease_timeout_s)
        self.policy, self.spec = policy, spec
        self.reset_run, self.endpoint = reset_run, endpoint
        self.lease_timeout_s = lease_timeout_s
        self._run_id: str | None = None
        self._last_request_id = -1
        self._warmed = False
        self._lease_deadline = 0.0

    @property
    def run_id(self) -> str | None:
        """Return the active run reservation identifier, or None if idle."""
        return self._run_id

    def expire_lease(self, now: float | None = None) -> bool:
        """Check lease expiration and automatically free abandoned run reservations."""
        now = time.monotonic() if now is None else now
        if self._run_id is None or now < self._lease_deadline:
            return False
        logger.warning(
            "Releasing run %s after %.1f s without requests",
            self._run_id,
            self.lease_timeout_s,
        )
        self._run_id = None
        return True

    def _dispatch(
        self, request: Mapping[str, Any], arrays: list[NDArray[Any]]
    ) -> tuple[dict[str, Any], list[NDArray[Any]], str | None]:
        """Dispatch a single parsed request and return response tuple (meta, arrays, run_id)."""
        op, request_id = request["operation"], request["request_id"]
        if type(request_id) is not int or request_id < 1:
            raise ValueError("invalid request ID")

        if op == "info":
            return dict(spec=self.spec.to_dict()), [], request.get("run_id")

        self.expire_lease()

        if op == "begin":
            if self._run_id is not None:
                raise RuntimeError(
                    "server already has an active run; wait for it to end or for "
                    f"its {self.lease_timeout_s:g} s lease to expire"
                )
            self.reset_run()
            self._run_id, self._warmed = uuid.uuid4().hex, False
            self._last_request_id = request_id
            return (
                dict(spec=self.spec.to_dict(), lease_timeout_s=self.lease_timeout_s),
                [],
                self._run_id,
            )

        if self._run_id is None or request.get("run_id") != self._run_id:
            raise ValueError("stale, expired or unknown run ID")
        if request_id <= self._last_request_id:
            raise ValueError("stale request ID")
        self._last_request_id = request_id

        if op == "end":
            run_id, self._run_id = self._run_id, None
            return {}, [], run_id

        if op not in ("warmup", "predict"):
            raise ValueError("unknown policy operation")
        if (op == "warmup") == self._warmed:
            raise ValueError("warmup is required exactly once before predict")

        inputs = read_values(request, arrays, self.spec.inputs)
        payload, values = values_payload(self.policy.predict(inputs), self.spec.outputs)
        self._warmed = True
        return payload, values, self._run_id

    def serve(
        self, *, stop_event: Event | None = None, ready_event: Event | None = None
    ) -> None:
        """Bind ZMQ REP socket and serve client requests synchronously until stopped."""
        import zmq

        stop = Event() if stop_event is None else stop_event
        with zmq.Context() as context:
            with context.socket(zmq.REP) as socket:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.MAXMSGSIZE, MAX_BYTES)
                socket.bind(self.endpoint)
                if ready_event is not None:
                    ready_event.set()

                while not stop.is_set():
                    if not socket.poll(100, zmq.POLLIN):
                        self.expire_lease()
                        continue

                    request: dict[str, Any] = {}
                    run_id = None
                    try:
                        request, arrays = unpack(socket.recv_multipart())
                        meta, values, run_id = self._dispatch(request, arrays)
                        reply = pack(
                            dict(
                                meta,
                                ok=True,
                                request_id=request["request_id"],
                                run_id=run_id,
                            ),
                            values,
                        )
                    except Exception as error:
                        reply = pack(
                            dict(
                                ok=False,
                                error=f"{type(error).__name__}: {error}",
                                request_id=request.get("request_id"),
                                run_id=request.get("run_id"),
                            )
                        )

                    socket.send_multipart(reply, copy=False)
                    # Extend active lease deadline upon successful response transmission.
                    if run_id is not None and run_id == self._run_id:
                        self._lease_deadline = time.monotonic() + self.lease_timeout_s
