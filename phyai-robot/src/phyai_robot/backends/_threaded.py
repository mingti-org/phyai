"""Private worker plumbing shared by asynchronous transports (ROS 2 and ZMQ).

Threading Architecture:
    Native transport handles (such as rclpy Nodes, executors, subscriptions, and
    pyzmq Sockets) are strictly confined to a single dedicated worker thread.
    Cross-thread interactions are constrained to:
    1. Lock-protected read cache (`_samples`): Worker stores immutable, frozen Samples
       with local arrival timestamps; external callers read instantly without blocking.
    2. Size-1 Request Queue (`_requests`): External callers submit bounded command writes,
       stops, or synchronous service calls using Future objects and monotonic deadlines.
       An overflow immediately raises an error instead of queueing backlog commands.
    3. Separate Timeout Bounds: Stop operations use an independent `stop_timeout_s`
       budget (longer than `io_timeout_s`) to accommodate synchronous controller service
       handshakes without premature timeout faults.
"""

from __future__ import annotations

import math
import time
import queue
import threading
from types import MappingProxyType
from typing import Any, Literal
from _thread import LockType
from dataclasses import dataclass
from collections.abc import Mapping, Callable, Iterable
from concurrent.futures import Future, wait

import numpy as np
from numpy.typing import NDArray

from ..types import Action, Sample
from .._compat import BaseExceptionGroup
from ..validation import freeze, freeze_values

_Kind = Literal["write", "stop", "call"]


@dataclass(frozen=True)
class _Request:
    """Internal task item dispatched to the transport worker thread."""

    kind: _Kind  # Task type: "write", "stop", or "call".
    value: Any  # Payload: detached Action for writes, callable for call, None for stop.
    deadline: (
        float  # Monotonic deadline in seconds; verified on caller and worker thread.
    )
    future: Future[Any]  # Completion future capturing return value or raised exception.


def _positive(name: str, value: float) -> None:
    """Verify that a timeout parameter is a finite, positive float."""
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


class _ThreadedBackend:
    """Base class managing worker thread lifecycle and synchronized request dispatch."""

    def __init__(
        self,
        observation_keys: Iterable[str],  # Sensor field names owned by this session.
        action_keys: Iterable[
            str
        ],  # Actuator command field names owned by this session.
        io_timeout_s: float,  # Maximum budget for publishing one single-step command.
        connect_timeout_s: float,  # Maximum budget for worker startup and shutdown.
        stop_timeout_s: float
        | None = None,  # Maximum budget for stop/service handshake (None = io).
    ) -> None:
        stop_timeout_s = io_timeout_s if stop_timeout_s is None else stop_timeout_s
        for name, value in (
            ("io_timeout_s", io_timeout_s),
            ("connect_timeout_s", connect_timeout_s),
            ("stop_timeout_s", stop_timeout_s),
        ):
            _positive(name, value)

        self._observation_keys: frozenset[str] = frozenset(observation_keys)
        self._action_keys: frozenset[str] = frozenset(action_keys)
        if any(
            not isinstance(key, str) or not key
            for key in self._observation_keys | self._action_keys
        ):
            raise ValueError("backend keys must be nonempty strings")

        self._io_timeout_s = io_timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._stop_timeout_s = stop_timeout_s

        # Lock synchronizing the observation sample cache and worker exception recording.
        self._lock: LockType = threading.Lock()
        self._samples: dict[str, Sample] = {}
        self._error: BaseException | None = None

        # Capacity-1 queue prevents accumulating an unbounded backlog of actuator commands.
        self._requests: queue.Queue[_Request] = queue.Queue(maxsize=1)
        self._ready = threading.Event()  # Set when worker finishes initialization.
        self._halt = threading.Event()  # Set to request worker thread loop termination.
        self._thread: threading.Thread | None = None
        self._closed = False
        self._stop_requested = False
        self._device_stopped = False  # Set by worker after completing _stop_device.

    @property
    def observation_keys(self) -> frozenset[str]:
        """Return the immutable set of sensor fields owned by this session."""
        return self._observation_keys

    @property
    def action_keys(self) -> frozenset[str]:
        """Return the immutable set of actuator command fields owned by this session."""
        return self._action_keys

    def _raise_error(self) -> None:
        """Raise RuntimeError if the transport worker thread has encountered an exception."""
        with self._lock:
            error: BaseException | None = self._error
        if error is not None:
            raise RuntimeError("transport worker failed") from error

    def _record_error(self, error: BaseException) -> None:
        """Atomically record the first exception or aggregate subsequent ones."""
        with self._lock:
            if self._error is None:
                self._error = error
            elif self._error is not error:
                self._error = BaseExceptionGroup(
                    "transport failures", [self._error, error]
                )

    def _cache(self, values: Mapping[str, NDArray[Any]], received_at_ns: int) -> None:
        """Store newly decoded sensor arrays in the observation cache.

        Arrays are frozen here once (flags.writeable=False). Downstream layers
        (read(), CompositeRobot, Deployment) share these immutable buffers without copies.
        """
        if not set(values) <= self.observation_keys:
            raise ValueError("decoder returned undeclared observation fields")
        if any(not isinstance(value, np.ndarray) for value in values.values()):
            raise TypeError("decoders must return NumPy arrays")

        samples = {key: Sample(freeze(v), received_at_ns) for key, v in values.items()}
        with self._lock:
            self._samples.update(samples)

    def _on_worker(self) -> bool:
        """Return True if currently executing on the backend's worker thread."""
        return self._thread is not None and threading.current_thread() is self._thread

    def connect(self) -> None:
        """Start the worker thread and wait until transport initialization completes."""
        if self._closed or self._stop_requested:
            raise RuntimeError("transport is closed or stopped")
        if self._thread is not None:
            self._raise_error()
            return

        self._thread = threading.Thread(
            target=self._run, name=type(self).__name__, daemon=True
        )
        self._thread.start()

        if not self._ready.wait(self._connect_timeout_s):
            error: TimeoutError = TimeoutError("transport initialization timed out")
            self._record_error(error)
            self._halt.set()
            raise error

        self._raise_error()

    def read(self) -> Mapping[str, Sample]:
        """Return an immutable mapping view of currently cached sensor samples."""
        if self._thread is None or self._closed:
            raise RuntimeError("transport is not connected")
        self._raise_error()
        with self._lock:
            return MappingProxyType(dict(self._samples))

    def _submit(self, kind: _Kind, value: Any, timeout_s: float) -> Any:
        """Submit a task to the worker thread and wait for completion up to timeout_s."""
        if self._thread is None or not self._thread.is_alive() or self._closed:
            self._raise_error()
            raise RuntimeError("transport worker is not running")
        if kind != "stop":
            self._raise_error()

        future: Future[Any] = Future()
        deadline: float = time.monotonic() + timeout_s
        try:
            self._requests.put_nowait(_Request(kind, value, deadline, future))
        except queue.Full:
            raise RuntimeError("a transport request is already pending") from None

        # Wait separately from future.result() to distinguish wait timeouts from task timeouts.
        if not wait([future], timeout=max(0.0, deadline - time.monotonic())).done:
            future.cancel()
            error = TimeoutError(f"transport {kind} timed out")
            self._record_error(error)
            self._halt.set()
            raise error

        return future.result()

    def write(self, action: Action) -> None:
        """Submit a single-step command mapping to be written by the worker thread."""
        if self._stop_requested:
            raise RuntimeError("transport has been stopped")
        if set(action) != self.action_keys:
            raise ValueError("transport command fields do not match")
        self._submit("write", freeze_values(action), self._io_timeout_s)

    def call(self, function: Callable[[], Any], timeout_s: float) -> Any:
        """Execute a function on the worker thread and return its result.

        Enables thread-safe auxiliary operations (such as ROS 2 service calls)
        using the worker's dedicated context and executor.
        """
        _positive("timeout_s", timeout_s)
        if self._on_worker():
            return function()
        if self._stop_requested:
            raise RuntimeError("transport has been stopped")
        return self._submit("call", function, timeout_s + self._io_timeout_s)

    def stop(self) -> None:
        """Command the session's actuators to enter safe state within stop_timeout_s."""
        if self._stop_requested:
            return
        self._stop_requested = True
        if (
            self._thread is not None
            and self._thread.is_alive()
            and not self._halt.is_set()
        ):
            self._submit("stop", None, self._stop_timeout_s)

    def close(self) -> None:
        """Terminate the worker thread and release underlying transport resources."""
        if self._closed:
            return
        self._halt.set()
        if self._thread is not None:
            self._thread.join(self._connect_timeout_s + self._stop_timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("transport worker did not release its resources")
        self._closed = True
        self._raise_error()

    def _execute(self, request: _Request) -> Any:
        """Execute a single submitted request on the worker thread."""
        if time.monotonic() >= request.deadline:
            raise TimeoutError("transport request expired before execution")
        if request.kind == "stop":
            self._stop_device()
            self._device_stopped = True
            return None
        if request.kind == "call":
            return request.value()
        self._write_device(request.value, request.deadline)
        return None

    def _run(self) -> None:
        """Worker thread main loop: process requests and poll transport events."""
        request: _Request | None
        try:
            self._open()
            self._ready.set()
            while not self._halt.is_set():
                try:
                    request = self._requests.get_nowait()
                except queue.Empty:
                    request = None

                if (
                    request is not None
                    and request.future.set_running_or_notify_cancel()
                ):
                    try:
                        request.future.set_result(self._execute(request))
                    except BaseException as error:
                        request.future.set_exception(error)
                        if request.kind != "call":
                            raise

                self._poll()
        except BaseException as error:  # noqa: BLE001 - propagate worker errors
            self._record_error(error)
        finally:
            self._ready.set()
            # Guarantee hardware stop and cleanup even if worker loop crashes.
            for cleanup in (
                self._stop_device if not self._device_stopped else lambda: None,
                self._close_device,
            ):
                try:
                    cleanup()
                except BaseException as error:  # noqa: BLE001
                    self._record_error(error)

            # Cancel any remaining pending request in queue.
            while True:
                try:
                    pending: _Request = self._requests.get_nowait()
                except queue.Empty:
                    break
                if pending.future.set_running_or_notify_cancel():
                    pending.future.set_exception(
                        RuntimeError("transport worker stopped")
                    )

    def _open(self) -> None:
        """Subclass hook: Initialize native context, sockets, nodes, and handlers."""
        raise NotImplementedError

    def _poll(self) -> None:
        """Subclass hook: Process native polling/spinning events (e.g. spin_once)."""
        raise NotImplementedError

    def _write_device(self, action: Action, deadline: float) -> None:
        """Subclass hook: Transmit command action to device within deadline."""
        raise NotImplementedError

    def _stop_device(self) -> None:
        """Subclass hook: Transition device to safe state upon stop or shutdown."""
        raise NotImplementedError

    def _close_device(self) -> None:
        """Subclass hook: Clean up and close native sockets, nodes, and contexts."""
        raise NotImplementedError
