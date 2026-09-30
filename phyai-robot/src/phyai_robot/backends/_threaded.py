"""Private worker plumbing shared by the two asynchronous transports.

Socket/node ownership stays on one thread. Only copied sensor snapshots and
bounded command requests cross the thread boundary. This is not a public
backend framework; its hooks only remove identical lifecycle code.
"""

from __future__ import annotations

import math
import time
import queue
import threading
from typing import Any, Literal
from _thread import LockType
from dataclasses import dataclass
from collections.abc import Mapping, Iterable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout

import numpy as np
from numpy.typing import NDArray

from ..types import Action, Sample
from ..validation import freeze_values, freeze_samples


@dataclass(frozen=True)
class _Request:
    kind: Literal[
        "write", "stop"
    ]  # "write" or "stop"; no unbounded stream of pending device commands.
    value: Action | None  # Detached Action for writes, None for stop.
    deadline: float  # Local monotonic deadline, checked again on the I/O thread.
    future: Future[None]  # Submission result; never interpreted as physical completion.


class _ThreadedBackend:
    def __init__(
        self,
        observation_keys: Iterable[str],  # Fixed sensor names owned by this session.
        action_keys: Iterable[str],  # Fixed command names owned by this session.
        io_timeout_s: float,  # Maximum wait for a submitted command or stop.
        connect_timeout_s: float,  # Maximum wait for worker startup and shutdown.
    ) -> None:
        for value in (io_timeout_s, connect_timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "I/O and connection timeouts must be finite and positive"
                )
        self._observation_keys: frozenset[str] = frozenset(observation_keys)
        self._action_keys: frozenset[str] = frozenset(action_keys)
        if any(
            not isinstance(key, str) or not key
            for key in self._observation_keys | self._action_keys
        ):
            raise ValueError("backend keys must be nonempty strings")
        self._io_timeout_s: float = io_timeout_s
        self._connect_timeout_s: float = connect_timeout_s
        self._lock: LockType = (
            threading.Lock()
        )  # Protect cached samples and the first worker error.
        self._samples: dict[str, Sample] = {}
        self._error: BaseException | None = None
        self._requests: queue.Queue[_Request] = queue.Queue(maxsize=1)
        self._ready: threading.Event = threading.Event()
        self._halt: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed: bool = False
        self._stop_requested: bool = False
        self._device_stopped: bool = (
            False  # Updated only by the worker after successful stop.
        )

    @property
    def observation_keys(self) -> frozenset[str]:
        return self._observation_keys

    @property
    def action_keys(self) -> frozenset[str]:
        return self._action_keys

    def _raise_error(self) -> None:
        with self._lock:
            error: BaseException | None = self._error
        if error is not None:
            raise RuntimeError("transport worker failed") from error

    def _record_error(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
            elif self._error is not error:
                self._error = BaseExceptionGroup(
                    "transport failures", [self._error, error]
                )

    def _cache(self, values: Mapping[str, NDArray[Any]], received_at_ns: int) -> None:
        if not set(values) <= self.observation_keys:
            raise ValueError("decoder returned undeclared observation fields")
        if any(not isinstance(value, np.ndarray) for value in values.values()):
            raise TypeError("decoders must return NumPy arrays")
        copied: Action = freeze_values(values)
        with self._lock:
            self._samples.update(
                {key: Sample(value, received_at_ns) for key, value in copied.items()}
            )

    def connect(self) -> None:
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
        if self._thread is None or self._closed:
            raise RuntimeError("transport is not connected")
        self._raise_error()
        with self._lock:
            return freeze_samples(self._samples)

    def _submit(
        self, kind: Literal["write", "stop"], value: Action | None = None
    ) -> None:
        if self._thread is None or not self._thread.is_alive() or self._closed:
            self._raise_error()
            raise RuntimeError("transport worker is not running")
        if kind == "write":
            self._raise_error()
        future: Future[None] = Future()
        deadline: float = time.monotonic() + self._io_timeout_s
        request: _Request = _Request(kind, value, deadline, future)
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            raise RuntimeError("a transport request is already pending") from None
        try:
            future.result(timeout=max(0.0, deadline - time.monotonic()))
        except FutureTimeout as error:
            # A pending request is cancelled. An already-running native operation
            # cannot be undone, so fault the session and stop it on the worker.
            future.cancel()
            self._record_error(error)
            self._halt.set()
            raise TimeoutError(f"transport {kind} timed out") from error

    def write(self, action: Action) -> None:
        if self._stop_requested:
            raise RuntimeError("transport has been stopped")
        if set(action) != self.action_keys:
            raise ValueError("transport command fields do not match")
        self._submit("write", freeze_values(action))

    def stop(self) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        if (
            self._thread is not None
            and self._thread.is_alive()
            and not self._halt.is_set()
        ):
            self._submit("stop")

    def close(self) -> None:
        if self._closed:
            return
        self._halt.set()
        if self._thread is not None:
            self._thread.join(self._connect_timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("transport worker did not release its resources")
        self._closed = True
        self._raise_error()

    def _run(self) -> None:
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
                        if time.monotonic() >= request.deadline:
                            raise TimeoutError(
                                "transport request expired before execution"
                            )
                        if request.kind == "stop":
                            self._stop_device()
                            self._device_stopped = True
                        else:
                            # write() always submits an Action; stop() submits None.
                            assert request.value is not None
                            self._write_device(request.value, request.deadline)
                        request.future.set_result(None)
                    except BaseException as error:
                        request.future.set_exception(error)
                        raise
                self._poll()
        except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
            self._record_error(error)
        finally:
            self._ready.set()
            # Even decoder failures stop the device. Cleanup still runs if stop
            # itself fails. Hooks tolerate resources that were only partly opened.
            for cleanup in (
                self._stop_device if not self._device_stopped else lambda: None,
                self._close_device,
            ):
                try:
                    cleanup()
                except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
                    self._record_error(error)
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
        raise NotImplementedError

    def _poll(self) -> None:
        raise NotImplementedError

    def _write_device(self, action: Action, deadline: float) -> None:
        raise NotImplementedError

    def _stop_device(self) -> None:
        raise NotImplementedError

    def _close_device(self) -> None:
        raise NotImplementedError
