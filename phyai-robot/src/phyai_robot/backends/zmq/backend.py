"""Hardware device transport over ZeroMQ: SUB observations and REQ/REP command actuation.

Design & Role:
    This module implements a physical device transport session (`ZmqBackend`) using ZeroMQ:
    - Sensors: Ingested via an asynchronous ZMQ SUB socket subscribing to remote device PUB endpoints.
    - Commands: Dispatched via a synchronous ZMQ REQ/REP socket exchanging command acceptance handshakes.
    - Wire Codecs: Application-provided encode/decode callables define the byte-level multipart framing.
    - Thread Safety: All ZMQ sockets and contexts are confined exclusively to the backend's worker thread.

Note on Architecture:
    `ZmqBackend` is an actuator/sensor transport for physical devices that expose a ZMQ interface.
    It is completely separate from `phyai_robot.remote.RemotePolicy` / `PolicyServer`, which
    implement the model policy inference RPC.
"""

from __future__ import annotations

import math
import time
from types import ModuleType
from typing import TYPE_CHECKING, Any
from collections.abc import Mapping, Callable, Sequence

from numpy.typing import NDArray

from ...types import Action
from .._threaded import _ThreadedBackend

if TYPE_CHECKING:
    import zmq


class ZmqBackend(_ThreadedBackend):
    """Transport session managing a ZMQ SUB socket for sensors and REQ socket for commands."""

    def __init__(
        self,
        *,
        observation_endpoint: str,  # URL of remote device PUB socket (e.g. 'tcp://192.168.1.10:6000').
        command_endpoint: str,  # URL of remote device REP socket (e.g. 'tcp://192.168.1.10:6001').
        observation_keys: frozenset[
            str
        ],  # Declared observation field names produced by this session.
        action_keys: frozenset[
            str
        ],  # Declared action command field names written to this session.
        decode_observation: Callable[[Sequence[bytes]], Mapping[str, NDArray[Any]]],
        # Codec translating raw multipart message frames into decoded field arrays.
        encode_action: Callable[
            [Action], Sequence[bytes]
        ],  # Codec translating command Action to wire frames.
        encode_stop: Callable[
            [], Sequence[bytes]
        ],  # Encodes device-specific safe-state request.
        check_reply: Callable[
            [Sequence[bytes]], None
        ],  # Validates remote REP acknowledgement reply.
        io_timeout_s: float = 0.1,  # Maximum budget for command transmission and acknowledgement.
        connect_timeout_s: float = 5.0,  # Budget for worker and socket initialization.
        stop_timeout_s: float | None = None,  # Budget for stop request handshake.
    ) -> None:
        super().__init__(
            observation_keys,
            action_keys,
            io_timeout_s,
            connect_timeout_s,
            stop_timeout_s,
        )
        self._observation_endpoint: str = observation_endpoint
        self._command_endpoint: str = command_endpoint
        self._decode: Callable[[Sequence[bytes]], Mapping[str, NDArray[Any]]] = (
            decode_observation
        )
        self._encode: Callable[[Action], Sequence[bytes]] = encode_action
        self._encode_stop: Callable[[], Sequence[bytes]] = encode_stop
        self._check_reply: Callable[[Sequence[bytes]], None] = check_reply
        self._zmq: ModuleType | None = None
        self._context: zmq.Context | None = None
        self._subscriber: zmq.Socket | None = None
        self._requester: zmq.Socket | None = None

    def _open(self) -> None:
        """Initialize ZMQ context, SUB socket, and REQ socket on the worker thread."""
        try:
            import zmq
        except ModuleNotFoundError as error:
            if error.name == "zmq":
                raise ImportError(
                    "ZmqBackend requires the 'zmq' extra of phyai-robot (pyzmq). "
                    "Install with: pip install 'phyai-robot[zmq]'"
                ) from error
            raise

        self._zmq = zmq
        context: zmq.Context = zmq.Context()
        self._context = context

        # Configure subscriber socket.
        subscriber: zmq.Socket = context.socket(zmq.SUB)
        self._subscriber = subscriber
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.RCVHWM, 32)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.connect(self._observation_endpoint)

        self._new_requester()

    def _new_requester(self) -> None:
        """Create a fresh REQ socket for commands, discarding any previous socket."""
        assert self._context is not None and self._zmq is not None
        if self._requester is not None:
            self._requester.close(linger=0)
        requester: zmq.Socket = self._context.socket(self._zmq.REQ)
        self._requester = requester
        requester.setsockopt(self._zmq.LINGER, 0)
        requester.setsockopt(self._zmq.SNDHWM, 1)
        requester.setsockopt(
            self._zmq.IMMEDIATE, 1
        )  # Drop message if remote peer is disconnected.
        requester.connect(self._command_endpoint)

    def _poll(self) -> None:
        """Poll the SUB socket for incoming sensor packets on the worker thread."""
        assert self._subscriber is not None
        if self._subscriber.poll(timeout=1):
            frames: list[bytes] = self._subscriber.recv_multipart()
            received_at_ns: int = time.monotonic_ns()
            self._cache(self._decode(frames), received_at_ns)

    def _exchange(self, frames: Sequence[bytes], deadline: float) -> None:
        """Send a request over REQ and receive response before monotonic deadline."""
        assert self._requester is not None and self._zmq is not None
        if not frames or any(not isinstance(frame, bytes) for frame in frames):
            raise TypeError("wire codecs must return a nonempty sequence of bytes")
        try:
            for event in (self._zmq.POLLOUT, self._zmq.POLLIN):
                remaining: float = deadline - time.monotonic()
                if remaining <= 0 or not self._requester.poll(
                    math.ceil(remaining * 1000), event
                ):
                    raise TimeoutError(
                        "ZMQ peer did not accept/reply before the deadline"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("ZMQ request expired")
                if event == self._zmq.POLLOUT:
                    self._requester.send_multipart(frames, flags=self._zmq.NOBLOCK)
                else:
                    self._check_reply(
                        self._requester.recv_multipart(flags=self._zmq.NOBLOCK)
                    )
        except BaseException:
            # On exchange error/timeout, REQ socket state is invalid. Replace socket.
            self._requester.close(linger=0)
            self._requester = None
            raise

    def _write_device(self, action: Action, deadline: float) -> None:
        """Encode action and execute REQ/REP exchange with remote device."""
        self._exchange(self._encode(action), deadline)

    def _stop_device(self) -> None:
        """Send device-specific safe state command to remote device."""
        if self._context is None or not self.action_keys:
            return
        if self._requester is None:
            self._new_requester()
        self._exchange(self._encode_stop(), time.monotonic() + self._stop_timeout_s)

    def _close_device(self) -> None:
        """Terminate sockets and context on worker thread."""
        for socket in (self._subscriber, self._requester):
            if socket is not None:
                socket.close(linger=0)
        if self._context is not None:
            self._context.term()
