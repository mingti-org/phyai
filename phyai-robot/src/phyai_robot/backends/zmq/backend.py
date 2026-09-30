"""A specific ZMQ transport: SUB observations and REQ/REP command acceptance.

Wire formats belong to application codecs. All sockets are created, used, and
closed on the same worker thread. This backend is not a policy RPC client.
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
    # Runtime imports stay inside connect so the base package needs no pyzmq.
    import zmq


class ZmqBackend(_ThreadedBackend):
    def __init__(
        self,
        *,
        observation_endpoint: str,  # Address of the peer's PUB socket.
        command_endpoint: str,  # Address of the peer's REP socket.
        observation_keys: frozenset[str],  # All allowed decoded sensor fields.
        action_keys: frozenset[str],  # Complete command subset sent to this peer.
        decode_observation: Callable[[Sequence[bytes]], Mapping[str, NDArray[Any]]],
        # Multipart sensor message -> only the fields updated by that message.
        encode_action: Callable[[Action], Sequence[bytes]],  # Command -> wire frames.
        encode_stop: Callable[
            [], Sequence[bytes]
        ],  # Device-specific safe-state request.
        check_reply: Callable[[Sequence[bytes]], None],  # Raise on rejection/bad reply.
        io_timeout_s: float = 0.1,  # End-to-end command acceptance budget.
        connect_timeout_s: float = 5.0,  # Local socket/worker setup budget.
    ) -> None:
        super().__init__(observation_keys, action_keys, io_timeout_s, connect_timeout_s)
        self._observation_endpoint: str = observation_endpoint
        self._command_endpoint: str = command_endpoint
        self._decode: Callable[[Sequence[bytes]], Mapping[str, NDArray[Any]]] = (
            decode_observation
        )
        self._encode: Callable[[Action], Sequence[bytes]] = encode_action
        self._encode_stop: Callable[[], Sequence[bytes]] = encode_stop
        self._check_reply: Callable[[Sequence[bytes]], None] = check_reply
        self._zmq: ModuleType | None = (
            None  # Loaded only on connect, so base package imports stay light.
        )
        self._context: zmq.Context | None = (
            None  # Owned context, never the application's global instance.
        )
        self._subscriber: zmq.Socket | None = None
        self._requester: zmq.Socket | None = None

    def _open(self) -> None:
        try:
            import zmq
        except ModuleNotFoundError as error:
            if error.name == "zmq":
                raise ImportError(
                    "ZmqBackend requires the 'zmq' extra of phyai-robot (pyzmq). "
                    "See the package README for uv installation commands."
                ) from error
            raise
        self._zmq = zmq
        context: zmq.Context = zmq.Context()
        self._context = context
        subscriber: zmq.Socket = context.socket(zmq.SUB)
        self._subscriber = subscriber
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.RCVHWM, 32)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.connect(self._observation_endpoint)
        self._new_requester()

    def _new_requester(self) -> None:
        assert self._context is not None and self._zmq is not None
        if self._requester is not None:
            self._requester.close(linger=0)
        requester: zmq.Socket = self._context.socket(self._zmq.REQ)
        self._requester = requester
        requester.setsockopt(self._zmq.LINGER, 0)
        requester.setsockopt(self._zmq.SNDHWM, 1)
        # Do not retain a command on a socket that has no connected peer.
        requester.setsockopt(self._zmq.IMMEDIATE, 1)
        requester.connect(self._command_endpoint)

    def _poll(self) -> None:
        assert self._subscriber is not None  # _open completed on this worker.
        if self._subscriber.poll(timeout=1):
            frames: list[bytes] = self._subscriber.recv_multipart()
            received_at_ns: int = time.monotonic_ns()
            self._cache(self._decode(frames), received_at_ns)

    def _exchange(self, frames: Sequence[bytes], deadline: float) -> None:
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
            # A timed-out REQ cannot send another request in its current state.
            # Destroy it so stop uses a fresh connection and cannot consume a
            # previous command's late reply as the stop acknowledgement.
            self._requester.close(linger=0)
            self._requester = None
            raise

    def _write_device(self, action: Action, deadline: float) -> None:
        self._exchange(self._encode(action), deadline)

    def _stop_device(self) -> None:
        if self._context is None or not self.action_keys:
            return
        if self._requester is None:
            self._new_requester()
        self._exchange(self._encode_stop(), time.monotonic() + self._io_timeout_s)

    def _close_device(self) -> None:
        for socket in (self._subscriber, self._requester):
            if socket is not None:
                socket.close(linger=0)
        if self._context is not None:
            self._context.term()
