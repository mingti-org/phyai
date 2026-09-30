"""Local protocol round trips without a physical robot or remote service."""

import json
import time
import threading
from typing import TYPE_CHECKING, Any, cast
from collections.abc import Iterator, Sequence

import numpy as np
import pytest
from phyai_robot import Sample
from phyai_robot.backends.zmq import ZmqBackend

if TYPE_CHECKING:
    import zmq

else:
    zmq = pytest.importorskip("zmq")


class Peer:
    """A tiny robot peer; each socket is owned by the thread that created it."""

    def __init__(self) -> None:
        self.ready: threading.Event = threading.Event()
        self.done: threading.Event = threading.Event()
        self.commands: list[dict[str, Any]] = []  # Decoded JSON command messages.
        self.value: float = 0.0
        self.error: BaseException | None = None
        self.reject: bool = False
        self.reply_delay: float = 0.0
        self.thread: threading.Thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        context: zmq.Context = zmq.Context()
        publisher: zmq.Socket = context.socket(zmq.PUB)
        responder: zmq.Socket = context.socket(zmq.REP)
        try:
            self.observation_endpoint: str = (
                f"tcp://127.0.0.1:{publisher.bind_to_random_port('tcp://127.0.0.1')}"
            )
            self.command_endpoint: str = (
                f"tcp://127.0.0.1:{responder.bind_to_random_port('tcp://127.0.0.1')}"
            )
            self.ready.set()
            while not self.done.is_set():
                publisher.send_json({"q": [self.value]})
                if responder.poll(5):
                    # This test peer's wire protocol always sends JSON objects.
                    message: dict[str, Any] = cast(
                        dict[str, Any], responder.recv_json()
                    )
                    self.commands.append(message)
                    if message["op"] == "write":
                        self.value = message["value"]
                    if self.reply_delay:
                        self.done.wait(self.reply_delay)
                    responder.send_json(
                        {"ok": not self.reject or message["op"] == "stop"}
                    )
        except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
            self.error = error
            self.ready.set()
        finally:
            publisher.close(linger=0)
            responder.close(linger=0)
            context.term()


@pytest.fixture
def peer() -> Iterator[Peer]:
    peer: Peer = Peer()
    peer.thread.start()
    assert peer.ready.wait(2)
    if peer.error:
        raise peer.error
    yield peer
    peer.done.set()
    peer.thread.join(3)
    assert not peer.thread.is_alive()
    assert peer.error is None


def make_backend(peer: Peer, **overrides: Any) -> ZmqBackend:
    def check_reply(frames: Sequence[bytes]) -> None:
        if not json.loads(frames[0])["ok"]:
            raise ValueError("peer rejected command")

    # Transport parameters include addresses, codecs, and numeric timeouts.
    config: dict[str, Any] = {
        "observation_endpoint": peer.observation_endpoint,
        "command_endpoint": peer.command_endpoint,
        "observation_keys": frozenset({"q"}),
        "action_keys": frozenset({"target"}),
        "decode_observation": lambda frames: {
            "q": np.asarray(json.loads(frames[0])["q"], dtype=np.float32)
        },
        "encode_action": lambda action: [
            json.dumps({"op": "write", "value": float(action["target"][0])}).encode()
        ],
        "encode_stop": lambda: [b'{"op":"stop"}'],
        "check_reply": check_reply,
        "io_timeout_s": 0.5,
    }
    config.update(overrides)
    return ZmqBackend(**config)


def test_observations_commands_and_stop(peer: Peer) -> None:
    backend: ZmqBackend = make_backend(peer)
    backend.connect()
    try:
        deadline: float = time.monotonic() + 2
        while "q" not in backend.read():
            assert time.monotonic() < deadline
            time.sleep(0.005)
        before: Sample = backend.read()["q"]
        backend.write({"target": np.array([0.25], dtype=np.float32)})
        while backend.read()["q"].value[0] != 0.25:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        np.testing.assert_array_equal(before.value, [0])
        backend.stop()
        backend.stop()
    finally:
        backend.close()
    assert peer.commands == [{"op": "write", "value": 0.25}, {"op": "stop"}]


def test_peer_rejection_faults_session_and_requests_stop(peer: Peer) -> None:
    peer.reject = True
    backend: ZmqBackend = make_backend(peer)
    backend.connect()
    with pytest.raises(ValueError, match="rejected"):
        backend.write({"target": np.zeros(1, dtype=np.float32)})
    # close reports the original worker failure rather than hiding it.
    with pytest.raises(RuntimeError, match="worker failed"):
        backend.close()
    assert any(message["op"] == "stop" for message in peer.commands)


def test_timeout_has_no_automatic_command_replay(peer: Peer) -> None:
    peer.reply_delay = 0.15
    backend: ZmqBackend = make_backend(peer, io_timeout_s=0.04)
    backend.connect()
    with pytest.raises(TimeoutError):
        backend.write({"target": np.zeros(1, dtype=np.float32)})
    with pytest.raises(RuntimeError, match="worker failed"):
        backend.close()
    assert sum(message["op"] == "write" for message in peer.commands) <= 1
