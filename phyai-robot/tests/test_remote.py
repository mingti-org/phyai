"""Wire and lifecycle tests use synthetic data; no model or ROS installation."""

from __future__ import annotations

import time
import socket
from threading import Event, Thread
from contextlib import contextmanager

import numpy as np
import pytest

pytest.importorskip("zmq")
from phyai_robot import FeatureSpec
from phyai_robot.remote import PolicySpec, PolicyServer, RemotePolicy, RemotePolicyError
from phyai_robot.validation import freeze
from phyai_robot.remote._wire import pack, unpack, read_values, values_payload

INPUTS = {"state": FeatureSpec((2,), "float32")}
OUTPUTS = {"actions": FeatureSpec((3, 2), "float64")}
SPEC = PolicySpec(INPUTS, OUTPUTS, {"horizon": 3})
STATE = {"state": np.array([1.0, 2.0], dtype=np.float32)}


def endpoint():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{sock.getsockname()[1]}"


class Predictor:
    def __init__(self):
        self.resets = 0
        self.calls = 0
        self.seen = []
        self.gate = None

    def reset(self):
        self.resets += 1
        self.calls = 0

    def predict(self, inputs):
        self.calls += 1
        self.seen.append(inputs)
        if self.gate is not None:
            assert self.gate.wait(2)
        state = inputs["state"].astype(np.float64)
        return {"actions": np.stack([state + self.calls + i for i in range(3)])}


@contextmanager
def running(lease_timeout_s=30.0):
    predictor = Predictor()
    stop, ready = Event(), Event()
    url = endpoint()
    server = PolicyServer(
        policy=predictor,
        spec=SPEC,
        reset_run=predictor.reset,
        endpoint=url,
        lease_timeout_s=lease_timeout_s,
    )
    errors = []

    def serve():
        try:
            server.serve(stop_event=stop, ready_event=ready)
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=serve)
    thread.start()
    try:
        assert ready.wait(3)
        yield server, predictor, url
    finally:
        if predictor.gate is not None:
            predictor.gate.set()
        stop.set()
        thread.join(3)
        assert not thread.is_alive()
        assert not errors


@pytest.mark.parametrize(
    "value",
    [
        np.array(1.5),
        np.arange(12, dtype=">f8").reshape(3, 4)[:, ::2],
        np.zeros((0, 2), dtype=np.uint8),
        np.array([True, False]),
        np.array([-0.0, 1.0], dtype=np.float32),
    ],
)
def test_wire_preserves_bytes_dtype_shape(value):
    frames = pack({"key": 4}, [value])
    # Received frames are bytes; their arrays are read-only and need no copy.
    header, values = unpack([bytes(frame) for frame in frames])
    actual = values[0]
    assert actual.shape == value.shape
    assert actual.dtype.str == value.dtype.str
    assert actual.tobytes() == value.tobytes()
    assert not actual.flags.writeable
    assert freeze(actual) is actual
    assert header["key"] == 4


def test_pack_does_not_copy_contiguous_arrays():
    value = np.arange(6, dtype=np.float32)
    frame = pack({}, [value])[1]
    assert np.shares_memory(np.frombuffer(frame, dtype=np.float32), value)


@pytest.mark.parametrize("dtype", ["object", "complex128", "U2"])
def test_wire_rejects_non_numeric(dtype):
    with pytest.raises(ValueError):
        pack({}, [np.zeros(1, dtype=dtype)])


@pytest.mark.parametrize(
    "frames",
    [
        [b"[]"],
        [b'{"protocol":1,"arrays":[]}'],
        [b'{"protocol":true,"arrays":[]}'],
        [b'{"protocol":2,"arrays":[{"shape":[-1],"dtype":"f8"}]}', b""],
        [b'{"protocol":2,"arrays":[{"shape":[1],"dtype":"O"}]}', b"12345678"],
        [b'{"protocol":2,"arrays":[]}', b"extra"],
    ],
)
def test_wire_rejects_invalid_frames(frames):
    with pytest.raises(ValueError):
        unpack(frames)


def test_named_values_roundtrip_and_validation():
    meta, arrays = values_payload(STATE, INPUTS)
    header, values = unpack([bytes(f) for f in pack(meta, arrays)])
    np.testing.assert_array_equal(read_values(header, values, INPUTS)["state"], [1, 2])
    for bad in [
        {},
        {"state": np.array([1.0, np.nan], dtype=np.float32)},
        {"state": np.array([1.0, 2.0])},
        {"state": np.zeros(3, dtype=np.float32)},
    ]:
        with pytest.raises((ValueError, TypeError)):
            values_payload(bad, INPUTS)
    with pytest.raises(ValueError, match="field mismatch"):
        read_values({"keys": ["state", "state"]}, values * 2, INPUTS)


def test_spec_roundtrip_preserves_schema_details():
    spec = PolicySpec(
        {"q": FeatureSpec((2,), "float32", "rad", ("a", "b"), "base")},
        OUTPUTS,
        {"image_size": 224},
    )
    assert PolicySpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


def test_run_reset_warmup_exactly_once_and_worker_socket_ownership():
    with running() as (_, predictor, url):
        for run in range(3):
            client = RemotePolicy(url)
            assert client.info().metadata["horizon"] == 3
            client.begin_run()
            with pytest.raises(RuntimeError):
                client.predict(STATE)
            warm = client.warmup(STATE)
            with pytest.raises(RuntimeError):
                client.warmup(STATE)
            result = []
            worker = Thread(target=lambda: result.append(client.predict(STATE)))
            worker.start()
            worker.join(3)
            assert not worker.is_alive()
            np.testing.assert_array_equal(warm["actions"][0], [2, 3])
            np.testing.assert_array_equal(result[0]["actions"][0], [3, 4])
            assert predictor.calls == 2
            client.close()
            client.close()
        assert predictor.resets == 3


def test_client_validates_inputs_before_sending():
    with running() as (_, predictor, url):
        client = RemotePolicy(url)
        client.begin_run()
        with pytest.raises(TypeError):
            client.warmup({"state": np.zeros(2)})
        assert predictor.calls == 0
        client.close()


def test_busy_server_does_not_reset_rng():
    with running() as (_, predictor, url):
        owner = RemotePolicy(url)
        owner.begin_run()
        other = RemotePolicy(url)
        with pytest.raises(RemotePolicyError, match="active run"):
            other.begin_run()
        other.close()
        assert predictor.resets == 1
        owner.warmup(STATE)
        owner.close()


def test_abandoned_run_is_released_after_its_lease():
    with running(lease_timeout_s=0.2) as (server, predictor, url):
        crashed = RemotePolicy(url)
        crashed.begin_run()
        # The owner disappears without end(); a new client is refused at first.
        other = RemotePolicy(url)
        with pytest.raises(RemotePolicyError, match="active run"):
            other.begin_run()
        other.close()
        deadline = time.monotonic() + 2
        while server.run_id is not None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.run_id is None
        replacement = RemotePolicy(url)
        replacement.begin_run()
        replacement.warmup(STATE)
        assert predictor.resets == 2
        # The old owner can no longer use the server, and closing it is harmless.
        with pytest.raises(RemotePolicyError, match="run ID"):
            crashed.warmup(STATE)
        crashed.close()
        assert server.run_id is not None
        replacement.close()
        assert server.run_id is None


def test_regular_requests_keep_the_lease_alive():
    with running(lease_timeout_s=0.3) as (server, _, url):
        client = RemotePolicy(url)
        client.begin_run()
        client.warmup(STATE)
        for _ in range(6):
            time.sleep(0.1)
            client.predict(STATE)
        assert server.run_id is not None
        client.close()
        assert server.run_id is None


def test_info_from_another_client_does_not_extend_the_lease():
    server = PolicyServer(
        policy=Predictor(), spec=SPEC, reset_run=lambda: None, lease_timeout_s=1
    )
    server._dispatch(dict(operation="begin", request_id=1), [])
    server._lease_deadline = 0.0
    _, _, run_id = server._dispatch(dict(operation="info", request_id=1), [])
    assert run_id is None
    assert server.expire_lease()


def test_request_and_run_ids_reject_replay():
    with running() as (server, _, url):
        client = RemotePolicy(url)
        client.begin_run()
        client.warmup(STATE)
        with pytest.raises(ValueError, match="stale request"):
            server._dispatch(
                dict(operation="predict", request_id=1, run_id=client._run_id), []
            )
        with pytest.raises(ValueError, match="run ID"):
            server._dispatch(dict(operation="end", request_id=99, run_id="old"), [])
        client.close()


def test_timeout_is_bounded_and_never_retried():
    client = RemotePolicy(endpoint(), timeout_s=0.04)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        client.begin_run()
    assert time.monotonic() - start < 1
    with pytest.raises(RuntimeError, match="failed"):
        client.begin_run()
    assert client._request_id == 1
    client.close()


def test_client_reuses_one_socket_until_a_request_fails():
    with running() as (_, _, url):
        client = RemotePolicy(url)
        client.info()
        first = client._socket
        client.begin_run()
        client.warmup(STATE)
        client.predict(STATE)
        assert client._socket is first
        client.close()
        assert client._socket is None and client._context is None


def test_server_survives_malformed_message():
    import zmq

    with running() as (_, _, url), zmq.Context() as ctx:
        with ctx.socket(zmq.REQ) as sock:
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(url)
            sock.send_multipart([b"[]"])
            assert sock.poll(1000)
            assert not unpack(sock.recv_multipart())[0]["ok"]
        client = RemotePolicy(url)
        client.begin_run()
        client.warmup(STATE)
        client.close()


def test_close_while_worker_inflight_releases_only_after_prediction():
    with running() as (server, predictor, url):
        client = RemotePolicy(url)
        client.begin_run()
        client.warmup(STATE)
        predictor.gate = Event()
        result = []
        thread = Thread(target=lambda: result.append(client.predict(STATE)))
        thread.start()
        limit = time.monotonic() + 1
        while predictor.calls != 2 and time.monotonic() < limit:
            time.sleep(0.001)
        assert predictor.calls == 2
        client.close()
        assert server.run_id is not None
        predictor.gate.set()
        thread.join(3)
        assert len(result) == 1
        assert server.run_id is None


@pytest.mark.parametrize("field,value", [("request_id", 999), ("run_id", "stale")])
def test_client_rejects_mismatched_reply(field, value):
    import zmq

    urls = []
    ready, answered = Event(), Event()
    errors = []

    def fake_server():
        try:
            with zmq.Context() as context, context.socket(zmq.REP) as sock:
                sock.setsockopt(zmq.LINGER, 0)
                # Bind directly: a probed-then-released port can be taken first.
                port = sock.bind_to_random_port("tcp://127.0.0.1")
                urls.append(f"tcp://127.0.0.1:{port}")
                ready.set()
                assert sock.poll(5000)
                request, _ = unpack(sock.recv_multipart())
                response = dict(
                    ok=True,
                    spec=SPEC.to_dict(),
                    request_id=request["request_id"],
                    run_id=request["run_id"],
                )
                response[field] = value
                sock.send_multipart(pack(response))
                # LINGER=0 drops unsent replies on close; keep the socket open
                # until the client has its answer.
                assert answered.wait(5)
        except BaseException as error:
            errors.append(error)
            ready.set()

    thread = Thread(target=fake_server)
    thread.start()
    assert ready.wait(2) and urls, errors
    client = RemotePolicy(urls[0], timeout_s=5)
    try:
        with pytest.raises(RemotePolicyError, match="ID mismatch"):
            client.info()
    finally:
        answered.set()
        client.close()
        thread.join(6)
    assert not thread.is_alive() and not errors


def test_predict_failure_is_not_retried_and_run_must_be_released():
    with running() as (server, predictor, url):
        client = RemotePolicy(url)
        client.begin_run()
        client.warmup(STATE)
        calls = []

        def fail(request):
            calls.append(request)
            raise ValueError("injected model error")

        predictor.predict = fail
        with pytest.raises(RemotePolicyError, match="injected model error"):
            client.predict(STATE)
        with pytest.raises(RuntimeError, match="failed"):
            client.predict(STATE)
        assert len(calls) == 1 and server.run_id is not None
        client.close()
        assert server.run_id is None


def test_server_rejects_outputs_that_do_not_match_its_spec():
    with running() as (_, predictor, url):
        predictor.predict = lambda inputs: {"actions": np.zeros((3, 2), np.float32)}
        client = RemotePolicy(url)
        client.begin_run()
        with pytest.raises(RemotePolicyError, match="dtype"):
            client.warmup(STATE)
        client.close()
