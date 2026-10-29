"""CPU capture, causality, persistence and HTTP checks; no hardware connections."""

import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen, Request

import numpy as np
from PIL import Image
import pytest

from tools.phyai_debug.storage import TraceRecorder
from tools.phyai_debug.viewer import TraceReader, make_server, safe_path
from tools.phyai_debug.assets import snapshot_robot
from tools.phyai_debug.capture import (
    Session,
    TracedRobot,
    TracedAdapter,
    TracedPolicy,
    TracedScheduler,
)
from tools.phyai_debug.compat import Probes, message_data, verify_contract
from phyai_robot import (
    FeatureSpec,
    CompositeRobot,
    VectorAdapter,
    StateSlot,
    CommandSlot,
    StopAndRefill,
)
from phyai_robot.backends.mock import MockBackend
from phyai_robot.robot import take_snapshot
from phyai_robot.types import ObservationNotReady


def test_writer_snapshots_arrays_pixels_and_packets(tmp_path):
    trace = TraceRecorder(tmp_path / "trace", {})
    value = np.array([1.0, 2.0], dtype=np.float32)
    pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    expected = pixels.copy()
    seq = trace.emit(
        "request", request_id=4, arrays={"state": value}, images={"head_left": pixels}
    )
    trace.emit("sensor_receive", blobs={"mosaic.h264": b"abc"})
    trace.emit("sensor_receive", blobs={"mosaic.h264": b"def"})
    value[:] = 99
    pixels[:] = 0
    result = trace.close()
    assert result["complete"] and result["blob_count"] == 2
    rows = TraceReader(trace.output).events()["events"]
    assert rows[0]["seq"] == seq and rows[0]["values"]["state"] == [1, 2]
    np.testing.assert_array_equal(
        np.asarray(Image.open(trace.output / rows[0]["images"]["head_left"])), expected
    )
    assert (trace.output / "video/mosaic.h264").read_bytes() == b"abcdef"
    assert rows[2]["blobs"]["mosaic.h264"]["offset"] == 3
    with pytest.raises(FileExistsError):
        TraceRecorder(trace.output, {})


def test_writer_death_never_throws_into_control(tmp_path):
    trace = TraceRecorder(tmp_path / "trace", {})
    trace.process.terminate()
    trace.process.join(2)
    trace.emit("command", arrays={"q": np.ones(7)})
    result = trace.close(grace_s=0.1)
    assert not result["complete"] and result["dropped_events"] == 1


def test_overflow_is_nonblocking_and_reported(tmp_path, monkeypatch):
    import queue

    trace = TraceRecorder(tmp_path / "trace", {})
    original = trace.events.put_nowait
    monkeypatch.setattr(
        trace.events, "put_nowait", lambda value: (_ for _ in ()).throw(queue.Full())
    )
    trace.emit("command", arrays={"q": np.zeros(7)})
    monkeypatch.setattr(trace.events, "put_nowait", original)
    result = trace.close()
    assert result["dropped_events"] == 1 and not result["complete"]


def test_trace_pagination_and_readonly_http(tmp_path):
    trace = TraceRecorder(tmp_path / "trace", {})
    for i in range(6):
        trace.emit("command", t_ns=trace.started_ns + i * 10**9, request_id=i // 2)
    trace.close()
    reader = TraceReader(trace.output)
    first = reader.events(limit=2)
    assert first["truncated"] and first["next_after"] == 2
    assert len(reader.events(after=first["next_after"])["events"]) == 4
    assert len(reader.events(start=2, end=3)["events"]) == 2
    with pytest.raises(ValueError):
        reader.events(start=float("nan"))
    server = make_server(trace.output, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert json.load(urlopen(base + "/api/manifest"))["complete"]
        for path in [
            "/data/../manifest.json",
            "/data/%2e%2e/manifest.json",
            "/data/trace.sqlite",
        ]:
            with pytest.raises(HTTPError):
                urlopen(base + path)
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/start", data=b"{}", method="POST"))
        assert error.value.code == 501
    finally:
        server.shutdown()
        thread.join(2)
        server.server_close()


def test_asset_snapshot_rejects_traversal_and_symlinks(tmp_path):
    source = tmp_path / "assets"
    source.mkdir()
    (source / "robot.urdf").write_text(
        '<robot name="demo"><link name="base"><visual><geometry><mesh filename="package://../outside.stl"/></geometry></visual></link></robot>'
    )
    with pytest.raises(ValueError):
        snapshot_robot(source, tmp_path / "trace")
    (source / "link").symlink_to(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        safe_path(source, "link/outside")


class Sink:
    def __init__(self):
        self.rows = []

    def emit(self, kind, **fields):
        self.rows.append({"kind": kind, **fields})
        return len(self.rows)


def make_robot():
    observation = {
        "cam": FeatureSpec((2, 2, 3), "uint8"),
        "q": FeatureSpec((2,), "float32"),
    }
    action = {"target": FeatureSpec((2,), "float32")}
    backend = MockBackend(
        initial_values={
            "cam": np.zeros((2, 2, 3), np.uint8),
            "q": np.array([0.1, 0.2], np.float32),
        },
        action_to_observation={"target": "q"},
    )
    robot = CompositeRobot(
        observation_schema=observation, action_schema=action, backends=[backend]
    )
    adapter = VectorAdapter(
        observation_schema=observation,
        action_schema=action,
        cameras={"main": "cam"},
        state=[StateSlot("q")],
        commands=[CommandSlot("target")],
    )
    return robot, adapter, backend


def test_wrappers_preserve_outputs_and_chunk_causality():
    sink = Sink()
    session = Session(sink, 0.5)
    robot, adapter, backend = make_robot()
    traced = TracedRobot(robot, session)
    mapping = TracedAdapter(adapter, session)
    native_schedule = StopAndRefill(action_hz=25, execution_horizon=2)
    schedule = TracedScheduler(native_schedule, session)
    schedule.bind(robot.action_schema)
    robot.connect()
    try:
        observed = traced.get_observation()
        request = mapping.to_request(
            observed, schedule.begin_inference(time.monotonic_ns())
        )
        np.testing.assert_array_equal(
            request["state"], adapter.to_request(observed, None)["state"]
        )
        result = {"actions": np.array([[1, 2], [3, 4], [5, 6]], np.float32)}
        chunk = mapping.to_actions(result, observed)
        start = time.monotonic_ns()
        schedule.add_chunk(chunk, None, start)
        first = schedule.action_at(start)
        traced.send_action(first)
        mid = schedule.action_at(start + 20_000_000)
        np.testing.assert_array_equal(mid["target"], [2, 3])
        traced.send_action(mid)
        assert schedule.action_at(start + 80_000_000) is None
        sent = [row for row in sink.rows if row["kind"] == "command"]
        assert len(sent) == 2 and {row["request_id"] for row in sent} == {1}
        assert sent[1]["interpolation_fraction"] == 0.5
        assert sent[1]["observation_ages_ms"]["q"] >= 0
        assert backend.write_count == 2
    finally:
        traced.close()


def test_freshness_probe_uses_original_receive_clock_and_restores():
    import phyai_robot.robot as module

    original = module.check_freshness
    sink = Sink()
    session = Session(sink, 0.001)
    robot, _, _ = make_robot()
    robot.connect()
    try:
        with Probes(session):
            traced = TracedRobot(robot, session)
            observed = traced.get_observation()
            now = max(s.received_at_ns for s in observed.samples.values()) + 2_000_000
            with pytest.raises(ObservationNotReady):
                module.check_freshness(observed, 0.001, now)
            row = sink.rows[-1]
            assert row["kind"] == "freshness" and row["status"] == "rejected"
            assert row["ages_ms"]["q"] == pytest.approx(2)
        assert module.check_freshness is original
    finally:
        robot.close()


def test_source_timestamp_absence_and_contract():
    arrays, stamp = message_data(SimpleNamespace(data=[1.0, 2.0]))
    assert stamp is None
    np.testing.assert_array_equal(arrays["data"], [1, 2])
    assert len(verify_contract()) == 6


def test_media_overflow_marks_missing_without_losing_numeric_event(
    tmp_path, monkeypatch
):
    import queue

    trace = TraceRecorder(tmp_path / "trace", {})
    original = trace.images.put_nowait
    monkeypatch.setattr(
        trace.images, "put_nowait", lambda value: (_ for _ in ()).throw(queue.Full())
    )
    trace.emit(
        "request",
        arrays={"state": np.ones(2)},
        images={"camera": np.zeros((2, 2, 3), np.uint8)},
    )
    monkeypatch.setattr(trace.images, "put_nowait", original)
    report = trace.close()
    row = TraceReader(trace.output).events()["events"][0]
    assert row["media_missing"] and row["values"]["state"] == [1, 1]
    assert (
        not row["images"] and report["dropped_images"] == 1 and not report["complete"]
    )


def test_nonfinite_array_is_preserved_as_diagnostic(tmp_path):
    trace = TraceRecorder(tmp_path / "trace", {})
    trace.emit(
        "invalid_prediction", arrays={"actions": np.array([float("nan"), float("inf")])}
    )
    trace.close()
    row = TraceReader(trace.output).events()["events"][0]
    assert row["values"]["actions"] == [{"nonfinite": "nan"}, {"nonfinite": "inf"}]


def test_probe_rejects_changed_implementation(monkeypatch):
    from phyai_robot.scheduling import StopAndRefill

    def changed(self, tick_ns):
        return None

    monkeypatch.setattr(StopAndRefill, "action_at", changed)
    with pytest.raises((RuntimeError, KeyError)):
        verify_contract()
