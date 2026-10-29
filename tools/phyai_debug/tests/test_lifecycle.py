"""Run the native lifecycle with a local fake policy server, never ROS."""

import socket
from threading import Event, Thread

import numpy as np
import pytest

pytest.importorskip("zmq")

from phyai_robot import (
    FeatureSpec,
    CompositeRobot,
    VectorAdapter,
    StateSlot,
    CommandSlot,
    StopAndRefill,
    DeploymentOptions,
)
from phyai_robot.backends.mock import MockBackend
from phyai_robot.remote import (
    RemotePolicy,
    PolicyServer,
    PolicySpec,
    Embodiment,
    run_remote,
)
from tools.phyai_debug.capture import (
    Session,
    TracedRobot,
    TracedAdapter,
    TracedPolicy,
    TracedScheduler,
)
from tools.phyai_debug.compat import Probes


class Sink:
    def __init__(self):
        self.rows = []

    def emit(self, kind, **fields):
        self.rows.append({"kind": kind, **fields})
        return len(self.rows)


def execute(tracing, fail=False):
    observations = {
        "cam": FeatureSpec((2, 2, 3), "uint8"),
        "q": FeatureSpec((2,), "float32"),
    }
    actions = {"target": FeatureSpec((2,), "float32")}
    spec = PolicySpec(
        {"image.main": observations["cam"], "state": FeatureSpec((2,), "float32")},
        {"actions": FeatureSpec((4, 2), "float32")},
        {"image_size": 2},
    )
    backend = MockBackend(
        initial_values={
            "cam": np.zeros((2, 2, 3), np.uint8),
            "q": np.zeros(2, np.float32),
        },
        action_to_observation={"target": "q"},
    )
    robot = CompositeRobot(
        observation_schema=observations, action_schema=actions, backends=[backend]
    )
    adapter = VectorAdapter(
        observation_schema=observations,
        action_schema=actions,
        cameras={"main": "cam"},
        state=[StateSlot("q")],
        commands=[CommandSlot("target")],
    )
    seen = []

    class Model:
        calls = 0

        def predict(self, request):
            self.calls += 1
            seen.append("predict")
            if fail and self.calls == 2:
                raise RuntimeError("synthetic inference failure")
            return {"actions": np.repeat((request["state"] + 0.1)[None], 4, axis=0)}

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{sock.getsockname()[1]}"
    server = PolicyServer(
        policy=Model(), spec=spec, reset_run=lambda: None, endpoint=endpoint
    )
    stop, ready = Event(), Event()
    thread = Thread(
        target=server.serve, kwargs={"stop_event": stop, "ready_event": ready}
    )
    thread.start()
    assert ready.wait(3)
    native_policy = RemotePolicy(endpoint)
    schedule = StopAndRefill(action_hz=25, execution_horizon=2)
    sink = Sink()
    session = Session(sink, 0.5)
    with Probes(session) as probes:
        if tracing:
            probes.policy(native_policy)
        policy = TracedPolicy(native_policy, session) if tracing else native_policy

        def build(spec):
            return Embodiment(
                TracedRobot(robot, session) if tracing else robot,
                TracedAdapter(adapter, session) if tracing else adapter,
                lambda: seen.append("activate"),
            )

        try:
            run_remote(
                policy,
                build=build,
                scheduler=TracedScheduler(schedule, session) if tracing else schedule,
                options=DeploymentOptions(
                    control_hz=50, max_sample_age_s=0.5, max_control_lateness_s=0.1
                ),
                max_steps=3,
            )
        except Exception:
            if not fail:
                raise
        finally:
            stop.set()
            thread.join(3)
    return backend, seen, sink.rows, server.run_id


def test_native_and_debug_lifecycle_are_equivalent():
    plain, plain_seen, _, plain_run = execute(False)
    traced, traced_seen, rows, traced_run = execute(True)
    assert plain_seen == traced_seen == ["predict", "activate", "predict"]
    assert plain.write_count == traced.write_count == 3
    assert plain.stop_count == traced.stop_count == 1
    assert plain_run is traced_run is None
    requests = [r for r in rows if r["kind"] == "request"]
    assert [r["warmup"] for r in requests] == [True, False]
    assert {r["request_id"] for r in rows if r["kind"] == "command"} == {2}
    rpc = [r for r in rows if r["kind"] == "rpc"]
    assert [r["operation"] for r in rpc] == [
        "info",
        "begin",
        "warmup",
        "predict",
        "end",
    ]
    assert len({r["rpc_request_id"] for r in rpc}) == len(rpc)


def test_native_failure_stops_before_any_command():
    backend, seen, rows, run_id = execute(True, fail=True)
    assert backend.write_count == 0 and backend.stop_count == 1
    assert run_id is None
    assert any(r["kind"] == "response" and r["status"] == "error" for r in rows)
