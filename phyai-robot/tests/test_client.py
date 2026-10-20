"""The shared remote client lifecycle against a real in-process server."""

from __future__ import annotations

import socket
from threading import Event, Thread

import numpy as np
import pytest

pytest.importorskip("zmq")
from phyai_robot import (
    StateSlot,
    CommandSlot,
    FeatureSpec,
    StopAndRefill,
    VectorAdapter,
    CompositeRobot,
    DeploymentOptions,
)
from phyai_robot.remote import (
    PolicySpec,
    Embodiment,
    PolicyServer,
    RemotePolicy,
    run_remote,
)
from phyai_robot.backends.mock import MockBackend

OBSERVATIONS = {
    "cam": FeatureSpec((2, 2, 3), "uint8"),
    "q": FeatureSpec((2,), "float32"),
}
ACTIONS = {"target": FeatureSpec((2,), "float32")}
SPEC = PolicySpec(
    {"image.main": OBSERVATIONS["cam"], "state": FeatureSpec((2,), "float32")},
    {"actions": FeatureSpec((4, 2), "float32")},
    {"image_size": 2},
)


class Model:
    def __init__(self, events):
        self.events = events

    def predict(self, inputs):
        self.events.append("predict")
        return {"actions": np.repeat(inputs["state"][None] + 0.1, 4, axis=0)}


def serve(spec, events):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"tcp://127.0.0.1:{sock.getsockname()[1]}"
    server = PolicyServer(
        policy=Model(events),
        spec=spec,
        reset_run=lambda: events.append("reset"),
        endpoint=url,
    )
    stop, ready = Event(), Event()
    thread = Thread(
        target=server.serve, kwargs=dict(stop_event=stop, ready_event=ready)
    )
    thread.start()
    assert ready.wait(3)
    return server, url, stop, thread


def build_factory(events, backend_holder, state_dim=2):
    def build(spec):
        events.append(f"build:{spec.metadata['image_size']}")
        backend = MockBackend(
            initial_values={
                "cam": np.zeros((2, 2, 3), dtype=np.uint8),
                "q": np.zeros(2, dtype=np.float32),
            },
            action_to_observation={"target": "q"},
        )
        backend_holder.append(backend)
        robot = CompositeRobot(
            observation_schema=OBSERVATIONS, action_schema=ACTIONS, backends=[backend]
        )
        state = [StateSlot("q", index=tuple(range(state_dim)))]
        adapter = VectorAdapter(
            observation_schema=OBSERVATIONS,
            action_schema=ACTIONS,
            cameras={"main": "cam"},
            state=state,
            commands=[CommandSlot("target")],
        )

        def activate():
            events.append(f"activate:{backend.write_count}")

        return Embodiment(robot, adapter, activate)

    return build


def test_run_remote_orders_warmup_activation_and_cleanup():
    events, backends = [], []
    server, url, stop, thread = serve(SPEC, events)
    try:
        run_remote(
            RemotePolicy(url),
            build=build_factory(events, backends),
            scheduler=StopAndRefill(action_hz=50, execution_horizon=2),
            options=DeploymentOptions(control_hz=50, max_sample_age_s=1),
            max_steps=6,
            report=events.append,
        )
    finally:
        stop.set()
        thread.join(3)
    assert events[:4] == ["build:2", "reset", "predict", events[3]]
    assert events[3].startswith("Warmup took")
    assert "activate:0" in events
    assert events.index("activate:0") < events.index("predict", 3)
    backend = backends[0]
    assert backend.write_count == 6 and backend.stop_count == 1
    assert server.run_id is None


def test_adapter_and_model_mismatch_fails_before_reserving_the_server():
    events, backends = [], []
    spec = PolicySpec(
        {"image.main": OBSERVATIONS["cam"], "state": FeatureSpec((1,), "float32")},
        SPEC.outputs,
        SPEC.metadata,
    )
    server, url, stop, thread = serve(spec, events)
    try:
        with pytest.raises(ValueError, match="model expects"):
            run_remote(
                RemotePolicy(url),
                build=build_factory(events, backends),
                scheduler=StopAndRefill(action_hz=50),
                options=DeploymentOptions(control_hz=50),
            )
    finally:
        stop.set()
        thread.join(3)
    assert "reset" not in events and backends[0].write_count == 0
    assert server.run_id is None


def test_horizon_longer_than_the_model_fails_before_activation():
    events, backends = [], []
    server, url, stop, thread = serve(SPEC, events)
    try:
        with pytest.raises(ValueError, match="execution_horizon is 5"):
            run_remote(
                RemotePolicy(url),
                build=build_factory(events, backends),
                scheduler=StopAndRefill(action_hz=50, execution_horizon=5),
                options=DeploymentOptions(control_hz=50, max_sample_age_s=1),
            )
    finally:
        stop.set()
        thread.join(3)
    assert not any(event.startswith("activate") for event in events)
    assert backends[0].stop_count == 1
    assert server.run_id is None
