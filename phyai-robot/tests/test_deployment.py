"""Use real mock feedback and controlled policy delays, without GPU or hardware."""

import math
import time
import threading
from typing import Any
from collections.abc import Mapping, Callable

import numpy as np
import pytest
import phyai_robot.deployment as deployment_module
from phyai_robot import (
    Robot,
    Action,
    Policy,
    Sample,
    ActionChunk,
    FeatureSpec,
    Observation,
    ActionSchema,
    EnginePolicy,
    CompositeRobot,
    RobotDeployment,
    DeploymentOptions,
)
from numpy.typing import NDArray
from phyai_robot.deployment import _interpolate, _prepare_chunk
from phyai_robot.backends.mock import MockBackend


def _sample_chunk(
    chunk: ActionChunk, schema: ActionSchema, cfg: DeploymentOptions
) -> tuple[Action, ...]:
    """Inspect lazy interpolation numerically, without wall-clock scheduling."""
    chunk = _prepare_chunk(chunk, schema, cfg)
    ratio = chunk.step_period_s * cfg.control_hz
    values = []
    for tick in range(math.ceil(len(chunk.actions) * ratio)):
        position = tick / ratio
        index = min(math.floor(position), len(chunk.actions) - 1)
        values.append(
            _interpolate(
                chunk.actions[index],
                chunk.actions[min(index + 1, len(chunk.actions) - 1)],
                position - index,
                cfg.interpolate_keys,
            )
        )
    return tuple(values)


class StepAdapter:
    def to_request(self, observation: Observation) -> NDArray[np.float32]:
        return observation.samples["q"].value.copy()

    def to_actions(
        self, result: NDArray[np.float32], observation: Observation
    ) -> ActionChunk:
        assert result.shape == observation.samples["q"].value.shape
        return ActionChunk(tuple({"target": result.copy()} for _ in range(4)), 0.05)


class StepPolicy:
    def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
        return request + np.float32(0.1)


def options(**overrides: Any) -> DeploymentOptions:
    # Tests override heterogeneous fields, including deliberately invalid values.
    values: dict[str, Any] = {
        "control_hz": 20,
        "max_sample_age_s": 1,
        "max_prediction_age_s": 2,
        "max_queued_actions": 8,
        "observation_timeout_s": 0.1,
    }
    values.update(overrides)
    return DeploymentOptions(**values)


def deploy(
    robot: Robot,
    policy: Policy[NDArray[np.float32], NDArray[np.float32]] | None = None,
    **overrides: Any,
) -> RobotDeployment[NDArray[np.float32], NDArray[np.float32]]:
    return RobotDeployment(
        robot=robot,
        policy=policy or StepPolicy(),
        adapter=StepAdapter(),
        options=options(**overrides),
    )


def test_mock_closed_loop_refills_and_exits(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend
    loop: RobotDeployment[NDArray[np.float32], NDArray[np.float32]] = deploy(robot)
    loop.run(max_steps=12)
    assert backend.write_count == 12
    assert backend.last_action is not None
    assert backend.last_action["target"][0] > 0.1
    assert backend.stop_count == 1
    with pytest.raises(RuntimeError, match="single-use"):
        loop.run()


def test_engine_policy_preserves_request_and_does_not_own_lifecycle() -> None:
    class Engine:
        def step(self, request: object) -> object:
            return request

    request: object = object()
    assert EnginePolicy[object, object](Engine()).predict(request) is request
    with pytest.raises(TypeError):
        EnginePolicy(object())


def test_stale_sensor_stops_before_any_command(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    original: Callable[[], Mapping[str, Sample]] = backend.read

    def stale_read() -> Mapping[str, Sample]:
        return {key: Sample(sample.value, 0) for key, sample in original().items()}

    monkeypatch.setattr(backend, "read", stale_read)
    with pytest.raises(TimeoutError, match="stale observation"):
        deploy(robot, max_sample_age_s=0.1).run(max_steps=1)
    assert backend.write_count == 0
    assert backend.stop_count == 1


def test_policy_error_is_propagated_and_robot_stops(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend

    class Broken:
        def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
            raise ValueError("prediction failed")

    with pytest.raises(ValueError, match="prediction failed"):
        deploy(robot, Broken()).run()
    assert backend.stop_count == 1
    assert backend.write_count == 0


def test_external_stop_discards_in_flight_prediction(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend
    entered, release = threading.Event(), threading.Event()

    class Waiting:
        def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
            entered.set()
            release.wait(2)
            return request

    loop: RobotDeployment[NDArray[np.float32], NDArray[np.float32]] = deploy(
        robot, Waiting()
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            loop.run()
        except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
            errors.append(error)

    actor: threading.Thread = threading.Thread(target=run)
    actor.start()
    assert entered.wait(2)
    loop.stop()
    release.set()
    actor.join(3)
    assert not actor.is_alive()
    assert not errors
    assert backend.write_count == 0
    assert backend.stop_count == 1


def test_inference_gap_is_idle_and_next_chunk_resumes(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    reads: list[tuple[int, int]] = []
    sends: list[float] = []
    original_read, original_write = backend.read, backend.write

    def read() -> Mapping[str, Sample]:
        reads.append((backend.write_count, threading.get_ident()))
        # Observation is deliberately much slower than the 50ms control period.
        time.sleep(0.08)
        return original_read()

    def write(action: Action) -> None:
        sends.append(time.monotonic())
        original_write(action)

    class SlowRefill:
        def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
            count = backend.write_count
            time.sleep(0.12)
            assert backend.write_count == count  # No replay while inference runs.
            return request + np.float32(0.1)

    monkeypatch.setattr(backend, "read", read)
    monkeypatch.setattr(backend, "write", write)
    deploy(robot, SlowRefill()).run(max_steps=9)
    assert backend.write_count == 9
    assert [count for count, _ in reads] == [0, 4, 8]
    assert all(thread != threading.get_ident() for _, thread in reads)
    assert sends[4] - sends[3] >= 0.24
    assert sends[8] - sends[7] >= 0.24
    assert backend.stop_count == 1


def test_expired_first_predictions_never_start_motion(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend

    class Slow:
        def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
            time.sleep(0.04)
            return request

    with pytest.raises(TimeoutError, match="expired before execution"):
        deploy(robot, Slow(), max_prediction_age_s=0.01, startup_timeout_s=0.15).run()
    assert backend.write_count == 0


def test_queued_commands_expire_while_waiting(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend
    with pytest.raises(TimeoutError, match="expired before execution"):
        deploy(robot, max_prediction_age_s=0.08).run()
    assert 0 < backend.write_count < 4
    assert backend.stop_count == 1


def test_resampling_continuous_and_discrete_fields() -> None:
    schema: ActionSchema = {
        "q": FeatureSpec((1,), "float32"),
        "mode": FeatureSpec((), "int32"),
    }
    chunk: ActionChunk = ActionChunk(
        (
            {"q": np.array([0], dtype=np.float32), "mode": np.array(0, dtype=np.int32)},
            {"q": np.array([2], dtype=np.float32), "mode": np.array(1, dtype=np.int32)},
        ),
        0.1,
    )
    values: tuple[Action, ...] = _sample_chunk(
        chunk, schema, options(control_hz=20, interpolate_keys={"q"})
    )
    np.testing.assert_allclose([a["q"][0] for a in values], [0, 1, 2, 2])
    np.testing.assert_array_equal([a["mode"] for a in values], [0, 0, 1, 1])
    chunk.actions[0]["q"][:] = 100
    assert values[0]["q"][0] == 0


def test_resampling_caps_prefix_and_validates_discarded_tail() -> None:
    schema: ActionSchema = {"q": FeatureSpec((1,), "float32")}
    good: Action = {"q": np.zeros(1, dtype=np.float32)}
    cfg: DeploymentOptions = options(max_queued_actions=3, execution_horizon=3)
    assert (
        len(_prepare_chunk(ActionChunk((good,) * 100, 0.1), schema, cfg).actions) == 3
    )
    assert len(_sample_chunk(ActionChunk((good,) * 100, 0.1), schema, cfg)) == 6
    with pytest.raises(ValueError, match="queue capacity"):
        _prepare_chunk(ActionChunk((good,) * 100, 0.1), schema, options())
    with pytest.raises(ValueError, match="NaN"):
        _sample_chunk(
            ActionChunk(
                (good,) * 100 + ({"q": np.array([np.nan], dtype=np.float32)},), 0.1
            ),
            schema,
            cfg,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"control_hz": float("nan")},
        {"control_hz": 0},
        {"max_control_lateness_s": 0},
        {"max_control_lateness_s": -0.01},
        {"max_control_lateness_s": True},
        {"max_control_lateness_s": float("nan")},
        {"max_control_lateness_s": float("inf")},
        {"action_hz": True},
        {"action_hz": float("inf")},
        {"action_hz": float("nan")},
        {"execution_horizon": 9, "max_queued_actions": 8},
        {"max_sample_age_s": -1},
        {"observation_timeout_s": 0},
        {"startup_timeout_s": float("inf")},
        {"action_hz": 0},
        {"shutdown_timeout_s": 0},
        {"max_queued_actions": True},
    ],
)
def test_invalid_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        options(**kwargs)


def test_slow_write_stops_instead_of_bursting_missed_ticks(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    original: Callable[[Action], None] = backend.write

    def slow_write(action: Action) -> None:
        original(action)
        time.sleep(0.12)

    monkeypatch.setattr(backend, "write", slow_write)
    with pytest.raises(TimeoutError, match="control deadline exceeded"):
        deploy(robot).run(max_steps=3)
    assert backend.write_count == 1
    assert backend.stop_count == 1


def test_stop_serializes_with_send_and_cleanup_stays_on_actor(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    entered, release = threading.Event(), threading.Event()
    stopped = threading.Event()
    threads: list[int] = []
    errors: list[BaseException] = []
    original_write, original_stop = backend.write, backend.stop

    def blocking_write(action: Action) -> None:
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        original_write(action)

    def record_stop() -> None:
        threads.append(threading.get_ident())
        original_stop()

    monkeypatch.setattr(backend, "write", blocking_write)
    monkeypatch.setattr(backend, "stop", record_stop)
    loop: RobotDeployment[NDArray[np.float32], NDArray[np.float32]] = deploy(robot)

    def run() -> None:
        try:
            loop.run()
        except BaseException as error:  # noqa: BLE001 - forward the actor's failure to pytest
            errors.append(error)

    actor: threading.Thread = threading.Thread(target=run)
    actor.start()
    assert entered.wait(2)

    def request_stop() -> None:
        loop.stop()
        stopped.set()

    stopper: threading.Thread = threading.Thread(target=request_stop)
    stopper.start()
    assert not stopped.wait(0.01)
    release.set()
    stopper.join(2)
    actor.join(2)
    assert stopped.is_set() and not actor.is_alive()
    assert not errors
    assert backend.write_count == 1
    assert set(threads) == {actor.ident}


def test_shutdown_timeout_stops_hardware_before_reporting_blocked_policy(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    class Blocked:
        def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
            entered.set()
            try:
                release.wait(2)
                return request
            finally:
                returned.set()

    loop: RobotDeployment[NDArray[np.float32], NDArray[np.float32]] = deploy(
        robot, Blocked(), shutdown_timeout_s=0.03
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            loop.run()
        except BaseException as error:  # noqa: BLE001 - forward the actor's failure to pytest
            errors.append(error)

    actor: threading.Thread = threading.Thread(target=run)
    actor.start()
    try:
        assert entered.wait(2)
        loop.stop()
        actor.join(1)
        assert not actor.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
        assert "policy is still running" in str(errors[0])
        assert backend.stop_count == 1
        assert backend.write_count == 0
    finally:
        release.set()
        assert returned.wait(2)


def test_stop_before_run_still_reports_cleanup_failure(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, _ = robot_and_backend
    loop: RobotDeployment[NDArray[np.float32], NDArray[np.float32]] = deploy(robot)
    loop.stop()

    def bad_close() -> None:
        raise OSError("cleanup failed")

    monkeypatch.setattr(robot, "close", bad_close)
    with pytest.raises(OSError, match="cleanup failed"):
        loop.run()


@pytest.mark.parametrize("horizon", [None, 1, 20, 50, 60])
def test_execution_horizon_precedes_25hz_to_200hz_interpolation(
    horizon: int | None,
) -> None:
    schema: ActionSchema = {"q": FeatureSpec((1,), "float64")}
    chunk = ActionChunk(
        tuple({"q": np.array([index], dtype=np.float64)} for index in range(50)),
        0.04,
    )
    cfg = options(
        control_hz=200,
        max_queued_actions=500,
        execution_horizon=horizon,
        interpolate_keys={"q"},
    )
    values = _sample_chunk(chunk, schema, cfg)
    retained = 50 if horizon is None else min(50, horizon)
    assert len(values) == retained * 8
    expected = np.minimum(np.arange(retained * 8) / 8, retained - 1)
    np.testing.assert_allclose([action["q"][0] for action in values], expected)
    # The discarded next action cannot influence the final retained interval.
    np.testing.assert_array_equal(
        [action["q"][0] for action in values[-8:]], retained - 1
    )
    assert not values[0]["q"].flags.writeable


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5, "20"])
def test_execution_horizon_requires_positive_integer(horizon: Any) -> None:
    with pytest.raises(ValueError, match="execution_horizon"):
        options(execution_horizon=horizon)


def test_execution_horizon_still_validates_discarded_tail() -> None:
    schema: ActionSchema = {"q": FeatureSpec((1,), "float32")}
    good = {"q": np.array([0], dtype=np.float32)}
    invalid = {"q": np.array([np.nan], dtype=np.float32)}
    with pytest.raises(ValueError, match="NaN"):
        _sample_chunk(
            ActionChunk((good, invalid), 0.04), schema, options(execution_horizon=1)
        )


@pytest.mark.parametrize(
    "action_hz, expected_count", [(None, 4), (10.0, 4), (5.0, 8), (8.0, 5)]
)
def test_action_rate_is_independent_of_control_rate(
    action_hz: float | None, expected_count: int
) -> None:
    schema = {"q": FeatureSpec((1,), "float32")}
    chunk = ActionChunk(
        tuple({"q": np.array([i], dtype=np.float32)} for i in range(2)), 0.1
    )
    cfg = options(action_hz=action_hz, interpolate_keys={"q"})
    values = _sample_chunk(chunk, schema, cfg)
    assert len(values) == expected_count
    assert values[0]["q"][0] == 0
    assert values[-1]["q"][0] == 1


def test_controller_consumes_source_queue_and_interpolates_lazily(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    original = backend.write
    sent: list[float] = []
    queue_sizes: list[int] = []

    class RampAdapter(StepAdapter):
        def to_actions(
            self, result: NDArray[np.float32], observation: Observation
        ) -> ActionChunk:
            return ActionChunk(
                tuple({"target": np.full_like(result, i)} for i in range(50)), 0.04
            )

    # Use the same 8:1 interpolation ratio as 25 -> 200Hz, but a relaxed clock
    # for CPU CI. Numeric tests above cover the exact 25/200Hz parameters.
    loop = RobotDeployment(
        robot=robot,
        policy=StepPolicy(),
        adapter=RampAdapter(),
        options=options(
            control_hz=40,
            action_hz=5,
            execution_horizon=2,
            max_queued_actions=2,
            interpolate_keys={"target"},
        ),
    )

    def write(action: Action) -> None:
        with loop._condition:
            queue_sizes.append(len(loop._queue))
        sent.append(float(action["target"][0]))
        original(action)

    monkeypatch.setattr(backend, "write", write)
    loop.run(max_steps=16)
    np.testing.assert_allclose(sent, np.minimum(np.arange(16) / 8, 1))
    assert queue_sizes == [2] * 8 + [1] * 8


def test_stop_waits_for_observation_before_cleanup(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    original = backend.read
    errors: list[BaseException] = []

    def read() -> Mapping[str, Sample]:
        entered.set()
        assert release.wait(2)
        assert backend.stop_count == 0
        return original()

    monkeypatch.setattr(backend, "read", read)
    loop = deploy(robot)

    def run() -> None:
        try:
            loop.run()
        except BaseException as error:  # noqa: BLE001 - report actor errors
            errors.append(error)

    actor = threading.Thread(target=run)
    actor.start()
    assert entered.wait(2)

    def stop() -> None:
        loop.stop()
        stopped.set()

    stopper = threading.Thread(target=stop)
    stopper.start()
    try:
        assert not stopped.wait(0.02)
        assert backend.stop_count == 0
    finally:
        release.set()
        stopper.join(2)
        actor.join(2)
    assert not actor.is_alive()
    assert not errors
    assert backend.write_count == 0
    assert backend.stop_count == 1


def test_missing_first_observation_times_out_without_sending(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    monkeypatch.setattr(backend, "read", lambda: {})
    with pytest.raises(TimeoutError, match="first observation/action"):
        deploy(robot, startup_timeout_s=0.04).run()
    assert backend.write_count == 0
    assert backend.stop_count == 1


def test_refill_failure_does_not_replay_previous_chunk(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend

    class BrokenRefill:
        def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
            if backend.write_count:
                raise ValueError("refill failed")
            return request

    with pytest.raises(ValueError, match="refill failed"):
        deploy(robot, BrokenRefill()).run()
    assert backend.write_count == 4
    assert backend.stop_count == 1


def test_refill_waits_for_fresh_observation_without_sending(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    original = backend.read
    stale_until = 0.0
    stale_reads = 0

    def read() -> Mapping[str, Sample]:
        nonlocal stale_until, stale_reads
        samples = original()
        if backend.write_count == 4:
            if not stale_until:
                stale_until = time.monotonic() + 0.06
            if time.monotonic() < stale_until:
                stale_reads += 1
                return {key: Sample(sample.value, 0) for key, sample in samples.items()}
        return samples

    monkeypatch.setattr(backend, "read", read)
    deploy(robot).run(max_steps=5)
    assert stale_reads > 1
    assert backend.write_count == 5
    assert backend.stop_count == 1


class ControlClock:
    """Inject a single delay on the second control tick without wall-clock sleep."""

    def __init__(self) -> None:
        self.now_ns = 0
        self.delays: dict[str, int] = {}
        self.calls: dict[str, int] = {}
        self.send_times: list[int] = []
        self.targets: list[float] = []

    def monotonic_ns(self) -> int:
        return self.now_ns

    def delay(self, stage: str, *, at_call: int = 2) -> None:
        self.calls[stage] = self.calls.get(stage, 0) + 1
        if self.calls[stage] == at_call:
            self.now_ns += self.delays.get(stage, 0)

    def wait(self, timeout: float) -> bool:
        self.now_ns += round(timeout * 1e9)
        self.delay("wake")
        return False

    def __enter__(self) -> None:
        self.delay("lock")

    def __exit__(self, *args: object) -> None:
        pass


@pytest.fixture
def clocked_control(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RobotDeployment, ControlClock, MockBackend]:
    robot, backend = robot_and_backend
    robot.connect()
    clock = ControlClock()
    original_write = backend.write

    def write(action: Action) -> None:
        clock.send_times.append(clock.now_ns)
        clock.targets.append(float(action["target"][0]))
        original_write(action)
        clock.delay("send", at_call=1)

    def interpolate(*args: Any) -> Action:
        clock.delay("prepare")
        return _interpolate(*args)

    loop = deploy(robot, control_hz=200, action_hz=25, interpolate_keys={"target"})
    loop._queue.extend(
        {"target": np.full(robot.action_schema["target"].shape, i, dtype=np.float32)}
        for i in range(3)
    )
    loop._source_period_s = 0.04
    loop._observed_at_ns = 0
    monkeypatch.setattr(deployment_module, "time", clock)
    monkeypatch.setattr(deployment_module, "_interpolate", interpolate)
    monkeypatch.setattr(loop._stop_event, "wait", clock.wait)
    monkeypatch.setattr(loop, "_io_lock", clock)
    monkeypatch.setattr(backend, "write", write)
    return loop, clock, backend


@pytest.mark.parametrize("delay_ns", [0, 5_000_000, 7_500_000, 10_000_000])
def test_control_allows_lateness_through_10ms_without_catch_up(
    clocked_control: tuple[RobotDeployment, ControlClock, MockBackend],
    delay_ns: int,
) -> None:
    loop, clock, _ = clocked_control
    clock.delays["wake"] = delay_ns
    sent = loop._execute_chunk(None)
    skipped = delay_ns // 5_000_000
    assert sent == 24 - skipped
    assert clock.send_times[1] == 5_000_000 + delay_ns
    assert clock.targets[1] == (1 + skipped) / 8
    # The source clock is unchanged and no immediate backlog is published.
    assert all(b > a for a, b in zip(clock.send_times, clock.send_times[1:]))
    np.testing.assert_allclose(
        clock.targets,
        [min((t // 5_000_000) / 8, 2) for t in clock.send_times],
    )
    assert clock.now_ns == 120_000_000
    assert not loop._queue


@pytest.mark.parametrize("stage", ["wake", "lock", "prepare"])
def test_control_rejects_more_than_10ms_before_sending(
    clocked_control: tuple[RobotDeployment, ControlClock, MockBackend],
    stage: str,
) -> None:
    loop, clock, backend = clocked_control
    clock.delays[stage] = 10_000_001
    with pytest.raises(TimeoutError, match="control deadline exceeded") as caught:
        loop._execute_chunk(None)
    message = str(caught.value)
    assert "lateness_ms=10.000001" in message
    assert "limit_ms=10.000000" in message
    assert "scheduled_tick=1" in message
    label = {"wake": "wake_lateness_ms", "lock": "io_wait_ms", "prepare": "prepare_ms"}
    assert f"{label[stage]}=10.000001" in message
    assert backend.write_count == 1


def test_control_reports_previous_send_duration(
    clocked_control: tuple[RobotDeployment, ControlClock, MockBackend],
) -> None:
    loop, clock, backend = clocked_control
    clock.delays["send"] = 15_000_001
    with pytest.raises(TimeoutError, match="previous_send_ms=15.000001"):
        loop._execute_chunk(None)
    assert backend.write_count == 1


@pytest.mark.parametrize("stage", ["lock", "prepare", "send"])
def test_tolerated_delays_do_not_replay_missed_ticks(
    clocked_control: tuple[RobotDeployment, ControlClock, MockBackend],
    stage: str,
) -> None:
    loop, clock, _ = clocked_control
    clock.delays[stage] = 12_500_000 if stage == "send" else 7_500_000
    assert loop._execute_chunk(None) == 23
    assert clock.send_times[:3] == [0, 12_500_000, 15_000_000]
    assert clock.targets[2] == 3 / 8
    assert clock.now_ns == 120_000_000


def test_steps_count_actual_sends_not_skipped_ticks(
    clocked_control: tuple[RobotDeployment, ControlClock, MockBackend],
) -> None:
    loop, clock, _ = clocked_control
    clock.delays["wake"] = 10_000_000
    assert loop._execute_chunk(3) == 3
    assert clock.send_times == [0, 15_000_000, 20_000_000]
    assert clock.targets == [0, 3 / 8, 4 / 8]


def test_lateness_tolerance_does_not_disable_prediction_expiry(
    clocked_control: tuple[RobotDeployment, ControlClock, MockBackend],
) -> None:
    loop, clock, backend = clocked_control
    clock.now_ns = 3_000_000_000
    with pytest.raises(TimeoutError, match="queued prediction expired"):
        loop._execute_chunk(None)
    assert backend.write_count == 0
