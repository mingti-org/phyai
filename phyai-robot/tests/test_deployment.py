"""Use real mock feedback and controlled policy delays, without GPU or hardware."""

import time
import threading
from typing import Any
from collections.abc import Mapping, Callable

import numpy as np
import pytest
import phyai_robot.scheduling as scheduling_module
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
    StopAndRefill,
    CompositeRobot,
    RobotDeployment,
    InferenceContext,
    DeploymentOptions,
)
from numpy.typing import NDArray
from phyai_robot.validation import freeze_values
from phyai_robot.backends.mock import MockBackend


def sample_schedule(
    chunk: ActionChunk,
    schema: ActionSchema,
    *,
    control_hz: float,
    **scheduler_options: Any,
) -> list[Action]:
    """Inspect a scheduler numerically, without wall-clock scheduling."""
    scheduler = StopAndRefill(**scheduler_options)
    scheduler.bind(schema)
    chunk = ActionChunk(tuple(freeze_values(action) for action in chunk.actions))
    scheduler.add_chunk(chunk, InferenceContext(), 0)
    values = []
    tick = 0
    while (action := scheduler.action_at(round(tick * 1e9 / control_hz))) is not None:
        values.append(action)
        tick += 1
    assert not scheduler.is_active() and scheduler.wants_inference()
    return values


class StepAdapter:
    def to_request(
        self, observation: Observation, context: InferenceContext
    ) -> NDArray[np.float32]:
        assert context == InferenceContext()
        return observation.samples["q"].value.copy()

    def to_actions(
        self, result: NDArray[np.float32], observation: Observation
    ) -> ActionChunk:
        assert result.shape == observation.samples["q"].value.shape
        return ActionChunk(tuple({"target": result.copy()} for _ in range(4)))


class StepPolicy:
    def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
        return request + np.float32(0.1)


def options(**overrides: Any) -> DeploymentOptions:
    values: dict[str, Any] = {
        "control_hz": 20,
        "max_sample_age_s": 1,
        "observation_timeout_s": 0.1,
    }
    values.update(overrides)
    return DeploymentOptions(**values)


def deploy(
    robot: Robot,
    policy: Policy[NDArray[np.float32], NDArray[np.float32]] | None = None,
    *,
    action_hz: float = 20,
    execution_horizon: int | None = None,
    interpolate_keys: Any = None,
    adapter: Any = None,
    **overrides: Any,
) -> RobotDeployment[NDArray[np.float32], NDArray[np.float32]]:
    return RobotDeployment(
        robot=robot,
        policy=policy or StepPolicy(),
        adapter=adapter or StepAdapter(),
        scheduler=StopAndRefill(
            action_hz=action_hz,
            execution_horizon=execution_horizon,
            interpolate_keys=interpolate_keys,
        ),
        options=options(**overrides),
    )


def test_mock_closed_loop_refills_and_exits(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend
    loop = deploy(robot)
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
    with pytest.raises(TimeoutError, match="stale sensor samples"):
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

    loop = deploy(robot, Waiting())
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


def test_resampling_continuous_and_discrete_fields() -> None:
    schema: ActionSchema = {
        "q": FeatureSpec((1,), "float32"),
        "mode": FeatureSpec((), "int32"),
    }
    chunk = ActionChunk(
        (
            {"q": np.array([0], dtype=np.float32), "mode": np.array(0, dtype=np.int32)},
            {"q": np.array([2], dtype=np.float32), "mode": np.array(1, dtype=np.int32)},
        )
    )
    values = sample_schedule(chunk, schema, control_hz=20, action_hz=10)
    np.testing.assert_allclose([a["q"][0] for a in values], [0, 1, 2, 2])
    np.testing.assert_array_equal([a["mode"] for a in values], [0, 0, 1, 1])
    assert not values[1]["q"].flags.writeable


def test_explicit_interpolation_keys_are_checked_against_the_schema() -> None:
    schema: ActionSchema = {
        "q": FeatureSpec((1,), "float32"),
        "mode": FeatureSpec((), "int32"),
    }
    with pytest.raises(ValueError, match="unknown"):
        StopAndRefill(action_hz=10, interpolate_keys={"x"}).bind(schema)
    with pytest.raises(ValueError, match="floating"):
        StopAndRefill(action_hz=10, interpolate_keys={"mode"}).bind(schema)
    chunk = ActionChunk(
        (
            {"q": np.array([0], dtype=np.float32), "mode": np.array(0, dtype=np.int32)},
            {"q": np.array([2], dtype=np.float32), "mode": np.array(1, dtype=np.int32)},
        )
    )
    held = sample_schedule(
        chunk, schema, control_hz=20, action_hz=10, interpolate_keys=()
    )
    np.testing.assert_allclose([a["q"][0] for a in held], [0, 0, 2, 2])


def test_prepare_chunk_validates_and_detaches_every_target(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, _ = robot_and_backend
    loop = deploy(robot, execution_horizon=3)
    good = {"target": np.zeros(2, dtype=np.float32)}
    with pytest.raises(ValueError, match="NaN"):
        loop.prepare_chunk(
            ActionChunk((good,) * 5 + ({"target": np.full(2, np.nan, np.float32)},))
        )
    with pytest.raises(ValueError, match="execution_horizon is 3"):
        loop.prepare_chunk(ActionChunk((good,) * 2))
    with pytest.raises(ValueError, match="nonempty"):
        loop.prepare_chunk(ActionChunk(()))
    prepared = loop.prepare_chunk(ActionChunk((good,) * 3))
    good["target"][:] = 5
    assert prepared.actions[0]["target"][0] == 0


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
        {"max_sample_age_s": -1},
        {"observation_timeout_s": 0},
        {"shutdown_timeout_s": 0},
    ],
)
def test_invalid_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        options(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_hz": True},
        {"action_hz": float("inf")},
        {"action_hz": float("nan")},
        {"action_hz": 0},
        {"action_hz": 10, "execution_horizon": 0},
        {"action_hz": 10, "execution_horizon": -1},
        {"action_hz": 10, "execution_horizon": True},
        {"action_hz": 10, "execution_horizon": 1.5},
        {"action_hz": 10, "execution_horizon": "20"},
    ],
)
def test_invalid_schedule(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        StopAndRefill(**kwargs)


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
    loop = deploy(robot)

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

    loop = deploy(robot, Blocked(), shutdown_timeout_s=0.03)
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
    loop = deploy(robot)
    loop.stop()

    def bad_close() -> None:
        raise OSError("cleanup failed")

    monkeypatch.setattr(robot, "close", bad_close)
    with pytest.raises(OSError, match="cleanup failed"):
        loop.run()


@pytest.mark.parametrize("horizon", [None, 1, 20, 50])
def test_execution_horizon_precedes_25hz_to_200hz_interpolation(
    horizon: int | None,
) -> None:
    schema: ActionSchema = {"q": FeatureSpec((1,), "float64")}
    chunk = ActionChunk(
        tuple({"q": np.array([index], dtype=np.float64)} for index in range(50))
    )
    values = sample_schedule(
        chunk, schema, control_hz=200, action_hz=25, execution_horizon=horizon
    )
    retained = 50 if horizon is None else horizon
    assert len(values) == retained * 8
    expected = np.minimum(np.arange(retained * 8) / 8, retained - 1)
    np.testing.assert_allclose([action["q"][0] for action in values], expected)
    # The discarded next action cannot influence the final retained interval.
    np.testing.assert_array_equal(
        [action["q"][0] for action in values[-8:]], retained - 1
    )
    assert not values[0]["q"].flags.writeable


@pytest.mark.parametrize("action_hz, expected_count", [(10.0, 4), (5.0, 8), (8.0, 5)])
def test_action_rate_is_independent_of_control_rate(
    action_hz: float, expected_count: int
) -> None:
    schema = {"q": FeatureSpec((1,), "float32")}
    chunk = ActionChunk(tuple({"q": np.array([i], dtype=np.float32)} for i in range(2)))
    values = sample_schedule(chunk, schema, control_hz=20, action_hz=action_hz)
    assert len(values) == expected_count
    assert values[0]["q"][0] == 0
    assert values[-1]["q"][0] == 1


def test_controller_interpolates_the_executed_prefix(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    original = backend.write
    sent: list[float] = []

    class RampAdapter(StepAdapter):
        def to_actions(
            self, result: NDArray[np.float32], observation: Observation
        ) -> ActionChunk:
            return ActionChunk(
                tuple({"target": np.full_like(result, i)} for i in range(50))
            )

    # Use the same 8:1 interpolation ratio as 25 -> 200Hz, but a relaxed clock
    # for CPU CI. Numeric tests above cover the exact 25/200Hz parameters.
    loop = deploy(
        robot,
        adapter=RampAdapter(),
        control_hz=40,
        action_hz=5,
        execution_horizon=2,
    )

    def write(action: Action) -> None:
        sent.append(float(action["target"][0]))
        original(action)

    monkeypatch.setattr(backend, "write", write)
    loop.run(max_steps=16)
    np.testing.assert_allclose(sent, np.minimum(np.arange(16) / 8, 1))


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
    with pytest.raises(TimeoutError, match="complete, fresh observation"):
        deploy(robot, observation_timeout_s=0.04).run()
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


class OverlappingScheduler(StopAndRefill):
    """RTC-shaped test double: predict while the last two targets execute."""

    def __init__(self) -> None:
        super().__init__(action_hz=20)
        self.contexts: list[InferenceContext] = []
        self._queue: list[Action] = []
        self._requested = False

    def wants_inference(self) -> bool:
        return not self._requested and len(self._queue) <= 2

    def begin_inference(self, now_ns: int) -> InferenceContext:
        self._requested = True
        context = InferenceContext(pending=tuple(self._queue))
        self.contexts.append(context)
        return context

    def add_chunk(
        self, chunk: ActionChunk, context: InferenceContext, now_ns: int
    ) -> None:
        self._requested = False
        self._queue.extend(chunk.actions)

    def is_active(self) -> bool:
        return bool(self._queue)

    def action_at(self, tick_ns: int) -> Action | None:
        return self._queue.pop(0) if self._queue else None


def test_custom_scheduler_can_predict_while_executing(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    scheduler = OverlappingScheduler()
    writes_during_predict: list[int] = []

    class Adapter(StepAdapter):
        def to_request(self, observation, context):
            return observation.samples["q"].value.copy()

    class Policy:
        def predict(self, request):
            count = backend.write_count
            time.sleep(0.06)
            writes_during_predict.append(backend.write_count - count)
            return request + np.float32(0.1)

    loop = RobotDeployment(
        robot=robot,
        policy=Policy(),
        adapter=Adapter(),
        scheduler=scheduler,
        options=options(),
    )
    loop.run(max_steps=10)
    assert backend.write_count == 10
    assert scheduler.contexts[0].pending == ()
    assert len(scheduler.contexts[1].pending) in (1, 2)
    assert any(count > 0 for count in writes_during_predict[1:])


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
    original_interpolate = scheduling_module.interpolate

    def write(action: Action) -> None:
        clock.send_times.append(clock.now_ns)
        clock.targets.append(float(action["target"][0]))
        original_write(action)
        clock.delay("send", at_call=1)

    def interpolate(*args: Any) -> Action:
        clock.delay("prepare")
        return original_interpolate(*args)

    loop = deploy(robot, control_hz=200, action_hz=25)
    loop.scheduler.add_chunk(
        ActionChunk(
            tuple(
                {"target": np.full(robot.action_schema["target"].shape, i, np.float32)}
                for i in range(3)
            )
        ),
        InferenceContext(),
        0,
    )
    monkeypatch.setattr(deployment_module, "time", clock)
    monkeypatch.setattr(scheduling_module, "interpolate", interpolate)
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
    sent = loop._execute(None)
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
    assert not loop.scheduler.is_active()


@pytest.mark.parametrize("stage", ["wake", "lock", "prepare"])
def test_control_rejects_more_than_10ms_before_sending(
    clocked_control: tuple[RobotDeployment, ControlClock, MockBackend],
    stage: str,
) -> None:
    loop, clock, backend = clocked_control
    clock.delays[stage] = 10_000_001
    with pytest.raises(TimeoutError, match="control deadline exceeded") as caught:
        loop._execute(None)
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
        loop._execute(None)
    assert backend.write_count == 1


@pytest.mark.parametrize("stage", ["lock", "prepare", "send"])
def test_tolerated_delays_do_not_replay_missed_ticks(
    clocked_control: tuple[RobotDeployment, ControlClock, MockBackend],
    stage: str,
) -> None:
    loop, clock, _ = clocked_control
    clock.delays[stage] = 12_500_000 if stage == "send" else 7_500_000
    assert loop._execute(None) == 23
    assert clock.send_times[:3] == [0, 12_500_000, 15_000_000]
    assert clock.targets[2] == 3 / 8
    assert clock.now_ns == 120_000_000


def test_steps_count_actual_sends_not_skipped_ticks(
    clocked_control: tuple[RobotDeployment, ControlClock, MockBackend],
) -> None:
    loop, clock, _ = clocked_control
    clock.delays["wake"] = 10_000_000
    assert loop._execute(3) == 3
    assert clock.send_times == [0, 15_000_000, 20_000_000]
    assert clock.targets == [0, 3 / 8, 4 / 8]
