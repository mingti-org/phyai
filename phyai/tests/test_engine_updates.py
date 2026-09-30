"""Lifecycle tests for streamed Engine weight updates."""

from __future__ import annotations

import threading

import pytest
from phyai.engine import EngineCore, _WeightUpdateState
from phyai.server.lifecycle import EngineUnavailableError


class _Entry:
    def __init__(
        self,
        *,
        fail_update: bool = False,
        fail_finish: bool = False,
        fail_abort: bool = False,
    ) -> None:
        self.fail_update = fail_update
        self.fail_finish = fail_finish
        self.fail_abort = fail_abort
        self.events = []

    def begin_weight_update(self):
        self.events.append("begin")

    def update_weights(self, weights):
        self.events.append(("update", weights))
        if self.fail_update:
            raise RuntimeError("partial update failed")

    def finish_weight_update(self):
        self.events.append("finish")
        if self.fail_finish:
            raise RuntimeError("strict update failed")
        return "report"

    def abort_weight_update(self):
        self.events.append("abort")
        if self.fail_abort:
            raise RuntimeError("abort failed")

    def close(self):
        self.events.append("close")


class _BlockingEntry(_Entry):
    def __init__(self) -> None:
        super().__init__()
        self.update_started = threading.Event()
        self.release_update = threading.Event()

    def update_weights(self, weights):
        self.events.append(("update", weights))
        self.update_started.set()
        assert self.release_update.wait(timeout=1)


class _BlockingBeginEntry(_Entry):
    def __init__(self) -> None:
        super().__init__()
        self.begin_started = threading.Event()
        self.release_begin = threading.Event()

    def begin_weight_update(self):
        self.events.append("begin")
        self.begin_started.set()
        assert self.release_begin.wait(timeout=1)


def _engine(entry: _Entry) -> EngineCore:
    engine = object.__new__(EngineCore)
    engine.entry = entry
    engine._model_lock = threading.Lock()
    engine._weight_update_lock = threading.Lock()
    engine._weight_update_state = _WeightUpdateState.IDLE
    engine._version = 0
    engine._dumper = None
    engine._closed = False
    engine._replica_world_size = 1
    engine._owns_pg = False
    return engine


def test_version_commits_only_after_successful_update():
    engine = _engine(_Entry())

    engine.begin_weight_update()
    engine.update_weights({"weight": 1})
    assert engine.finish_weight_update(version=7) == "report"

    assert engine.version == 7
    engine.begin_weight_update()
    engine.abort_weight_update()


def test_finish_waits_for_inflight_update():
    entry = _BlockingEntry()
    engine = _engine(entry)
    engine.begin_weight_update()

    update_thread = threading.Thread(
        target=engine.update_weights, args=({"weight": 1},)
    )
    update_thread.start()
    assert entry.update_started.wait(timeout=1)

    finish_done = threading.Event()
    finish_thread = threading.Thread(
        target=lambda: (engine.finish_weight_update(), finish_done.set())
    )
    finish_thread.start()
    assert not finish_done.wait(timeout=0.05)

    entry.release_update.set()
    update_thread.join(timeout=1)
    finish_thread.join(timeout=1)

    assert finish_done.is_set()
    assert entry.events == ["begin", ("update", {"weight": 1}), "finish"]


def test_failed_finish_keeps_version_and_poison_engine():
    engine = _engine(_Entry(fail_finish=True))

    engine.begin_weight_update()
    engine.update_weights({"weight": 1})
    with pytest.raises(RuntimeError, match="strict update failed"):
        engine.finish_weight_update(version=7)

    assert engine.version == 0
    with pytest.raises(RuntimeError, match="partially applied"):
        engine.step(1)


def test_failed_update_poison_engine_after_abort():
    engine = _engine(_Entry(fail_update=True))

    engine.begin_weight_update()
    with pytest.raises(RuntimeError, match="partial update failed"):
        engine.update_weights({"weight": 1})
    engine.abort_weight_update()

    assert engine.version == 0
    with pytest.raises(RuntimeError, match="partially applied"):
        engine.step(1)


def test_abort_failure_still_poison_engine_and_release_session():
    engine = _engine(_Entry(fail_abort=True))

    engine.begin_weight_update()
    engine.update_weights({"weight": 1})
    with pytest.raises(RuntimeError, match="abort failed"):
        engine.abort_weight_update()

    with pytest.raises(RuntimeError, match="partially applied"):
        engine.step(1)
    assert engine._model_lock.acquire(blocking=False)


def test_close_serializes_with_weight_update_start(monkeypatch):
    monkeypatch.setattr("phyai.engine.reset_kernel_selector", lambda: None)
    monkeypatch.setattr("phyai.engine.release_global_fi_workspaces", lambda: None)
    entry = _BlockingBeginEntry()
    engine = _engine(entry)

    begin_thread = threading.Thread(target=engine.begin_weight_update)
    begin_thread.start()
    assert entry.begin_started.wait(timeout=1)

    close_done = threading.Event()
    close_thread = threading.Thread(target=lambda: (engine.close(), close_done.set()))
    close_thread.start()
    assert not close_done.wait(timeout=0.05)

    entry.release_begin.set()
    begin_thread.join(timeout=1)
    close_thread.join(timeout=1)

    assert close_done.is_set()
    assert entry.events == ["begin", "abort", "close"]
    with pytest.raises(EngineUnavailableError, match="closed EngineCore"):
        engine.begin_weight_update()
