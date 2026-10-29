"""Transparent decorators around native robot, adapter, policy and scheduler seams."""

from __future__ import annotations

import itertools
import threading
import time
from collections import OrderedDict

import numpy as np


def signature(observation):
    return tuple(
        (key, sample.received_at_ns, id(sample.value))
        for key, sample in sorted(observation.samples.items())
    )


class Session:
    def __init__(self, sink, max_age_s):
        self.sink = sink
        self.max_age_s = max_age_s
        self.requests = itertools.count(1)
        self.commands = itertools.count(1)
        self.observations = OrderedDict()
        self.samples = OrderedDict()
        self.local = threading.local()
        self.active_chunk = None
        self.active_tick = None
        self.command = None
        self.warmed = False
        self.request_samples = {}

    def emit(self, kind, **fields):
        return self.sink.emit(kind, **fields)

    def sample_metadata(self, key, sample):
        return self.samples.get((key, sample.received_at_ns), {})

    def remember_samples(self, values, received_at_ns, metadata):
        for key in values:
            self.samples[(key, received_at_ns)] = metadata
        while len(self.samples) > 8192:
            self.samples.popitem(last=False)

    def fields(self, observation, now):
        return {
            key: {
                "received_at_ns": sample.received_at_ns,
                "age_ms": (now - sample.received_at_ns) / 1e6,
                **self.sample_metadata(key, sample),
            }
            for key, sample in observation.samples.items()
        }


class Delegate:
    def __init__(self, base, session):
        self.base, self.trace = base, session

    def __getattr__(self, name):
        return getattr(self.base, name)


class TracedRobot(Delegate):
    def get_observation(self):
        begin = time.monotonic_ns()
        try:
            observation = self.base.get_observation()
        except BaseException as error:
            self.trace.emit("observation_error", read_begin_ns=begin, error=repr(error))
            raise
        end = time.monotonic_ns()
        fields = self.trace.fields(observation, end)
        seq = self.trace.emit(
            "observation",
            t_ns=end,
            read_begin_ns=begin,
            read_end_ns=end,
            samples=fields,
            arrays={
                k: s.value for k, s in observation.samples.items() if s.value.ndim < 3
            },
        )
        self.trace.observations[signature(observation)] = {
            "observation_id": seq,
            "read_begin_ns": begin,
            "read_end_ns": end,
        }
        while len(self.trace.observations) > 32:
            self.trace.observations.popitem(last=False)
        return observation

    def send_action(self, action):
        begin = time.monotonic_ns()
        context = dict(self.trace.active_tick or {})
        context.update(command_id=next(self.trace.commands), send_begin_ns=begin)
        context["observation_ages_ms"] = {
            key: (begin - received) / 1e6
            for key, received in self.trace.request_samples.get(
                context.get("request_id"), {}
            ).items()
        }
        self.trace.command = context
        try:
            result = self.base.send_action(action)
        except BaseException as error:
            self.trace.emit(
                "command", arrays=action, status="error", error=repr(error), **context
            )
            raise
        else:
            end = time.monotonic_ns()
            self.trace.emit(
                "command",
                t_ns=begin,
                arrays=action,
                status="success",
                send_end_ns=end,
                duration_ms=(end - begin) / 1e6,
                **context,
            )
            return result
        finally:
            self.trace.command = None

    def stop(self):
        self.trace.emit("event", name="robot_stop_begin")
        try:
            return self.base.stop()
        finally:
            self.trace.emit("event", name="robot_stop_return")

    def close(self):
        try:
            return self.base.close()
        finally:
            self.trace.emit("event", name="robot_close_return")


class TracedAdapter(Delegate):
    def to_request(self, observation, context):
        request = self.base.to_request(observation, context)
        now = time.monotonic_ns()
        request_id = next(self.trace.requests)
        self.trace.local.request_id = request_id
        self.trace.local.observation = observation
        self.trace.local.request = request
        self.trace.request_samples[request_id] = {
            key: sample.received_at_ns for key, sample in observation.samples.items()
        }
        while len(self.trace.request_samples) > 32:
            self.trace.request_samples.pop(next(iter(self.trace.request_samples)))
        samples = self.trace.fields(observation, now)
        self.trace.emit(
            "request",
            t_ns=now,
            request_id=request_id,
            warmup=not self.trace.warmed,
            arrays={
                key: value
                for key, value in request.items()
                if not key.startswith("image.")
            },
            images={
                key.removeprefix("image."): value
                for key, value in request.items()
                if key.startswith("image.")
            },
            samples=samples,
            ages_ms={key: value["age_ms"] for key, value in samples.items()},
            max_age_ms=self.trace.max_age_s * 1000,
            **self.trace.observations.get(signature(observation), {}),
        )
        return request

    def to_actions(self, result, observation):
        request_id = self.trace.local.request_id
        try:
            chunk = self.base.to_actions(result, observation)
        except BaseException as error:
            self.trace.emit(
                "event",
                name="action_validation_error",
                request_id=request_id,
                error=repr(error),
            )
            raise
        self.trace.emit(
            "adapted_actions",
            request_id=request_id,
            arrays={
                key: np.stack([action[key] for action in chunk.actions])
                for key in chunk.actions[0]
            },
        )
        return chunk


class TracedPolicy(Delegate):
    def info(self):
        spec = self.base.info()
        self.trace.emit("event", name="model_spec", spec=spec.to_dict())
        return spec

    def warmup(self, request):
        return self._predict(request, warmup=True)

    def predict(self, request):
        return self._predict(request, warmup=False)

    def _predict(self, request, *, warmup):
        begin = time.monotonic_ns()
        request_id = self.trace.local.request_id
        try:
            result = self.base.warmup(request) if warmup else self.base.predict(request)
        except BaseException as error:
            self.trace.emit(
                "response",
                request_id=request_id,
                status="error",
                error=repr(error),
                duration_ms=(time.monotonic_ns() - begin) / 1e6,
                warmup=warmup,
            )
            raise
        self.trace.emit(
            "response",
            request_id=request_id,
            arrays=result,
            status="success",
            duration_ms=(time.monotonic_ns() - begin) / 1e6,
            warmup=warmup,
        )
        self.trace.warmed = True
        return result


class TracedScheduler(Delegate):
    def __init__(self, base, session):
        super().__init__(base, session)
        self.start_ns = None
        self.chunk = None

    def add_chunk(self, chunk, context, now_ns):
        result = self.base.add_chunk(chunk, context, now_ns)
        request_id = self.trace.local.request_id
        self.chunk = {
            "chunk_id": request_id,
            "request_id": request_id,
            "execution_horizon": min(
                self.base.execution_horizon or len(chunk.actions), len(chunk.actions)
            ),
            "action_hz": self.base.action_hz,
            "horizon": len(chunk.actions),
        }
        self.start_ns = None
        self.trace.emit("chunk", t_ns=now_ns, phase="accepted", **self.chunk)
        return result

    def action_at(self, tick_ns):
        action = self.base.action_at(tick_ns)
        if action is None:
            if self.start_ns is not None:
                self.trace.emit("chunk", t_ns=tick_ns, phase="finished", **self.chunk)
            self.start_ns = None
            self.trace.active_tick = None
            return None
        if self.start_ns is None:
            self.start_ns = tick_ns
            self.trace.emit(
                "chunk", t_ns=tick_ns, phase="started", start_ns=tick_ns, **self.chunk
            )
        position = (tick_ns - self.start_ns) * self.base.action_hz / 1e9
        self.trace.active_tick = {
            **self.chunk,
            "tick_ns": tick_ns,
            "chunk_start_ns": self.start_ns,
            "action_index": int(position),
            "interpolation_fraction": position % 1,
        }
        return action
