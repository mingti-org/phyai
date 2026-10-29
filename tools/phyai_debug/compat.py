"""Scoped probes for information that public deployment interfaces discard."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import time

import numpy as np

from .capture import signature


def message_data(message):
    arrays = {}
    for name in ("arm_positions", "arm_velocities", "arm_efforts", "positions", "data"):
        value = getattr(message, name, None)
        if value is not None and not isinstance(value, (str, bytes, bytearray)):
            array = np.asarray(value)
            if array.dtype.kind in "biuf" and array.size < 4096:
                arrays[name] = np.atleast_1d(array)
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    source = (
        None
        if not hasattr(stamp, "sec")
        else int(stamp.sec) * 10**9 + int(stamp.nanosec)
    )
    return arrays, source


def verify_contract():
    from phyai_robot.backends.ros2.backend import Ros2Backend
    from phyai_robot.remote.policy import RemotePolicy
    from phyai_robot.scheduling import StopAndRefill
    from phyai_robot.validation import check_freshness

    expected = [
        (Ros2Backend._open, ("self",)),
        (Ros2Backend._cache, ("self", "values", "received_at_ns")),
        (Ros2Backend._write_device, ("self", "action", "deadline")),
        (RemotePolicy._rpc, ("self", "operation", "payload", "arrays", "timeout_s")),
        (StopAndRefill.action_at, ("self", "tick_ns")),
        (check_freshness, ("observation", "max_age_s", "now_ns")),
    ]
    result = {}
    reviewed = {
        "Ros2Backend._open": "1550b49f9ca386caa41ef30012faa64188aa038c1ef14fddc45ebdc529eadf6a",
        "_ThreadedBackend._cache": "c06a8cfb3cda53dd49a39f98edb5214b002f1fb1733ab1b34852bb8f4b805c57",
        "Ros2Backend._write_device": "5057f5bb9f39c80c5115cf48e959c7c9c985af1732895f9ebf5537cb043df2b9",
        "RemotePolicy._rpc": "4dfc8d57de653864fd9a7484ae733846c7f41623b2f15594202934b03c09c91d",
        "StopAndRefill.action_at": "340ef48e599ced9dadb824feaccfdfbb86450eef0e697c4f2b5a4232842b3b2f",
        "check_freshness": "af36562a34fb09ccc8a85006271178d6afb9f8deec5459b01f8114e7f2888bd1",
    }
    for function, parameters in expected:
        if tuple(inspect.signature(function).parameters) != parameters:
            raise RuntimeError(f"Unsupported debug interface: {function.__qualname__}")
        result[function.__qualname__] = hashlib.sha256(
            inspect.getsource(function).encode()
        ).hexdigest()
        if result[function.__qualname__] != reviewed.get(function.__qualname__):
            raise RuntimeError(
                f"Review debug probes before using changed implementation: {function.__qualname__}"
            )
    return result


class Probes:
    def __init__(self, trace, *, raw_video=True):
        self.trace, self.raw_video = trace, raw_video
        self.restore = []

    def replace(self, obj, name, value):
        original = getattr(obj, name)
        self.restore.append((obj, name, original))
        setattr(obj, name, value)

    def __enter__(self):
        import phyai_robot.robot as robot_module

        original = robot_module.check_freshness

        def freshness(observation, max_age_s, now_ns=None):
            owned = signature(observation) in self.trace.observations
            if not owned:
                return original(observation, max_age_s, now_ns)
            now = time.monotonic_ns() if now_ns is None else now_ns
            status = "passed"
            try:
                return original(observation, max_age_s, now)
            except BaseException:
                status = "rejected"
                raise
            finally:
                self.trace.observations[signature(observation)].update(
                    freshness_checked_at_ns=now, freshness_status=status
                )
                self.trace.emit(
                    "freshness",
                    t_ns=now,
                    status=status,
                    ages_ms={
                        k: (now - s.received_at_ns) / 1e6
                        for k, s in observation.samples.items()
                    },
                    max_age_ms=max_age_s * 1000,
                    **self.trace.observations[signature(observation)],
                )

        self.replace(robot_module, "check_freshness", freshness)
        return self

    def policy(self, policy):
        original = policy._rpc

        def rpc(operation, payload=None, arrays=(), *, timeout_s=None):
            fields = {
                "operation": operation,
                "rpc_request_id": policy._request_id + 1,
                "run_id": policy._run_id,
            }
            if operation in ("warmup", "predict"):
                fields["request_id"] = getattr(self.trace.local, "request_id", None)
            begin = time.monotonic_ns()
            try:
                reply = original(operation, payload, arrays, timeout_s=timeout_s)
            except BaseException as error:
                self.trace.emit(
                    "rpc",
                    **fields,
                    status="error",
                    error=repr(error),
                    duration_ms=(time.monotonic_ns() - begin) / 1e6,
                )
                raise
            self.trace.emit(
                "rpc",
                **fields,
                status="success",
                duration_ms=(time.monotonic_ns() - begin) / 1e6,
            )
            return reply

        self.replace(policy, "_rpc", rpc)

    def backend(self, backend):
        if backend._thread is not None:
            raise RuntimeError("Attach debug probes before backend.connect()")
        metadata = {}
        original_cache = backend._cache

        def cache(values, received_at_ns):
            details = {**metadata, "cache_enter_at_ns": time.monotonic_ns()}
            self.trace.remember_samples(values, received_at_ns, details)
            result = original_cache(values, received_at_ns)
            details["cache_ready_at_ns"] = time.monotonic_ns()
            self.trace.emit(
                "feedback"
                if all(v.ndim < 3 for v in values.values())
                else "image_sample",
                t_ns=received_at_ns,
                arrays={k: v for k, v in values.items() if v.ndim < 3},
                fields=list(values),
                received_at_ns=received_at_ns,
                **details,
            )
            return result

        self.replace(backend, "_cache", cache)
        for key, spec in list(backend._observations.items()):
            original_decode = spec.decode

            def decode(message, spec=spec, original_decode=original_decode):
                begin = time.monotonic_ns()
                raw, source_stamp_ns = message_data(message)
                metadata.clear()
                metadata.update(
                    topic=spec.topic,
                    source_stamp_ns=source_stamp_ns,
                    source_clock="ros",
                    source_stamp_semantics="message_header_unverified",
                    callback_probe_ns=begin,
                    callback_wall_ns=time.time_ns(),
                )
                blobs = None
                if self.raw_video and "h264" in getattr(
                    message, "format", ""
                ).lower().replace(".", ""):
                    blobs = {"mosaic.h264": bytes(message.data)}
                try:
                    result = original_decode(message)
                except BaseException as error:
                    self.trace.emit(
                        "sensor_receive",
                        t_ns=begin,
                        arrays=raw,
                        blobs=blobs,
                        status="decode_error",
                        error=repr(error),
                        **metadata,
                    )
                    raise
                metadata["decode_duration_ms"] = (time.monotonic_ns() - begin) / 1e6
                self.trace.emit(
                    "sensor_receive",
                    t_ns=begin,
                    arrays=raw,
                    blobs=blobs,
                    status="decoded" if result is not None else "no_sample",
                    **metadata,
                )
                return result

            backend._observations[key] = dataclasses.replace(spec, decode=decode)
            self.restore.append((backend._observations, key, spec))

        original_open = backend._open
        trace = self.trace

        class Publisher:
            def __init__(self, base, topic):
                self.base, self.topic = base, topic

            def publish(self, message):
                begin = time.monotonic_ns()
                context = dict(trace.command or {})
                arrays, source = message_data(message)
                try:
                    result = self.base.publish(message)
                except BaseException as error:
                    trace.emit(
                        "ros_publish",
                        t_ns=begin,
                        arrays=arrays,
                        topic=self.topic,
                        source_stamp_ns=source,
                        status="error",
                        error=repr(error),
                        **context,
                    )
                    raise
                trace.emit(
                    "ros_publish",
                    t_ns=begin,
                    arrays=arrays,
                    topic=self.topic,
                    source_stamp_ns=source,
                    status="success",
                    publish_end_ns=time.monotonic_ns(),
                    **context,
                )
                return result

            def __getattr__(self, name):
                return getattr(self.base, name)

        def opened():
            result = original_open()
            for key, publisher in backend._publishers.items():
                backend._publishers[key] = Publisher(
                    publisher, backend._actions[key].topic
                )
            return result

        self.replace(backend, "_open", opened)

    def __exit__(self, *exc):
        for obj, name, original in reversed(self.restore):
            if isinstance(obj, dict):
                obj[name] = original
            else:
                setattr(obj, name, original)
        self.restore.clear()
