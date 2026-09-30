"""ROS2 topic observations and commands with application-defined codecs.

ROS2, message packages, and a Python-compatible rclpy installation are supplied
by the host. Topic commands describe submission, not motion completion. Devices
requiring a specialized action/service protocol can implement Backend directly.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass
from collections.abc import Mapping, Callable

from numpy.typing import NDArray

from ...types import Action
from .._threaded import _ThreadedBackend

if TYPE_CHECKING:
    # Type-only imports keep this module importable without a ROS installation.
    from rclpy.qos import QoSProfile
    from rclpy.node import Node
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.publisher import Publisher


@dataclass(frozen=True)
class RosObservation:
    topic: str  # Full subscription topic, not a field name inferred by the backend.
    message_type: type[Any]  # Actual ROS message class supplied by the application.
    decode: Callable[[Any], NDArray[Any]]  # Convert order, units, frame, and dtype.
    qos: int | QoSProfile = (
        10  # Explicit QoSProfile or depth; match the actual publisher.
    )


@dataclass(frozen=True)
class RosCommand:
    topic: str  # Actual controller command topic.
    message_type: type[Any]  # Message class accepted by that controller.
    encode: Callable[[NDArray[Any]], Any]  # One named target -> one ROS message.
    qos: int | QoSProfile = 10  # Explicit publisher QoS or depth.


class Ros2Backend(_ThreadedBackend):
    def __init__(
        self,
        *,
        observations: Mapping[str, RosObservation],  # Field -> subscription/codec.
        actions: Mapping[str, RosCommand],  # Field -> publisher/codec.
        on_stop: Callable[[Node], None] | None = None,
        # Called with our node on its worker thread; mandatory for command backends.
        node_name: str = "phyai_robot",  # Node owned exclusively by this backend.
        io_timeout_s: float = 0.1,  # Bound for publishing/stop task completion.
        connect_timeout_s: float = 5.0,  # Context/node/executor initialization budget.
    ) -> None:
        if actions and on_stop is None:
            raise ValueError("ROS command backends require a device-specific on_stop")
        super().__init__(observations, actions, io_timeout_s, connect_timeout_s)
        self._observations: dict[str, RosObservation] = dict(observations)
        self._actions: dict[str, RosCommand] = dict(actions)
        self._on_stop: Callable[[Node], None] | None = on_stop
        self._node_name: str = node_name
        self._context: Context | None = (
            None  # Private ROS context; never shut down another user's context.
        )
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._publishers: dict[
            str, Publisher[Any]
        ] = {}  # Field -> publisher, accessed only on the worker.

    def _open(self) -> None:
        try:
            from rclpy.node import Node
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
        except ModuleNotFoundError as error:
            if error.name and (
                error.name == "rclpy" or error.name.startswith("rclpy.")
            ):
                raise ImportError(
                    "Ros2Backend needs a sourced ROS2 installation with rclpy "
                    "and message packages matching this Python version; "
                    "the ros2 extra does not install ROS middleware"
                ) from error
            raise
        context: Context = Context()
        self._context = context
        context.init(args=[])
        node: Node = Node(self._node_name, context=context, use_global_arguments=False)
        self._node = node
        executor: SingleThreadedExecutor = SingleThreadedExecutor(context=context)
        self._executor = executor
        executor.add_node(node)
        for key, spec in self._observations.items():

            def receive(
                message: Any, key: str = key, spec: RosObservation = spec
            ) -> None:
                # Message classes vary by topic; the application owns each codec.
                received_at_ns: int = time.monotonic_ns()
                self._cache({key: spec.decode(message)}, received_at_ns)

            node.create_subscription(spec.message_type, spec.topic, receive, spec.qos)
        for key, spec in self._actions.items():
            self._publishers[key] = node.create_publisher(
                spec.message_type, spec.topic, spec.qos
            )

    def _poll(self) -> None:
        # A short spin lets the same thread service command tasks promptly.
        # Callback exceptions escape to the worker and become observable faults.
        assert self._executor is not None  # _open completed on this worker.
        self._executor.spin_once(timeout_sec=0.001)

    def _write_device(self, action: Action, deadline: float) -> None:
        messages: dict[str, Any] = {
            key: self._actions[key].encode(value) for key, value in action.items()
        }
        # Encode every field before publishing any of them. Publishing two topics
        # still is not atomic and does not guarantee simultaneous physical motion.
        for key, message in messages.items():
            if not isinstance(message, self._actions[key].message_type):
                raise TypeError(f"{key}: encoder returned the wrong ROS message type")
        for key, message in messages.items():
            if time.monotonic() >= deadline:
                raise TimeoutError("ROS command task expired before publishing")
            self._publishers[key].publish(message)

    def _stop_device(self) -> None:
        if self._node is not None and self._on_stop is not None:
            # The hook must be bounded; it must not spin this same executor
            # recursively or wait indefinitely for a callback on this worker.
            self._on_stop(self._node)

    def _close_device(self) -> None:
        # Independent cleanup keeps a failed executor shutdown from leaking the
        # owned node/context. Do not call the global rclpy.shutdown().
        errors: list[BaseException] = []
        for cleanup in (
            lambda: (
                self._executor.shutdown(timeout_sec=self._io_timeout_s)
                if self._executor
                else None
            ),
            lambda: self._node.destroy_node() if self._node else None,
            lambda: self._context.try_shutdown() if self._context else None,
        ):
            try:
                cleanup()
            except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("ROS resource cleanup failed", errors)
