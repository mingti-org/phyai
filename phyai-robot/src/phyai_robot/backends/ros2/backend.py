"""ROS 2 topic subscriptions, command publishing, and service coordination.

Architecture:
    Each `Ros2Backend` manages its own private `rclpy.context.Context`, `rclpy.node.Node`,
    and `rclpy.executors.SingleThreadedExecutor` confined strictly to its worker thread:
    - Isolation: Private contexts guarantee that shutting down one backend does not call
      the global `rclpy.shutdown()`, which would destroy sibling backends or host nodes.
    - Synchronized Subscriptions: Sensor topics are converted via user-provided decoders
      in the worker thread and cached as immutable Samples with local arrival timestamps.
    - Grouped Topic Bindings: Multiple fields can be decoded from a single multi-signal topic
      (e.g., both left and right arm joint feedback from one JointState message), guaranteeing
      strictly identical arrival timestamps without redundant network subscriptions.
    - Synchronous Service Coordination: `call_service()` enables executing ROS 2 service calls
      (such as controller mode switching or input mux enabling) directly on the worker's
      executor with bounded timeouts.
    - Non-Atomic Publish Pipeline: Commands are verified and encoded in full before the first
      message is published. However, multi-topic publication is not atomic at the DDS layer.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass
from collections.abc import Mapping, Callable

from numpy.typing import NDArray

from ...types import Action, Sample
from ..._compat import BaseExceptionGroup
from .._threaded import _ThreadedBackend

if TYPE_CHECKING:
    from rclpy.node import Node
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.publisher import Publisher

# Margin in seconds added to service timeouts to accommodate rclpy's 0.25 s polling interval.
SERVICE_POLL_MARGIN_S = 0.5

# YAML-friendly QoS profile aliases or native depths/objects:
#   "sensor_data": rclpy's best-effort sensor data profile (volatile, depth=10)
#   "reliable":    reliable delivery, volatile durability, depth=10
#   "best_effort": best-effort delivery, volatile durability, depth=10
QoS = Any


def resolve_qos(qos: QoS) -> Any:
    """Resolve a QoS descriptor (integer depth, string profile name, or QoSProfile).

    Args:
        qos: Integer history depth, string alias, or an existing QoSProfile.

    Returns:
        QoSProfile instance or integer depth suitable for rclpy subscriptions/publishers.

    Raises:
        ValueError: If qos string name is unknown or depth is not positive.
        TypeError: If qos is boolean or an unsupported type.
    """
    if isinstance(qos, bool):
        raise TypeError("QoS must be a depth, a profile name or a QoSProfile")
    if isinstance(qos, int):
        if qos < 1:
            raise ValueError("QoS depth must be positive")
        return qos
    if not isinstance(qos, str):
        return qos

    from rclpy.qos import (
        QoSProfile,
        HistoryPolicy,
        DurabilityPolicy,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )

    if qos == "sensor_data":
        return qos_profile_sensor_data
    reliability = {
        "reliable": ReliabilityPolicy.RELIABLE,
        "best_effort": ReliabilityPolicy.BEST_EFFORT,
    }.get(qos)
    if reliability is None:
        raise ValueError(f"unknown QoS profile name: {qos!r}")
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )


@dataclass(frozen=True)
class RosObservation:
    """Specification for decoding a ROS 2 topic into observation fields.

    Attributes:
        topic: Full ROS 2 topic path (e.g. '/tj/info/joint_feedback').
        message_type: Generated ROS 2 message class (e.g. sensor_msgs.msg.JointState).
        decode: Callable converting a ROS message instance into a NumPy array (for single
            field bindings) or a dictionary mapping field names to NumPy arrays (for tuple
            grouped bindings). Returning None indicates no sample was produced (e.g. video
            decoder waiting for an IDR keyframe).
        qos: Target QoS profile or depth matching the remote publisher.
        recover_on_next_message: If True, decoding exceptions report a sample fault in read()
            without terminating the worker thread, recovering automatically on the next valid frame.
    """

    topic: str
    message_type: type[Any]
    decode: Callable[[Any], NDArray[Any] | Mapping[str, NDArray[Any]] | None]
    qos: QoS = 10
    recover_on_next_message: bool = False


@dataclass(frozen=True)
class RosCommand:
    """Specification for encoding an action field or group into a ROS 2 command message.

    Attributes:
        topic: Full ROS 2 actuator command topic (e.g. '/tj/control/user/joint_cmd_A').
        message_type: Message class accepted by the remote controller (e.g. JointcmdArm).
        encode: Callable converting an Action array or grouped dictionary into a message instance.
        qos: Target QoS profile matching the actuator subscriber.
        stamp: If True, assigns message.header.stamp from the backend's ROS node clock immediately
            before publishing on the worker thread.
    """

    topic: str
    message_type: type[Any]
    encode: Callable[[NDArray[Any] | Mapping[str, NDArray[Any]]], Any]
    qos: QoS = 10
    stamp: bool = False


def _field_keys(bindings: Mapping[str | tuple[str, ...], Any]) -> list[str]:
    """Extract and flatten field names from binding keys, asserting no overlaps."""
    keys: list[str] = []
    for key in bindings:
        fields = (key,) if isinstance(key, str) else key
        if not isinstance(fields, tuple) or not fields:
            raise ValueError("ROS binding keys must be strings or nonempty tuples")
        if any(not isinstance(field, str) or not field for field in fields):
            raise ValueError("ROS fields must be nonempty strings")
        keys.extend(fields)
    if len(keys) != len(set(keys)):
        raise ValueError("ROS bindings must not overlap fields")
    return keys


class Ros2Backend(_ThreadedBackend):
    """ROS 2 transport session managing subscriptions, publishers, and service clients.

    Thread Model:
        Spins a SingleThreadedExecutor on its dedicated worker thread. Subscriptions
        cache frozen Samples with reception monotonic timestamps. Action writes encode
        and publish topic messages sequentially under a monotonic deadline.
    """

    def __init__(
        self,
        *,
        observations: Mapping[str | tuple[str, ...], RosObservation],
        actions: Mapping[str | tuple[str, ...], RosCommand],
        on_stop: Callable[[Ros2Backend], None] | None = None,
        # Mandatory for backends owning action commands; called on worker during stop.
        node_name: str = "phyai_robot",
        io_timeout_s: float = 0.1,
        stop_timeout_s: float | None = None,
        connect_timeout_s: float = 5.0,
        domain_id: int | None = None,  # None inherits ROS_DOMAIN_ID from environment.
    ) -> None:
        """Initialize ROS 2 session bindings without creating native handles.

        Args:
            observations: Mapping of field names (or tuples) to RosObservation specs.
            actions: Mapping of command names (or tuples) to RosCommand specs.
            on_stop: Safety callback invoked on the worker during stop/close.
            node_name: Unique name for this backend's ROS 2 node.
            io_timeout_s: Maximum budget for publishing single-tick commands.
            stop_timeout_s: Maximum budget for on_stop (service handshakes).
            connect_timeout_s: Budget for worker and context initialization.
            domain_id: ROS 2 domain ID (0-232), or None for shell default.
        """
        if actions and on_stop is None:
            raise ValueError("ROS command backends require a device-specific on_stop")
        super().__init__(
            _field_keys(observations),
            _field_keys(actions),
            io_timeout_s,
            connect_timeout_s,
            stop_timeout_s,
        )
        self._observations = dict(observations)
        self._actions = dict(actions)
        self._grouped_actions = any(isinstance(key, tuple) for key in actions)
        self._decode_errors: dict[str | tuple[str, ...], str] = {}
        self._domain_id = domain_id
        self._on_stop = on_stop
        self._node_name = node_name

        self._context: Context | None = None
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._publishers: dict[str | tuple[str, ...], Publisher[Any]] = {}
        self._service_clients: dict[str, Any] = {}

    def _open(self) -> None:
        """Initialize private rclpy context, node, subscriptions, and publishers."""
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

        context = Context()
        self._context = context
        context.init(
            args=[],
            **({"domain_id": self._domain_id} if self._domain_id is not None else {}),
        )
        node = Node(self._node_name, context=context, use_global_arguments=False)
        self._node = node
        executor = SingleThreadedExecutor(context=context)
        self._executor = executor
        executor.add_node(node)

        # Create topic subscriptions.
        for key, spec in self._observations.items():

            def receive(
                message: Any,
                key: str | tuple[str, ...] = key,
                spec: RosObservation = spec,
            ) -> None:
                received_at_ns: int = time.monotonic_ns()
                try:
                    decoded = spec.decode(message)
                    if decoded is None:
                        return
                    if isinstance(key, tuple):
                        if not isinstance(decoded, Mapping) or set(decoded) != set(key):
                            raise ValueError(
                                "ROS decoder fields do not match its binding"
                            )
                        values = decoded
                    else:
                        values = {key: decoded}
                    self._cache(values, received_at_ns)
                except Exception as error:
                    if not spec.recover_on_next_message:
                        raise
                    with self._lock:
                        self._decode_errors[key] = str(error)
                else:
                    with self._lock:
                        self._decode_errors.pop(key, None)

            node.create_subscription(
                spec.message_type, spec.topic, receive, resolve_qos(spec.qos)
            )

        # Create topic publishers.
        for key, spec in self._actions.items():
            self._publishers[key] = node.create_publisher(
                spec.message_type, spec.topic, resolve_qos(spec.qos)
            )

    def _poll(self) -> None:
        """Spin executor briefly on worker thread to dispatch subscription callbacks."""
        assert self._executor is not None
        self._executor.spin_once(timeout_sec=0.001)

    def read(self) -> Mapping[str, Sample]:
        """Return cached samples, propagating any active recoverable decode errors."""
        samples = super().read()
        with self._lock:
            if self._decode_errors:
                raise RuntimeError(f"ROS sensor error: {self._decode_errors}")
        return samples

    def call_service(
        self, service_type: type[Any], name: str, request: Any, *, timeout_s: float
    ) -> Any:
        """Call a ROS 2 service synchronously on this backend's worker thread.

        Args:
            service_type: Generated ROS service type (e.g. std_srvs.srv.Trigger).
            name: Full ROS service topic path (e.g. '/tj/control/set_mode').
            request: Service Request instance.
            timeout_s: End-to-end deadline covering service discovery and response.

        Returns:
            Service response object.

        Raises:
            TimeoutError: If service is not available or does not reply before timeout.
            RuntimeError: If response has success=False.
        """
        return self.call(
            lambda: self._call_service(service_type, name, request, timeout_s),
            timeout_s + SERVICE_POLL_MARGIN_S,
        )

    def _call_service(
        self, service_type: type[Any], name: str, request: Any, timeout_s: float
    ) -> Any:
        assert self._node is not None and self._executor is not None
        deadline = time.monotonic() + timeout_s
        client = self._service_clients.get(name)
        if client is None:
            client = self._node.create_client(service_type, name)
            self._service_clients[name] = client

        if not client.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError(f"ROS service {name} is not available")

        future = client.call_async(request)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            self._executor.spin_until_future_complete(future, timeout_sec=remaining)

        if not future.done():
            client.remove_pending_request(future)
            raise TimeoutError(f"ROS service {name} did not respond in time")

        response = future.result()
        if response is None:
            raise RuntimeError(f"ROS service {name} returned no response")
        if getattr(response, "success", True) is False:
            message = getattr(response, "message", "")
            raise RuntimeError(f"ROS service {name} rejected the request: {message}")
        return response

    def _write_device(self, action: Action, deadline: float) -> None:
        """Encode and publish single-step commands across all owned action topics."""
        keys = self._actions if self._grouped_actions else action
        messages = {
            key: self._actions[key].encode(
                {field: action[field] for field in key}
                if isinstance(key, tuple)
                else action[key]
            )
            for key in keys
        }

        # Verify message types and stamp presence prior to emitting any message.
        for key, message in messages.items():
            if not isinstance(message, self._actions[key].message_type):
                raise TypeError(f"{key}: encoder returned the wrong ROS message type")
            if self._actions[key].stamp and not hasattr(
                getattr(message, "header", None), "stamp"
            ):
                raise TypeError(f"{key}: stamped commands require header.stamp")

        # Publish messages sequentially.
        for key, message in messages.items():
            if time.monotonic() >= deadline:
                raise TimeoutError("ROS command task expired before publishing")
            if self._actions[key].stamp:
                message.header.stamp = self._node.get_clock().now().to_msg()
            self._publishers[key].publish(message)

    def _stop_device(self) -> None:
        """Execute safety callback on the worker thread during stop or shutdown."""
        if self._node is not None and self._on_stop is not None:
            self._on_stop(self)

    def _close_device(self) -> None:
        """Shutdown executor, destroy node, and terminate private rclpy context."""
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
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("ROS resource cleanup failed", errors)


def image_backends(
    topics: Mapping[str, str],
    *,
    message_type: type[Any],
    decode: Callable[[Any], NDArray[Any]],
    node_name: str,
    qos: QoS = "sensor_data",
    **options: Any,
) -> list[Ros2Backend]:
    """Create a list of observation-only backends, allocating one dedicated worker per camera.

    Prevents CPU-heavy image decoding/letterboxing on one camera from delaying frame
    arrival on other cameras or stalling actuator command dispatch.

    Args:
        topics: Mapping of observation field names to ROS image topic strings.
        message_type: Image message class (e.g. sensor_msgs.msg.Image).
        decode: Image decoder function producing an HWC uint8 array.
        node_name: Base node name prefix.
        qos: Subscriber QoS profile.
        options: Additional backend parameters forwarded to each Ros2Backend.

    Returns:
        List of configured Ros2Backend instances ready to be passed to CompositeRobot.
    """
    return [
        Ros2Backend(
            observations={
                field: RosObservation(
                    topic, message_type, decode, qos, recover_on_next_message=True
                )
            },
            actions={},
            node_name=f"{node_name}_{field}",
            **options,
        )
        for field, topic in topics.items()
    ]
