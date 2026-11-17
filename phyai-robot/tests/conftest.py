"""CPU-only fixtures; invoke pytest with --confcutdir=phyai-robot."""

import numpy as np
import pytest
from phyai_robot import FeatureSpec, CompositeRobot
from numpy.typing import NDArray
from phyai_robot.backends.mock import MockBackend


@pytest.fixture
def vector() -> NDArray[np.float32]:
    return np.array([0.0, 0.0], dtype=np.float32)


@pytest.fixture
def robot_and_backend(
    vector: NDArray[np.float32],
) -> tuple[CompositeRobot, MockBackend]:
    backend: MockBackend = MockBackend(
        initial_values={"q": vector, "camera": np.zeros((2, 3, 3), dtype=np.uint8)},
        action_to_observation={"target": "q"},
    )
    robot: CompositeRobot = CompositeRobot(
        observation_schema={
            "q": FeatureSpec((2,), "float32"),
            "camera": FeatureSpec((2, 3, 3), "uint8"),
        },
        action_schema={"target": FeatureSpec((2,), "float32")},
        backends=[backend],
    )
    return robot, backend


class FakeRos:
    """In-memory ROS transport for backend tests; no installed rclpy is imported."""

    def __init__(self, monkeypatch):
        import sys
        import queue
        import types
        import threading
        from types import SimpleNamespace

        self.nodes = []
        self.contexts = []
        self.events = []
        self.publications = []
        self.stamps = 0
        self.sensor_qos = object()
        # Service name -> handler(request) -> response; calls record (name, data, thread).
        self.services = {}
        self.calls = []
        owner = self

        def record(kind, obj):
            owner.events.append((kind, obj, threading.get_ident()))

        class Context:
            def __init__(self):
                self.alive = False
                self.domain_id = None
                owner.contexts.append(self)

            def init(self, args=None, domain_id=None):
                self.domain_id = domain_id
                self.alive = True
                record("context.init", self)

            def ok(self):
                return self.alive

            def shutdown(self):
                self.alive = False
                record("context.close", self)

            try_shutdown = shutdown

        class Publisher:
            def __init__(self, node, topic, qos):
                self.node, self.topic, self.qos = node, topic, qos

            def publish(self, message):
                import copy

                record("publish", self.node)
                owner.publications.append((self.topic, copy.deepcopy(message)))

        class Node:
            def __init__(self, name, *, context, use_global_arguments=False):
                self.name, self.context = name, context
                self.subscriptions = {}
                self.publishers = {}
                self.pending = queue.Queue()
                self.pending_calls = []
                self.closed = False
                self.owner = threading.get_ident()
                owner.nodes.append(self)
                record("node.open", self)

            def create_subscription(self, message_type, topic, callback, qos):
                self.subscriptions.setdefault(topic, []).append(
                    (message_type, callback, qos)
                )
                record("subscribe", self)

            def create_publisher(self, message_type, topic, qos):
                publisher = Publisher(self, topic, qos)
                self.publishers[topic] = publisher
                record("publisher.open", self)
                return publisher

            def create_client(self, service_type, name):
                record("client.open", self)
                return Client(self, name)

            def get_clock(self):
                def stamp():
                    record("stamp", self)
                    owner.stamps += 1
                    return owner.stamps

                return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=stamp))

            def destroy_node(self):
                self.closed = True
                record("node.close", self)

        class Future:
            def __init__(self):
                self.value, self.finished = None, False

            def done(self):
                return self.finished

            def result(self):
                return self.value

        class Client:
            def __init__(self, node, name):
                self.node, self.name = node, name
                self.removed = []

            def wait_for_service(self, timeout_sec=None):
                return self.name in owner.services

            def call_async(self, request):
                future = Future()
                self.node.pending_calls.append((self.name, request, future))
                return future

            def remove_pending_request(self, future):
                self.removed.append(future)

        class Executor:
            def __init__(self, *, context):
                self.context = context
                self.nodes = []
                self.stopped = False

            def add_node(self, node):
                self.nodes.append(node)

            def spin_once(self, timeout_sec=0.001):
                for node in self.nodes:
                    try:
                        callback, message, done = node.pending.get(timeout=timeout_sec)
                    except queue.Empty:
                        continue
                    try:
                        callback(message)
                    finally:
                        done.set()

            def spin(self):
                while not self.stopped:
                    self.spin_once()

            def spin_until_future_complete(self, future, timeout_sec=None):
                import time

                deadline = time.monotonic() + (timeout_sec or 0)
                for node in self.nodes:
                    while node.pending_calls and time.monotonic() < deadline:
                        name, request, pending = node.pending_calls.pop(0)
                        record("service." + name, node)
                        owner.calls.append(
                            (
                                name,
                                getattr(request, "data", None),
                                threading.get_ident(),
                            )
                        )
                        response = owner.services[name](request)
                        if response is not None and time.monotonic() < deadline:
                            pending.value, pending.finished = response, True

            def shutdown(self, timeout_sec=None):
                self.stopped = True

        class JointState:
            def __init__(self):
                self.header = SimpleNamespace(stamp=None)
                self.name, self.position, self.velocity, self.effort = [], [], [], []

        class Image:
            pass

        def module(name, **values):
            obj = types.ModuleType(name)
            obj.__dict__.update(values)
            monkeypatch.setitem(sys.modules, name, obj)
            return obj

        rclpy = module("rclpy")
        rclpy.init = lambda args, context, domain_id: context.init(args, domain_id)
        rclpy.create_node = lambda name, context: Node(name, context=context)
        rclpy.context = module("rclpy.context", Context=Context)
        rclpy.node = module("rclpy.node", Node=Node)
        rclpy.executors = module("rclpy.executors", SingleThreadedExecutor=Executor)

        class QoSProfile(SimpleNamespace):
            pass

        policy = SimpleNamespace(
            KEEP_LAST="keep_last",
            VOLATILE="volatile",
            RELIABLE="reliable",
            BEST_EFFORT="best_effort",
        )
        rclpy.qos = module(
            "rclpy.qos",
            qos_profile_sensor_data=self.sensor_qos,
            QoSProfile=QoSProfile,
            HistoryPolicy=policy,
            DurabilityPolicy=policy,
            ReliabilityPolicy=policy,
        )
        sensor_msgs = module("sensor_msgs")
        sensor_msgs.msg = module("sensor_msgs.msg", Image=Image, JointState=JointState)
        self.JointState, self.Image = JointState, Image

    def feed(self, topic, message):
        import threading

        pending = []
        for node in self.nodes:
            if topic in node.subscriptions and not node.closed:
                for _, callback, _ in node.subscriptions[topic]:
                    done = threading.Event()
                    node.pending.put((callback, message, done))
                    pending.append(done)
        assert pending, f"No subscription for {topic}"
        assert all(done.wait(2) for done in pending), f"Callback timed out: {topic}"

    def assert_closed(self):
        assert all(node.closed for node in self.nodes)
        assert all(not context.alive for context in self.contexts)


@pytest.fixture
def fake_ros(monkeypatch):
    ros = FakeRos(monkeypatch)
    yield ros
    ros.assert_closed()
