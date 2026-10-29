"""Passive downstream telemetry; this module never creates publishers or clients."""

import json
import threading
import time

from .compat import message_data


class RosMonitor:
    def __init__(self, trace, domain_id):
        self.trace, self.domain_id = trace, domain_id
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(
            target=self.run, name="trace-passive-ros", daemon=True
        )

    def start(self):
        self.thread.start()
        if not self.ready.wait(8) or self.error:
            self.close()
            raise RuntimeError(
                f"Passive trace subscriber failed before control startup: {self.error}"
            )

    def receive(self, topic, message):
        now = time.monotonic_ns()
        arrays, source = message_data(message)
        if topic.endswith("input_mode"):
            self.trace.emit("mode", t_ns=now, topic=topic, value=message.data)
        elif topic.endswith("events"):
            try:
                value = json.loads(message.data)
            except ValueError:
                value = {"unparsed": message.data}
            self.trace.emit("event", t_ns=now, name="dagger", topic=topic, detail=value)
        else:
            self.trace.emit(
                "ros_observed",
                t_ns=now,
                arrays=arrays,
                topic=topic,
                source_stamp_ns=source,
                observer_wall_ns=time.time_ns(),
                stage="mux_output",
                association="timestamp_only",
            )

    def run(self):
        node = executor = context = None
        try:
            from rclpy.context import Context
            from rclpy.node import Node
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import (
                QoSProfile,
                ReliabilityPolicy,
                DurabilityPolicy,
                qos_profile_sensor_data,
            )
            from marvin_msgs.msg import JointcmdArm
            from std_msgs.msg import Float32, Int32, String

            context = Context()
            context.init(domain_id=self.domain_id)
            node = Node(
                "phyai_debug_passive", context=context, use_global_arguments=False
            )
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            latched = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            topics = [
                ("/tj/control/joint_cmd_A", JointcmdArm, qos_profile_sensor_data),
                ("/tj/control/joint_cmd_B", JointcmdArm, qos_profile_sensor_data),
                ("/control/gripperValueL", Float32, 10),
                ("/control/gripperValueR", Float32, 10),
                ("/tj/control/input_mode", Int32, latched),
                ("/tj/control/gripper_input_mode", Int32, latched),
                ("/tj/dagger/events", String, 10),
            ]
            for topic, typ, qos in topics:
                node.create_subscription(
                    typ, topic, lambda m, t=topic: self.receive(t, m), qos
                )
            self.ready.set()
            while not self.stop_event.is_set():
                executor.spin_once(timeout_sec=0.05)
        except BaseException as error:
            self.error = repr(error)
            self.trace.sink.mark_incomplete("Passive ROS recorder: " + self.error)
            self.trace.emit("event", name="passive_recorder_failed", error=self.error)
            self.ready.set()
        finally:
            if executor:
                executor.shutdown()
            if node:
                node.destroy_node()
            if context:
                context.try_shutdown()

    def close(self):
        self.stop_event.set()
        if self.thread.ident:
            self.thread.join(3)
            if self.thread.is_alive():
                self.trace.sink.mark_incomplete(
                    "Passive ROS recorder did not shut down"
                )
