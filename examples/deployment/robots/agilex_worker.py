#!/usr/bin/env python3
"""ROS bridge for AgileX. The parent process owns inference and action pacing.

Run with the system ROS Python, not the PhyAI virtual environment. One write
publishes one dual-arm target. No interpolation, clipping or controller services.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import threading
import time

from agilex_protocol import (
    JOINT_NAMES,
    SIDES,
    VIEWS,
    decode_image,
    decode_joint_state,
    encode_array,
    receive_packet,
    send_packet,
)


class Runtime:
    def __init__(self, config):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, JointState

        self.config = config
        self.rclpy, self.JointState = rclpy, JointState
        self.lock = threading.Lock()
        self.samples, self.errors = {}, {}
        self.stopped = False
        self.sent = 0
        rclpy.init(args=[])
        try:
            self.node = rclpy.create_node(f"{config['node_name']}_{os.getpid()}")
            for view in VIEWS:
                self.node.create_subscription(
                    Image,
                    config["image_topics"][view],
                    lambda msg, key=view: self.on_sample(key, msg, image=True),
                    qos_profile_sensor_data,
                )
            for side in SIDES:
                self.node.create_subscription(
                    JointState,
                    config["state_topics"][side],
                    lambda msg, key=side: self.on_sample(
                        f"state_{key}", msg, image=False
                    ),
                    qos_profile_sensor_data,
                )
            self.publishers = {
                side: self.node.create_publisher(
                    JointState, config["command_topics"][side], 10
                )
                for side in SIDES
            }
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self.spin, daemon=True)
            self.thread.start()
        except BaseException:
            if hasattr(self, "node"):
                self.node.destroy_node()
            rclpy.shutdown()
            raise

    def spin(self):
        try:
            self.executor.spin()
        except Exception as error:
            with self.lock:
                self.errors["executor"] = str(error)

    def on_sample(self, key, message, *, image):
        stamp = time.monotonic_ns()
        try:
            value = decode_image(message) if image else decode_joint_state(message)
            with self.lock:
                self.samples[key] = (value, stamp)
                self.errors.pop(key, None)
        except Exception as error:
            with self.lock:
                self.errors[key] = str(error)

    def snapshot(self):
        keys = (*VIEWS, *(f"state_{side}" for side in SIDES))
        with self.lock:
            if self.errors:
                raise RuntimeError(f"ROS sensor error: {self.errors}")
            missing = set(keys) - set(self.samples)
            if missing:
                raise LookupError(f"Waiting for sensor samples: {sorted(missing)}")
            samples = dict(self.samples)
        return {
            "samples": {
                key: {"array": encode_array(value), "received_at_ns": stamp}
                for key, (value, stamp) in samples.items()
            }
        }

    def write(self, target):
        if self.stopped:
            raise RuntimeError("Robot is stopped")
        if len(target) != 14:
            raise ValueError("Expected 14 joint/gripper positions")
        for index, side in enumerate(SIDES):
            message = self.JointState()
            message.header.stamp = self.node.get_clock().now().to_msg()
            message.name = list(JOINT_NAMES)
            message.position = target[index * 7 : (index + 1) * 7]
            self.publishers[side].publish(message)
        self.sent += 1

    def stop(self):
        # Stop publishing only. Do not move, home, hold, or disable the motors.
        self.stopped = True

    def close(self):
        self.stop()
        self.executor.shutdown(timeout_sec=2.0)
        self.thread.join(timeout=2.0)
        self.node.destroy_node()
        self.rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    channel = socket.socket(fileno=parser.parse_args().fd)
    runtime = None
    try:
        first = receive_packet(channel, 10.0)
        if first.get("command") != "init":
            raise ValueError("First worker request must be init")
        runtime = Runtime(first["config"])

        def interrupted(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        send_packet(channel, {"ok": True})
        while True:
            request = receive_packet(channel)
            try:
                command = request["command"]
                result = {}
                if command == "observe":
                    result = runtime.snapshot()
                elif command == "write":
                    runtime.write(request["target"])
                elif command in ("stop", "close"):
                    runtime.stop()
                else:
                    raise ValueError(f"Unknown command: {command}")
                send_packet(
                    channel, {"ok": True, "published_commands": runtime.sent, **result}
                )
                if command == "close":
                    break
            except Exception as error:
                send_packet(
                    channel,
                    {
                        "ok": False,
                        "error": str(error),
                        "kind": "not_ready"
                        if isinstance(error, LookupError)
                        else "error",
                    },
                )
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        if runtime is not None:
            runtime.close()
        channel.close()


if __name__ == "__main__":
    main()
