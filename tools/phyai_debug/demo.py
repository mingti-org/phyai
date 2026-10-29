"""Deterministic synthetic replay data, with no robot or model connection."""

from pathlib import Path

import numpy as np

from .assets import snapshot_robot
from .storage import TraceRecorder


def create_demo(output, robot_assets=None):
    recorder = TraceRecorder(
        output,
        {
            "synthetic": True,
            "label": "Synthetic tracking and freshness example",
            "client": {
                "control": {"max_sample_age_s": 0.5, "control_hz": 200},
                "schedule": {"action_hz": 25, "execution_horizon": 25},
            },
        },
        queue_capacity=16000,
    )
    origin = recorder.started_ns
    records = []

    def event(t, kind, **fields):
        records.append((t, kind, fields))

    def state(t):
        q = np.sin(np.arange(16) * 0.4 + t * 0.7) * 0.45
        q[[7, 15]] = (np.sin(t) + 1) / 2
        return q.astype(np.float32)

    for request in range(1, 10):
        t = request * 1.25 - 1
        q = state(t)
        images = {}
        for i, camera in enumerate(("head_left", "left_wrist", "right_wrist")):
            pixels = np.zeros((224, 224, 3), dtype=np.uint8)
            pixels[49:175, :, i] = 50 + request * 15
            pixels[65:140, 35 + request * 3 : 110 + request * 3, (i + 1) % 3] = 180
            images[camera] = pixels
        ages = {
            "joint_position_left": 12 + request * 4,
            "head_camera": 465 if request == 6 else 50 + request * 9,
            "gripper_feedback_left": 22,
        }
        event(
            t,
            "request",
            request_id=request,
            arrays={"state": q},
            images=images,
            ages_ms=ages,
            samples={
                key: {"received_at_ns": origin + round(t * 1e9) - round(age * 1e6)}
                for key, age in ages.items()
            },
            read_end_ns=origin + round(t * 1e9) - 1_000_000,
            freshness_checked_at_ns=origin + round(t * 1e9) - 500_000,
            max_age_ms=500,
            warmup=False,
        )
        actions = np.stack([state(t + 0.18 + k / 25) for k in range(50)])
        event(
            t + 0.18,
            "response",
            request_id=request,
            arrays={"actions": actions},
            duration_ms=180,
            status="success",
        )
        event(
            t + 0.19,
            "chunk",
            request_id=request,
            chunk_id=request,
            phase="started",
            execution_horizon=25,
            horizon=50,
            action_hz=25,
            start_ns=origin + round((t + 0.19) * 1e9),
        )
        for tick in range(200):
            at = t + 0.19 + tick / 200
            target = state(at)
            event(
                at,
                "command",
                request_id=request,
                chunk_id=request,
                status="success",
                arrays={
                    "joint_position_left": target[:7],
                    "joint_position_right": target[8:15],
                    "gripper_left": target[7:8],
                    "gripper_right": target[15:16],
                },
                duration_ms=0.2,
                tick_ns=origin + round(at * 1e9),
            )
    for t in np.arange(0, 12, 0.02):
        q = state(t - 0.12)
        event(
            float(t),
            "feedback",
            arrays={
                "joint_position_left": q[:7],
                "joint_position_right": q[8:15],
                "gripper_feedback_left": np.array([q[7] * 1.6]),
                "gripper_feedback_right": np.array([q[15] * 1.6]),
            },
        )
    event(0, "mode", topic="/tj/control/input_mode", value=3)
    event(7.5, "mode", topic="/tj/control/input_mode", value=1)
    event(8.4, "mode", topic="/tj/control/input_mode", value=3)
    event(
        11.8,
        "event",
        name="synthetic_stop",
        message="Synthetic fixture, never robot telemetry",
    )
    try:
        if robot_assets:
            snapshot_robot(robot_assets, recorder.output)
        for t, kind, fields in sorted(records, key=lambda row: row[0]):
            recorder.emit(kind, t_ns=origin + round(t * 1e9), **fields)
    finally:
        result = recorder.close(grace_s=30)
    if not result["complete"]:
        raise RuntimeError(f"Incomplete demo: {result}")
    print(Path(output).resolve())
