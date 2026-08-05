"""Run matched closed-loop LIBERO rollouts through a remote policy server.

This client intentionally uses the same observation and action conversions as
Isaac-GR00T's official LIBERO rollout code. It is Python 3.8 compatible so it
can run in a disposable LIBERO simulator container while the model server uses
its own independent environment.
"""

import argparse
import json
import math
import os
import time

import msgpack
import msgpack_numpy
import numpy as np
import zmq


ACTION_KEYS = (
    "action.x",
    "action.y",
    "action.z",
    "action.roll",
    "action.pitch",
    "action.yaw",
    "action.gripper",
)


class MsgSerializer:
    """Serialize requests using the official msgpack-numpy wire format."""

    @staticmethod
    def to_bytes(data):
        def encode(value):
            if isinstance(value, np.ndarray) and value.dtype.kind == "O":
                raise TypeError("Object-dtype arrays are not safe to serialize")
            return msgpack_numpy.encode(value)

        return msgpack.packb(data, default=encode)

    @staticmethod
    def from_bytes(data):
        def decode(value):
            if isinstance(value, dict):
                marker_key = "".join(("n", "d"))
                ndarray_marker = value.get(marker_key.encode(), value.get(marker_key))
                kind = value.get(b"kind", value.get("kind"))
                if ndarray_marker and kind in (b"O", "O"):
                    raise ValueError("Refusing a pickle-bearing ndarray")
            return msgpack_numpy.decode(value)

        return msgpack.unpackb(data, object_hook=decode, raw=False)


class PolicyClient:
    """Minimal client for the endpoints used during LIBERO evaluation."""

    def __init__(self, host, port, timeout_ms):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.socket = None
        self.init_socket()

    def init_socket(self):
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect("tcp://{}:{}".format(self.host, self.port))

    def call(self, endpoint, data=None, requires_input=True):
        request = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            response = MsgSerializer.from_bytes(self.socket.recv())
        except zmq.error.Again:
            self.init_socket()
            raise
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError("Policy server error: {}".format(response["error"]))
        return response

    def ping(self):
        return self.call("ping", requires_input=False)

    def reset(self, seed):
        return self.call("reset", {"options": {"seed": int(seed)}})

    def get_action(self, observation):
        response = self.call(
            "get_action", {"observation": observation, "options": None}
        )
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise TypeError("Policy response must be an (action, info) pair")
        return response[0], response[1]

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


def quat2axisangle(quaternion):
    """Convert an (x, y, z, w) quaternion to axis-angle coordinates."""
    quaternion = np.asarray(quaternion).copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quaternion[:3] * 2.0 * math.acos(quaternion[3]) / denominator


def process_observation(observation, task_description):
    """Match ``gr00t.eval.sim.LIBERO.libero_env.LiberoEnv`` exactly."""
    xyz = observation["robot0_eef_pos"]
    rpy = quat2axisangle(observation["robot0_eef_quat"])
    return {
        "video.image": np.ascontiguousarray(
            observation["agentview_image"][::-1, ::-1][None, None, ...]
        ),
        "video.wrist_image": np.ascontiguousarray(
            observation["robot0_eye_in_hand_image"][::-1, ::-1][None, None, ...]
        ),
        "state.x": np.asarray([[[xyz[0]]]], dtype=np.float32),
        "state.y": np.asarray([[[xyz[1]]]], dtype=np.float32),
        "state.z": np.asarray([[[xyz[2]]]], dtype=np.float32),
        "state.roll": np.asarray([[[rpy[0]]]], dtype=np.float32),
        "state.pitch": np.asarray([[[rpy[1]]]], dtype=np.float32),
        "state.yaw": np.asarray([[[rpy[2]]]], dtype=np.float32),
        "state.gripper": np.asarray(
            observation["robot0_gripper_qpos"][None, None, ...], dtype=np.float32
        ),
        "annotation.human.action.task_description": [task_description],
    }


def get_action_chunk(action, n_action_steps):
    arrays = []
    horizon = None
    for key in ACTION_KEYS:
        if key not in action:
            raise KeyError("Policy action is missing {!r}".format(key))
        value = np.asarray(action[key])
        if value.ndim < 2 or value.shape[0] != 1:
            raise ValueError(
                "Expected action {!r} to begin with (batch=1, horizon), got {}".format(
                    key, value.shape
                )
            )
        if horizon is None:
            horizon = value.shape[1]
        elif value.shape[1] != horizon:
            raise ValueError("Action keys have inconsistent horizons")
        arrays.append(value[0])
    chunk = np.concatenate(arrays, axis=-1)
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise ValueError("Decoded action chunk must have shape (horizon, 7)")
    return chunk[:n_action_steps].astype(np.float32, copy=False)


def convert_action_for_libero(action):
    """Apply the official LIBERO gripper normalization and inversion."""
    action = np.asarray(action, dtype=np.float32).copy()
    action[-1] = np.sign(2.0 * action[-1] - 1.0)
    action[-1] *= -1.0
    return action


def make_environment(task):
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    return OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
        ignore_done=True,
    )


def run_episode(
    env,
    task,
    client,
    seed,
    max_episode_steps,
    n_action_steps,
    initial_state=None,
):
    env.seed(int(seed))
    observation = env.reset()
    if initial_state is not None:
        observation = env.set_init_state(initial_state)
    client.reset(seed)
    success = bool(env.check_success())
    low_level_steps = 0
    policy_calls = 0
    inference_seconds = 0.0
    started = time.time()

    while not success and low_level_steps < max_episode_steps:
        policy_observation = process_observation(observation, task.language)
        inference_started = time.time()
        action, _ = client.get_action(policy_observation)
        inference_seconds += time.time() - inference_started
        policy_calls += 1
        chunk = get_action_chunk(action, n_action_steps)
        remaining = max_episode_steps - low_level_steps
        for action_step in chunk[:remaining]:
            observation, _, _, _ = env.step(convert_action_for_libero(action_step))
            low_level_steps += 1
            success = bool(env.check_success())
            if success:
                break

    return {
        "success": success,
        "low_level_steps": low_level_steps,
        "policy_calls": policy_calls,
        "inference_seconds": inference_seconds,
        "wall_seconds": time.time() - started,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-ids", nargs="*", type=int, default=None)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--fixed-init-states", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if args.episodes_per_task <= 0:
        parser.error("--episodes-per-task must be positive")
    if args.max_episode_steps <= 0 or args.n_action_steps <= 0:
        parser.error("step counts must be positive")
    return args


def wilson_interval(successes, trials):
    if trials == 0:
        return [None, None]
    z = 1.959963984540054
    proportion = successes / float(trials)
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [center - radius, center + radius]


def build_summary(args, task_ids, results):
    successes = sum(int(item["success"]) for item in results)
    per_task = []
    for task_id in task_ids:
        task_results = [item for item in results if item["task_id"] == task_id]
        task_successes = sum(int(item["success"]) for item in task_results)
        task_trials = len(task_results)
        per_task.append(
            {
                "task_id": task_id,
                "successes": task_successes,
                "trials": task_trials,
                "success_rate": (
                    task_successes / float(task_trials) if task_trials else None
                ),
                "wilson_95": wilson_interval(task_successes, task_trials),
            }
        )
    expected_trials = len(task_ids) * args.episodes_per_task
    return {
        "backend": args.backend,
        "server_info": args.server_info,
        "suite": args.suite,
        "successes": successes,
        "trials": len(results),
        "expected_trials": expected_trials,
        "complete": len(results) == expected_trials,
        "success_rate": successes / float(len(results)) if results else None,
        "wilson_95": wilson_interval(successes, len(results)),
        "seed": args.seed,
        "episodes_per_task": args.episodes_per_task,
        "max_episode_steps": args.max_episode_steps,
        "n_action_steps": args.n_action_steps,
        "fixed_init_states": args.fixed_init_states,
        "per_task": per_task,
        "results": sorted(results, key=lambda item: (item["task_id"], item["episode"])),
    }


def write_summary(args, task_ids, results):
    summary = build_summary(args, task_ids, results)
    result_dir = os.path.dirname(os.path.abspath(args.result_file))
    os.makedirs(result_dir, exist_ok=True)
    temporary_file = args.result_file + ".tmp"
    with open(temporary_file, "w") as result_file:
        json.dump(summary, result_file, indent=2, sort_keys=True)
        result_file.write("\n")
    os.replace(temporary_file, args.result_file)
    return summary


def load_resume_results(args):
    if not args.resume or not os.path.exists(args.result_file):
        return []
    with open(args.result_file) as result_file:
        summary = json.load(result_file)
    expected = {
        "backend": args.backend,
        "suite": args.suite,
        "seed": args.seed,
        "episodes_per_task": args.episodes_per_task,
        "max_episode_steps": args.max_episode_steps,
        "n_action_steps": args.n_action_steps,
        "fixed_init_states": args.fixed_init_states,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                "Cannot resume: result field {!r} is {!r}, expected {!r}".format(
                    key, summary.get(key), value
                )
            )
    return list(summary.get("results", []))


def main():
    args = parse_args()
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = args.task_ids
    if task_ids is None or len(task_ids) == 0:
        task_ids = list(range(task_suite.get_num_tasks()))
    for task_id in task_ids:
        if task_id < 0 or task_id >= task_suite.get_num_tasks():
            raise ValueError("Invalid task id: {}".format(task_id))

    if args.probe_only:
        task = task_suite.get_task(task_ids[0])
        env = make_environment(task)
        try:
            env.seed(args.seed)
            observation = env.reset()
            if args.fixed_init_states:
                initial_states = task_suite.get_task_init_states(task_ids[0])
                observation = env.set_init_state(initial_states[0])
            processed = process_observation(observation, task.language)
            print(
                json.dumps(
                    {
                        "probe": "ok",
                        "task": task.name,
                        "image_shape": list(processed["video.image"].shape),
                        "wrist_shape": list(processed["video.wrist_image"].shape),
                        "fixed_init_states": args.fixed_init_states,
                    }
                ),
                flush=True,
            )
        finally:
            env.close()
        return

    client = PolicyClient(args.host, args.port, args.timeout_ms)
    try:
        ping = client.ping()
        print("Policy server: {}".format(ping), flush=True)
        args.server_info = ping
        results = load_resume_results(args)
        completed = {(item["task_id"], item["episode"]) for item in results}
        if results:
            print(
                "Resuming from {} completed episodes".format(len(results)), flush=True
            )
        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            initial_states = None
            if args.fixed_init_states:
                initial_states = task_suite.get_task_init_states(task_id)
                if args.episodes_per_task > len(initial_states):
                    raise ValueError(
                        "Task {} has {} fixed initial states, fewer than the {} "
                        "requested episodes".format(
                            task_id, len(initial_states), args.episodes_per_task
                        )
                    )
            env = make_environment(task)
            try:
                for episode in range(args.episodes_per_task):
                    if (task_id, episode) in completed:
                        continue
                    result = run_episode(
                        env,
                        task,
                        client,
                        args.seed + episode,
                        args.max_episode_steps,
                        args.n_action_steps,
                        initial_state=(
                            initial_states[episode]
                            if initial_states is not None
                            else None
                        ),
                    )
                    result.update(
                        {
                            "backend": args.backend,
                            "suite": args.suite,
                            "task_id": task_id,
                            "task_name": task.name,
                            "task_description": task.language,
                            "episode": episode,
                            "seed": args.seed + episode,
                            "init_state_id": (
                                episode if initial_states is not None else None
                            ),
                        }
                    )
                    results.append(result)
                    write_summary(args, task_ids, results)
                    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
            finally:
                env.close()
    finally:
        client.close()

    summary = write_summary(args, task_ids, results)
    compact_summary = {key: value for key, value in summary.items() if key != "results"}
    print("SUMMARY " + json.dumps(compact_summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
