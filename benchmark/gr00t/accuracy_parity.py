"""GR00T-N1.7 LIBERO decoded-action accuracy alignment benchmark.

The benchmark evaluates recorded LIBERO trajectories with the standard
``gr00t_n17`` Engine plugin. It reads state and ground-truth actions from the
dataset's parquet files, decodes both camera videos, and compares decoded model
actions with the recorded actions. MSE and MAE are therefore measured in the
decoded LIBERO action representation, not model-normalized space.

Run::

    uv run --with pyarrow python benchmark/gr00t/accuracy_parity.py \
        --checkpoint <gr00t-checkpoint-dir> \
        --dataset-path <isaac-gr00t-dir>/demo_data/libero_demo \
        --embodiment-tag LIBERO_PANDA \
        --traj-ids 0 1 2 3 4 --steps 200 --action-horizon 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.gr00t_n17.configuration_gr00t_n17 import GR00TN17Config
from phyai.models.gr00t_n17.main_gr00t_n17 import GR00TN17Args
from phyai.models.gr00t_n17.scheduler_ws1_gr00t_n17 import GR00TN17Request
from phyai.utils import load_config
from phyai_utils_tools.models.gr00t import GR00TObservation, GR00TProcessor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_tasks(dataset: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    with (dataset / "meta" / "tasks.jsonl").open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            tasks[int(item["task_index"])] = str(item["task"])
    return tasks


def episode_path(dataset: Path, trajectory_id: int) -> Path:
    return dataset / "data" / "chunk-000" / f"episode_{trajectory_id:06d}.parquet"


def load_trajectory(dataset: Path, trajectory_id: int):
    path = episode_path(dataset, trajectory_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing LIBERO trajectory: {path}")
    return pq.read_table(path)


def trajectory_row(trajectory, index: int) -> dict[str, Any]:
    return {
        name: trajectory[name][index].as_py()
        for name in ("observation.state", "action", "task_index")
    }


def split_vector(
    vector: np.ndarray,
    modality_meta: dict[str, Any],
    modality: str,
    key: str,
) -> np.ndarray:
    start = int(modality_meta[modality][key]["start"])
    end = int(modality_meta[modality][key]["end"])
    return np.asarray(vector[start:end], dtype=np.float32)


def load_frames(
    dataset: Path,
    frame_cache: Path | None,
    trajectory_id: int,
    video_keys: list[str],
) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for key in video_keys:
        if frame_cache is not None:
            path = frame_cache / f"traj_{trajectory_id:06d}_{key}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing decoded frame cache: {path}")
            frames[key] = np.load(path, mmap_mode="r")
            continue

        path = (
            dataset
            / "videos"
            / "chunk-000"
            / f"observation.images.{key}"
            / f"episode_{trajectory_id:06d}.mp4"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing LIBERO video: {path}. Clone Isaac-GR00T with Git LFS "
                "so demo_data/libero_demo contains the MP4 files."
            )
        with av.open(str(path)) as container:
            decoded = [
                frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
            ]
        if not decoded:
            raise ValueError(f"LIBERO video contains no frames: {path}")
        frames[key] = np.stack(decoded, axis=0)
    return frames


def make_observation(
    row: dict[str, Any],
    frames: dict[str, np.ndarray],
    step: int,
    task: str,
    processor: GR00TProcessor,
    modality_meta: dict[str, Any],
) -> GR00TObservation:
    config = processor.modality_config
    state_vector = np.asarray(row["observation.state"], dtype=np.float32)
    state = {
        key: split_vector(state_vector, modality_meta, "state", key)[None, None, :]
        for key in config["state"].modality_keys
    }
    video = {
        key: np.array(frames[key][step], dtype=np.uint8, copy=True)[None, None, ...]
        for key in config["video"].modality_keys
    }
    language_key = config["language"].modality_keys[0]
    return GR00TObservation(
        video=video,
        state=state,
        language={language_key: [[task]]},
    )


def prepare_request(
    processor: GR00TProcessor,
    trajectory,
    frames: dict[str, np.ndarray],
    step: int,
    modality_meta: dict[str, Any],
    tasks: dict[int, str],
) -> tuple[GR00TN17Request, dict[str, np.ndarray]]:
    if step >= trajectory.num_rows:
        raise ValueError(
            f"Trajectory has {trajectory.num_rows} rows; step {step} is out of range."
        )
    row = trajectory_row(trajectory, step)
    task = tasks[int(row["task_index"])]
    observation = make_observation(
        row,
        frames,
        step,
        task,
        processor,
        modality_meta,
    )
    prepared = processor.process_observation(observation)
    return GR00TN17Request(tensors=prepared.tensors), prepared.raw_state


def build_engine_and_processor(
    args: argparse.Namespace,
    dataset: Path,
    modality_meta: dict[str, Any],
    tasks: dict[int, str],
    frame_cache: Path | None,
) -> tuple[Engine, GR00TProcessor]:
    checkpoint = Path(args.checkpoint)
    config = load_config(checkpoint, GR00TN17Config)
    loading_kwargs = {
        "local_files_only": not args.online,
        "trust_remote_code": args.trust_remote_code,
    }
    processor = GR00TProcessor.from_pretrained(
        checkpoint,
        embodiment_tag=args.embodiment_tag,
        model_name=args.processor_model_name_or_path or config.backbone.model_name,
        transformers_loading_kwargs=loading_kwargs,
    )
    capture_profiles = []
    for trajectory_id in args.traj_ids:
        capture_trajectory = load_trajectory(dataset, trajectory_id)
        capture_frames = load_frames(
            dataset,
            frame_cache,
            trajectory_id,
            processor.modality_config["video"].modality_keys,
        )
        capture_profile, _ = prepare_request(
            processor,
            capture_trajectory,
            capture_frames,
            0,
            modality_meta,
            tasks,
        )
        capture_profiles.append(capture_profile)
    engine = Engine(
        EngineArgs(
            plugin="gr00t_n17",
            plugin_args=GR00TN17Args(
                checkpoint_dir=checkpoint,
                config=config,
                max_batch_size=1,
                capture_profiles=tuple(capture_profiles),
            ),
            config=EngineConfig(
                device=DeviceConfig(
                    target=args.device,
                    params_dtype=torch.bfloat16,
                ),
                runtime=RuntimeConfig(use_cuda_graph=not args.no_cuda_graph),
            ),
        )
    )
    return engine, processor


def concat_action_vector(
    action: dict[str, np.ndarray],
    action_keys: list[str],
    index: int,
) -> np.ndarray:
    return np.concatenate(
        [np.atleast_1d(action[f"action.{key}"][0, index]) for key in action_keys]
    )


def evaluate_trajectory(
    engine: Engine,
    processor: GR00TProcessor,
    dataset: Path,
    frame_cache: Path | None,
    modality_meta: dict[str, Any],
    tasks: dict[int, str],
    trajectory_id: int,
    *,
    steps: int,
    action_horizon: int,
) -> tuple[float, float, int]:
    trajectory = load_trajectory(dataset, trajectory_id)
    actual_steps = min(steps, trajectory.num_rows)
    if actual_steps <= 0:
        raise ValueError(f"Trajectory {trajectory_id} has no steps to evaluate.")

    action_keys = processor.modality_config["action"].modality_keys
    frames = load_frames(
        dataset,
        frame_cache,
        trajectory_id,
        processor.modality_config["video"].modality_keys,
    )
    if any(len(value) < actual_steps for value in frames.values()):
        raise ValueError(
            f"Trajectory {trajectory_id} video is shorter than {actual_steps} steps."
        )

    predicted: list[np.ndarray] = []
    request_count = 0
    for step in range(0, actual_steps, action_horizon):
        request, raw_state = prepare_request(
            processor,
            trajectory,
            frames,
            step,
            modality_meta,
            tasks,
        )
        normalized_action = engine.step(request)
        decoded_action = processor.decode_action(
            normalized_action,
            raw_state=raw_state,
        )
        valid_steps = min(action_horizon, actual_steps - step)
        if decoded_action[f"action.{action_keys[0]}"].shape[1] < valid_steps:
            raise ValueError(
                f"Decoded action horizon is shorter than the requested {valid_steps} steps."
            )
        predicted.extend(
            concat_action_vector(decoded_action, action_keys, index)
            for index in range(valid_steps)
        )
        request_count += 1

    prediction = np.asarray(predicted, dtype=np.float32)
    ground_truth = np.stack(
        [
            np.concatenate(
                [
                    split_vector(
                        np.asarray(row["action"], dtype=np.float32),
                        modality_meta,
                        "action",
                        key,
                    )
                    for key in action_keys
                ]
            )
            for row in (
                trajectory_row(trajectory, index) for index in range(actual_steps)
            )
        ]
    )
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match ground truth "
            f"{ground_truth.shape}."
        )
    mse = float(np.mean((ground_truth - prediction) ** 2))
    mae = float(np.mean(np.abs(ground_truth - prediction)))
    return mse, mae, request_count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    parser.add_argument("--traj-ids", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--processor-model-name-or-path", default=None)
    parser.add_argument(
        "--frame-cache",
        default=None,
        help="optional directory of pre-decoded traj_<id>_<camera>.npy files",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="allow Hugging Face downloads for tokenizer and preprocessor files",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "allow Transformers to execute custom code from the processor model "
            "repository; only use this with a repository you trust"
        ),
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="disable CUDA graphs for the backbone and action head",
    )
    args = parser.parse_args(argv)
    if not args.traj_ids:
        parser.error("--traj-ids requires at least one trajectory")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.action_horizon <= 0:
        parser.error("--action-horizon must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.trust_remote_code:
        warnings.warn(
            "--trust-remote-code allows execution of code from the processor "
            "model repository.",
            stacklevel=1,
        )
    set_seed(args.seed)

    dataset = Path(args.dataset_path)
    frame_cache = Path(args.frame_cache) if args.frame_cache else None
    modality_meta = load_json(dataset / "meta" / "modality.json")
    tasks = load_tasks(dataset)

    engine, processor = build_engine_and_processor(
        args,
        dataset,
        modality_meta,
        tasks,
        frame_cache,
    )
    try:
        mses: list[float] = []
        maes: list[float] = []
        total_requests = 0
        for trajectory_id in args.traj_ids:
            mse, mae, requests = evaluate_trajectory(
                engine,
                processor,
                dataset,
                frame_cache,
                modality_meta,
                tasks,
                trajectory_id,
                steps=args.steps,
                action_horizon=args.action_horizon,
            )
            print(
                f"trajectory {trajectory_id}: MSE={mse:.10f} "
                f"MAE={mae:.10f} requests={requests}"
            )
            mses.append(mse)
            maes.append(mae)
            total_requests += requests

        average_mse = float(np.mean(mses))
        average_mae = float(np.mean(maes))
        average_accuracy = 1.0 - average_mae
        print(
            f"average: MSE={average_mse:.6f} MAE={average_mae:.6f} "
            f"Avg Acc={average_accuracy:.6f} requests={total_requests}"
        )
        return 0
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
