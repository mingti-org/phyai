"""Benchmark raw RGB to eight RobotTrack waypoints through PhyAI."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from phyai.engine import Engine, EngineArgs
from phyai.engine_config import (
    BackendConfig,
    DeviceConfig,
    EngineConfig,
    ParallelConfig,
    RuntimeConfig,
)
from phyai.models.minicpm_robot_track.configuration_minicpm_robot_track import (
    MiniCPMRobotTrackConfig,
)
from phyai.models.minicpm_robot_track.main_minicpm_robot_track import (
    MiniCPMRobotTrackArgs,
)
from phyai.models.minicpm_robot_track.scheduler_ws1_minicpm_robot_track import (
    MiniCPMRobotTrackImageOutput,
    MiniCPMRobotTrackImageRequest,
)
from phyai.utils import load_config
from phyai_utils_tools.models.minicpm_robot_track import MiniCPMRobotTrackProcessor
from PIL import Image
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dino-engine", required=True, type=Path)
    parser.add_argument("--siglip-engine", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--instruction", default="Follow the person in the red shirt.")
    parser.add_argument(
        "--attention-backend", choices=("flashinfer", "sdpa"), default="flashinfer"
    )
    parser.add_argument(
        "--linear-kernel", choices=("auto", "flashinfer", "torch"), default="auto"
    )
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-vision-cuda-graph", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--cold-iters", type=int, default=3)
    parser.add_argument(
        "--resize-workers",
        type=int,
        help="PIL resize workers; defaults to min(8, CPU count). Use 1 for serial.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_rgb_window(frames_dir: Path, window_size: int = 32) -> torch.Tensor:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(f"No RGB images found under {frames_dir}.")
    paths = paths[-window_size:]
    paths = [paths[0]] * (window_size - len(paths)) + paths
    arrays = [
        np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths
    ]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"All source frames must have the same shape, got {shapes}.")
    return torch.from_numpy(np.stack(arrays, axis=0))


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if (
        args.warmup < 0
        or args.iters <= 0
        or args.cold_iters <= 0
        or (args.resize_workers is not None and args.resize_workers <= 0)
    ):
        raise ValueError(
            "--warmup must be non-negative; --iters, --cold-iters, and an "
            "explicit --resize-workers must be positive."
        )

    config = load_config(args.checkpoint, MiniCPMRobotTrackConfig)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = MiniCPMRobotTrackProcessor(
        tokenizer=tokenizer,
        tokenizer_name=str(args.checkpoint),
        image_size=384,
        history_frames=config.history_frames,
        text_capacity=config.text_capacity,
        resize_workers=args.resize_workers,
    )
    raw_frames = load_rgb_window(args.frames_dir, config.history_frames + 1)
    processor_start = time.perf_counter()
    processed = processor.preprocess({"images": raw_frames, "task": args.instruction})
    cold_processor_ms = (time.perf_counter() - processor_start) * 1000.0

    engine_config = EngineConfig(
        backends=BackendConfig(attn=args.attention_backend, norm="flashinfer"),
        device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
        parallel=ParallelConfig(),
        runtime=RuntimeConfig(
            use_cuda_graph=not args.no_cuda_graph,
            force_linear_kernel=(
                None if args.linear_kernel == "auto" else args.linear_kernel
            ),
        ),
    )
    engine = Engine(
        EngineArgs(
            plugin="minicpm_robot_track",
            plugin_args=MiniCPMRobotTrackArgs(
                checkpoint_dir=args.checkpoint,
                config=config,
                dino_engine_path=args.dino_engine,
                siglip_engine_path=args.siglip_engine,
                use_vision_cuda_graph=not args.no_vision_cuda_graph,
            ),
            config=engine_config,
        )
    )
    try:
        cold = engine.step(
            MiniCPMRobotTrackImageRequest(
                frames=processed.frames,
                input_ids=processed.input_ids,
                text_lengths=processed.text_lengths,
                stream_id="benchmark",
                frame_index=config.history_frames,
                collect_timing=True,
            )
        )
        if not isinstance(cold, MiniCPMRobotTrackImageOutput):
            raise TypeError(f"Unexpected cold-start output: {type(cold).__name__}.")

        raw_new_frame = raw_frames[-1:]
        next_frame_index = config.history_frames + 1
        for _ in range(args.warmup):
            steady_inputs = processor.preprocess(
                {"images": raw_new_frame, "task": args.instruction}
            )
            engine.step(
                MiniCPMRobotTrackImageRequest(
                    frames=steady_inputs.frames,
                    input_ids=steady_inputs.input_ids,
                    text_lengths=steady_inputs.text_lengths,
                    stream_id="benchmark",
                    frame_index=next_frame_index,
                )
            )
            next_frame_index += 1
        torch.cuda.synchronize()

        samples: dict[str, list[float]] = {}
        processor_samples: list[float] = []
        raw_total_samples: list[float] = []
        output = cold
        for _ in range(args.iters):
            raw_start = time.perf_counter()
            processor_start = time.perf_counter()
            steady_inputs = processor.preprocess(
                {"images": raw_new_frame, "task": args.instruction}
            )
            processor_samples.append((time.perf_counter() - processor_start) * 1000.0)
            result = engine.step(
                MiniCPMRobotTrackImageRequest(
                    frames=steady_inputs.frames,
                    input_ids=steady_inputs.input_ids,
                    text_lengths=steady_inputs.text_lengths,
                    stream_id="benchmark",
                    frame_index=next_frame_index,
                    collect_timing=True,
                )
            )
            next_frame_index += 1
            if not isinstance(result, MiniCPMRobotTrackImageOutput):
                raise TypeError(f"Unexpected sliding output: {type(result).__name__}.")
            processor.postprocess(result.waypoints)
            raw_total_samples.append((time.perf_counter() - raw_start) * 1000.0)
            output = result
            for name, value in result.timing_ms.items():
                samples.setdefault(name, []).append(value)

        sliding_output = output
        complete_window_engine_samples: list[float] = []
        complete_window_processor_samples: list[float] = []
        complete_window_raw_samples: list[float] = []
        for index in range(args.cold_iters):
            raw_start = time.perf_counter()
            processor_start = time.perf_counter()
            window_inputs = processor.preprocess(
                {"images": raw_frames, "task": args.instruction}
            )
            complete_window_processor_samples.append(
                (time.perf_counter() - processor_start) * 1000.0
            )
            result = engine.step(
                MiniCPMRobotTrackImageRequest(
                    frames=window_inputs.frames,
                    input_ids=window_inputs.input_ids,
                    text_lengths=window_inputs.text_lengths,
                    stream_id="benchmark",
                    frame_index=10_000 + index * (config.history_frames + 1),
                    collect_timing=True,
                )
            )
            if not isinstance(result, MiniCPMRobotTrackImageOutput):
                raise TypeError(
                    f"Unexpected complete-window output: {type(result).__name__}."
                )
            processor.postprocess(result.waypoints)
            complete_window_engine_samples.append(result.timing_ms["total_ms"])
            complete_window_raw_samples.append(
                (time.perf_counter() - raw_start) * 1000.0
            )
            output = result

        waypoints = processor.postprocess(output.waypoints)
        if tuple(waypoints.shape) != (1, config.num_waypoints, config.action_dim):
            raise RuntimeError(f"Unexpected waypoint shape: {tuple(waypoints.shape)}.")
        if not bool(torch.isfinite(waypoints).all()):
            raise RuntimeError("Waypoints contain non-finite values.")

        result_json = {
            "model": "MiniCPM-RobotTrack",
            "device": torch.cuda.get_device_name(),
            "input_seq_length": config.input_seq_length,
            "policy_cuda_graph": not args.no_cuda_graph,
            "vision_cuda_graph": not args.no_cuda_graph
            and not args.no_vision_cuda_graph,
            "resize_workers": args.resize_workers or "auto",
            "raw_image_shape": list(raw_frames.shape),
            "processed_image_shape": list(processed.frames.shape),
            "cold_start_encoded_frames": cold.encoded_frames,
            "cold_start_processor_ms": cold_processor_ms,
            "cold_start_32_frames_ms": cold.timing_ms,
            "cold_start_raw_to_waypoint_ms": (
                cold_processor_ms + cold.timing_ms["total_ms"]
            ),
            "complete_window_32_engine_ms": summarize(complete_window_engine_samples),
            "complete_window_32_processor_ms": summarize(
                complete_window_processor_samples
            ),
            "complete_window_32_raw_to_waypoint_ms": summarize(
                complete_window_raw_samples
            ),
            "sliding_encoded_frames_per_request": sliding_output.encoded_frames,
            "sliding_single_frame_request_ms": {
                name: summarize(values) for name, values in samples.items()
            },
            "client_processor_ms": summarize(processor_samples),
            "raw_rgb_to_waypoint_ms": summarize(raw_total_samples),
            "sliding_throughput_hz": 1000.0 / statistics.mean(samples["total_ms"]),
            "waypoints_shape": list(waypoints.shape),
            "waypoints": waypoints[0].tolist(),
            "warmup": args.warmup,
            "iterations": args.iters,
            "cold_iterations": args.cold_iters,
        }
        print(json.dumps(result_json, indent=2, ensure_ascii=True))
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(result_json, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        return 0
    finally:
        engine.close()
        processor.close()


if __name__ == "__main__":
    raise SystemExit(main())
