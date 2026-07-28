"""Benchmark MiniCPM-RobotTrack through the PhyAI Engine."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

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
    MiniCPMRobotTrackRequest,
)
from phyai.utils import load_config
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instruction", default="Follow the person in the red shirt.")
    parser.add_argument(
        "--attention-backend", choices=("flashinfer", "sdpa"), default="flashinfer"
    )
    parser.add_argument(
        "--linear-kernel", choices=("auto", "flashinfer", "torch"), default="auto"
    )
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def compact_text_inputs(
    tokenizer: Any, text: str, capacity: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        [text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=capacity,
    )
    mask = encoded.attention_mask[0].bool()
    valid = encoded.input_ids[0, mask]
    input_ids = torch.full(
        (1, capacity),
        int(tokenizer.pad_token_id),
        dtype=torch.long,
        device=device,
    )
    input_ids[0, : valid.numel()] = valid.to(device)
    text_lengths = torch.tensor([valid.numel()], dtype=torch.long, device=device)
    return input_ids, text_lengths


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")

    checkpoint = Path(args.checkpoint)
    config = load_config(checkpoint, MiniCPMRobotTrackConfig)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda"
    torch.manual_seed(args.seed)
    input_ids, text_lengths = compact_text_inputs(
        tokenizer, args.instruction, config.text_capacity, device
    )
    coarse_tokens = torch.randn(
        1,
        config.coarse_token_count,
        config.vision_feature_dim,
        dtype=torch.float32,
        device=device,
    )
    fine_tokens = torch.randn(
        1,
        config.fine_tokens_current_frame,
        config.vision_feature_dim,
        dtype=torch.float32,
        device=device,
    )
    coarse_time_indices = torch.arange(
        config.history_frames, dtype=torch.long, device=device
    ).repeat_interleave(config.coarse_tokens_per_frame)[None]
    fine_time_indices = torch.full(
        (1, config.fine_tokens_current_frame),
        config.history_frames,
        dtype=torch.long,
        device=device,
    )
    request = MiniCPMRobotTrackRequest(
        input_ids=input_ids,
        text_lengths=text_lengths,
        coarse_tokens=coarse_tokens,
        coarse_time_indices=coarse_time_indices,
        fine_tokens=fine_tokens,
        fine_time_indices=fine_time_indices,
    )

    engine_config = EngineConfig(
        backends=BackendConfig(
            attn=args.attention_backend,
            norm="flashinfer",
        ),
        device=DeviceConfig(target=device, params_dtype=torch.bfloat16),
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
                checkpoint_dir=checkpoint,
                config=config,
            ),
            config=engine_config,
        )
    )
    try:
        for _ in range(args.warmup):
            engine.step(request)
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        output = None
        for start, end in zip(starts, ends):
            start.record()
            output = engine.step(request)
            end.record()
        torch.cuda.synchronize()
        latencies_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        if output is None:
            raise RuntimeError("RobotTrack benchmark produced no output.")
        if output.shape != (1, config.num_waypoints, config.action_dim):
            raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("model output contains non-finite values")

        result = {
            "model": "MiniCPM-RobotTrack",
            "device": torch.cuda.get_device_name(),
            "input_seq_length": config.input_seq_length,
            "text_tokens": int(text_lengths.item()),
            "attention_backend": args.attention_backend,
            "linear_kernel": args.linear_kernel,
            "cuda_graph": not args.no_cuda_graph,
            "warmup": args.warmup,
            "iterations": args.iters,
            "latency_ms": {
                "mean": statistics.mean(latencies_ms),
                "p50": statistics.median(latencies_ms),
                "p95": percentile(latencies_ms, 95),
                "min": min(latencies_ms),
                "max": max(latencies_ms),
                "std": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
            },
        }
        result["throughput_hz"] = 1000.0 / result["latency_ms"]["mean"]
        print(json.dumps(result, indent=2, ensure_ascii=True))
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(result, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        return 0
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
