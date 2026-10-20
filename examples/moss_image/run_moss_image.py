"""Generate an RGB image from a local MOSS-Image checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from phyai import DeploymentConfig, Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.moss_image import MOSSImageRequest
from phyai.models.moss_image.main_moss_image import MOSSImageArgs
from phyai.models.moss_image.model_runner_moss_image import (
    normalize_moss_image_graph_buckets,
)
from phyai.server import WorkerSupervisorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload text encoder and VAE; CUDA graphs keep the transformer on GPU and need more VRAM",
    )
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument(
        "--no-cuda-graph", action="store_true", help="Disable CUDA graph capture"
    )
    parser.add_argument("--graph-cache-size", type=int, default=4)
    parser.add_argument(
        "--graph-bucket",
        nargs=3,
        type=int,
        action="append",
        default=[],
        metavar=("HEIGHT", "WIDTH", "TEXT_TOKENS"),
        help="Prewarm an image size; text tokens round up to 64, 128, ..., 4096. "
        "Graphs are reused per sample at any batch size (two cache entries per unique bucket)",
    )
    parser.add_argument("--uncond-skip-layers", type=int, nargs="*", default=[22])
    parser.add_argument("--slg-layers", type=int, nargs="*", default=[])
    parser.add_argument("--slg-scale", type=float, default=0.0)
    parser.add_argument("--slg-start", type=float, default=0.0)
    parser.add_argument("--slg-stop", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path(".cache/moss_image.png"))
    args = parser.parse_args()
    if args.graph_cache_size < 1:
        parser.error("--graph-cache-size must be positive")
    if args.no_cuda_graph and args.graph_bucket:
        parser.error("--graph-bucket requires CUDA graphs")
    for height, width, tokens in args.graph_bucket:
        if (
            height <= 0
            or width <= 0
            or height % 16
            or width % 16
            or not 1 <= tokens <= 4096
        ):
            parser.error(
                "--graph-bucket requires positive image sizes divisible by 16 and 1..4096 text tokens"
            )
    args.graph_bucket = normalize_moss_image_graph_buckets(args.graph_bucket)
    if args.graph_cache_size < 2 * len(args.graph_bucket):
        parser.error(
            "--graph-cache-size must hold two entries per unique --graph-bucket"
        )
    request = MOSSImageRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        uncond_skip_layers=tuple(args.uncond_skip_layers),
        slg_layers=tuple(args.slg_layers),
        slg_scale=args.slg_scale,
        slg_start=args.slg_start,
        slg_stop=args.slg_stop,
    )
    engine = Engine(
        EngineArgs(
            plugin="moss_image",
            plugin_args=MOSSImageArgs(
                checkpoint_dir=args.checkpoint,
                cpu_offload=args.cpu_offload,
                torch_compile=args.torch_compile,
                vae_slicing=args.vae_slicing,
                graph_cache_size=args.graph_cache_size,
                graph_warmup_buckets=args.graph_bucket,
            ),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                runtime=RuntimeConfig(use_cuda_graph=not args.no_cuda_graph),
            ),
        ),
        deployment=DeploymentConfig(
            process_config=WorkerSupervisorConfig(startup_timeout_s=1800)
        ),
    )
    output = None
    try:
        output = engine.step(request)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        output.images[0].save(args.out)
    finally:
        output = None
        engine.close()


if __name__ == "__main__":
    main()
