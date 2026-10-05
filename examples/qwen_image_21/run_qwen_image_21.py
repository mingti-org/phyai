"""Generate or edit an RGBA image with the Qwen-Image 2.1 engine plugin."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from phyai import DeploymentConfig, Engine, EngineArgs
from phyai.engine_config import (
    AttentionParallelConfig,
    DenseParallelConfig,
    DeviceConfig,
    EngineConfig,
    KernelConfig,
    OuterParallelConfig,
    ParallelConfig,
    RuntimeConfig,
)
from phyai.models.qwen_image_21 import QwenImage21Request
from phyai.models.qwen_image_21.main_qwen_image_21 import QwenImage21Args
from phyai.server import WorkerSupervisorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--negative-prompt")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cfg", type=int, choices=(1, 2), default=1)
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument(
        "--kernel-policy", type=Path, help="Path to a kernel policy YAML file."
    )
    parser.add_argument("--out", type=Path, default=Path(".cache/qwen_image_21.png"))
    args = parser.parse_args()
    images = [Image.open(path).convert("RGBA") for path in args.image]
    engine = Engine(
        EngineArgs(
            plugin="qwen_image_21",
            plugin_args=QwenImage21Args(
                checkpoint_dir=args.checkpoint,
                use_kv_cache=not args.disable_kv_cache,
                vae_tiling=args.vae_tiling,
                torch_compile=args.torch_compile,
            ),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                kernel=KernelConfig(
                    config_path=str(args.kernel_policy.expanduser().resolve())
                    if args.kernel_policy is not None
                    else None,
                ),
                parallel=ParallelConfig(
                    outer=OuterParallelConfig(cfg_size=args.cfg),
                    dense=DenseParallelConfig(tp_size=args.tp),
                    attention=AttentionParallelConfig(tp_size=args.tp),
                ),
                runtime=RuntimeConfig(use_cuda_graph=not args.disable_cuda_graph),
            ),
        ),
        deployment=DeploymentConfig(
            process_config=WorkerSupervisorConfig(startup_timeout_s=1800)
        ),
    )
    output = None
    try:
        output = engine.step(
            QwenImage21Request(
                prompt=args.prompt,
                image=images or None,
                negative_prompt=args.negative_prompt,
                true_cfg_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                seed=args.seed,
                output_type="pil",
            )
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        output.images[0].save(args.out)
    finally:
        # Release CUDA IPC tensors before stopping the workers that own them.
        output = None
        engine.close()


if __name__ == "__main__":
    main()
