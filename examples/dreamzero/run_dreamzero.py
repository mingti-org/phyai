"""Run one initial DreamZero-DROID action chunk from three RGB images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.kernel.config import KernelConfig
from phyai.models.dreamzero.main_dreamzero import DreamZeroArgs, DreamZeroEntry
from phyai_utils_tools.models.dreamzero import DreamZeroProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: uv run python examples/dreamzero/run_dreamzero.py "
            "--checkpoint /path/to/DreamZero-DROID "
            "--tokenizer /path/to/umt5-xxl "
            "--right-image right.png --left-image left.png --wrist-image wrist.png "
            "--state 0 0 0 0 0 0 0 0 --task 'Pick up the object' "
            "--sequential-cpu-offload --output action.npy. "
            "Replace the sample state with seven measured joint positions and "
            "one gripper position. This runs a fresh sequence, not a robot control loop. "
            "CPU offload reduces memory use but does not guarantee the model fits."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="UMT5 tokenizer directory or Hugging Face ID.",
    )
    parser.add_argument("--right-image", type=Path, required=True)
    parser.add_argument("--left-image", type=Path, required=True)
    parser.add_argument("--wrist-image", type=Path, required=True)
    parser.add_argument(
        "--state",
        type=float,
        nargs=8,
        required=True,
        help="Seven DROID joint positions followed by gripper position.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--device", default="cuda:0", help="CUDA device within CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument("--seed", type=int, default=1140)
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Fixed denoising steps (default: 4; 1 is a smoke test only).",
    )
    parser.add_argument(
        "--dynamic-dit",
        action="store_true",
        help="Use the 16-step dynamic schedule instead of --steps.",
    )
    parser.add_argument("--sequential-cpu-offload", action="store_true")
    parser.add_argument(
        "--official-attention",
        action="store_true",
        help="Use cuDNN-prioritized SDPA self-attention and official FlashAttention 2 cross-attention.",
    )
    parser.add_argument(
        "--compile-scheduler",
        action="store_true",
        help="Compile UniPC updates to match the official BF16 rounding path.",
    )
    parser.add_argument("--kernel-config", type=str, help="Kernel policy YAML path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("action.npy"),
        help="Save the postprocessed [1, 24, 8] action array.",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        parser.error("--checkpoint must be an existing DreamZero-DROID directory.")
    if args.steps <= 0:
        parser.error("--steps must be positive.")
    if not args.task.strip():
        parser.error("--task must not be empty.")
    if not np.isfinite(args.state).all():
        parser.error("--state must contain only finite values.")
    if torch.device(args.device).type != "cuda":
        parser.error("This example requires a CUDA device.")
    return args


def load_observation(args: argparse.Namespace) -> dict:
    images = []
    for path in (args.right_image, args.left_image, args.wrist_image):
        with Image.open(path) as image:
            images.append(np.array(image.convert("RGB")))
    if len({image.shape for image in images}) != 1:
        raise ValueError("All three camera images must have the same dimensions.")
    return {
        "video": np.stack(images)[None, None],  # [B, T, views, H, W, RGB]
        "state": torch.tensor(args.state, dtype=torch.float32).reshape(1, 1, 8),
        "task": [args.task],
    }


def main() -> None:
    args = parse_args()
    observation = load_observation(args)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine = Engine(
        EngineArgs(
            plugin="dreamzero",
            plugin_args=DreamZeroArgs(
                checkpoint_dir=args.checkpoint,
                seed=args.seed,
                weight_strict=True,
                sequential_cpu_offload=args.sequential_cpu_offload,
                num_inference_steps=args.steps,
                dynamic_dit=args.dynamic_dit,
                dynamic_dit_scheduler_steps=16,
                official_attention=args.official_attention,
                compile_scheduler=args.compile_scheduler,
            ),
            config=EngineConfig(
                device=DeviceConfig(target=args.device, params_dtype=torch.bfloat16),
                runtime=RuntimeConfig(use_cuda_graph=False),
                kernel=KernelConfig(config_path=args.kernel_config),
            ),
        )
    )
    try:
        if not isinstance(engine.entry, DreamZeroEntry) or engine.entry.bundle is None:
            raise RuntimeError("DreamZero engine setup did not produce a pipeline.")
        config = engine.entry.bundle.config
        processor = DreamZeroProcessor.from_pretrained(
            args.checkpoint,
            tokenizer_name=args.tokenizer,
            max_length=config.text_encoder.max_length,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            action_horizon=config.action_horizon,
            num_views=3,
        )
        with torch.inference_mode():
            processed = processor.preprocess(observation)
            output = engine.step(processed)
            action = processor.postprocess(output, obs=observation).action.float().cpu()
        if not torch.isfinite(action).all():
            raise RuntimeError("Inference returned non-finite actions.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as file:
            np.save(file, action.numpy(), allow_pickle=False)
        print(f"Saved action shape {tuple(action.shape)} to {args.output}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
