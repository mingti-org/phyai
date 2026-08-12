"""Run LingBot V2 inference on concrete camera images.

The released policy consumes three ordered camera views:

1. camera_top
2. camera_wrist_left
3. camera_wrist_right

For a pipeline-only check, one image can be repeated into all three camera
slots with ``--repeat-single-image``. The resulting action is not meaningful
for robot control because the missing viewpoints are not reconstructed.

Examples
--------
Three-camera inference with a real robot state::

    python3 examples/lingbot_v2/run_lingbot_v2.py \
        --checkpoint /workspace/models/lingbot-vla-v2-6b \
        --processor /workspace/models/Qwen3-VL-4B-Instruct-processor \
        --image /workspace/data/camera_top.png \
                /workspace/data/camera_wrist_left.png \
                /workspace/data/camera_wrist_right.png \
        --state-file /workspace/data/state.json \
        --stats-json /workspace/data/dataset_stats.json \
        --task "pick up the red cup" \
        --output /workspace/data/lingbot_v2_actions.pt

Single-image pipeline check::

    python3 examples/lingbot_v2/run_lingbot_v2.py \
        --checkpoint /workspace/models/lingbot-vla-v2-6b \
        --processor /workspace/models/Qwen3-VL-4B-Instruct-processor \
        --image /workspace/data/test.png \
        --repeat-single-image \
        --zero-state \
        --task "pick up the object" \
        --output /workspace/data/lingbot_v2_single_image_actions.pt

Conv3D is the released PatchEmbed operator and the default. On targets where
the required BF16 Conv3D engine is unavailable or slow, select the validated
mathematically equivalent path with ``--patch-embed-backend gemm``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.lingbot_v2 import (
    LingBotV2Args,
    LingBotV2Request,
    LingBotVLA2Config,
)
from phyai.utils import load_config
from phyai_utils_tools.models.lingbot_v2 import LingBotV2Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processor", type=str, required=True)
    parser.add_argument(
        "--image",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One image with --repeat-single-image, or three images ordered as "
            "camera_top, camera_wrist_left, camera_wrist_right."
        ),
    )
    parser.add_argument(
        "--repeat-single-image",
        action="store_true",
        help="Repeat one image into all three camera slots for a pipeline check.",
    )
    state_group = parser.add_mutually_exclusive_group(required=True)
    state_group.add_argument(
        "--state-file",
        type=Path,
        help="JSON or NPY file containing one already ordered robot state vector.",
    )
    state_group.add_argument(
        "--zero-state",
        action="store_true",
        help="Use an all-zero state for a pipeline check, not robot control.",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help=(
            "Dataset normalization statistics. Without this file, state and "
            "action values remain in model space."
        ),
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--linear-kernel",
        choices=("torch", "flashinfer"),
        default="torch",
        help=(
            "PHYAI dense Linear backend. Torch matches the released F.linear "
            "path and is the validated Thor correctness setting."
        ),
    )
    parser.add_argument(
        "--patch-embed-backend",
        choices=("conv3d", "gemm"),
        default="conv3d",
        help=(
            "Vision PatchEmbed backend. Conv3D matches the released operator; "
            "GEMM is a validated compatibility optimization."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .pt file for raw and postprocessed actions.",
    )
    return parser.parse_args()


def load_camera_images(
    paths: list[Path],
    *,
    repeat_single_image: bool,
) -> tuple[list[torch.Tensor], list[Path], list[tuple[int, int]]]:
    if len(paths) == 1:
        if not repeat_single_image:
            raise ValueError(
                "LingBot V2 expects three cameras. Pass three --image paths or "
                "add --repeat-single-image for a pipeline-only check."
            )
        paths = paths * 3
    elif len(paths) != 3:
        raise ValueError(f"expected one or three image paths, got {len(paths)}.")
    elif repeat_single_image:
        raise ValueError("--repeat-single-image is only valid with one image path.")

    images: list[torch.Tensor] = []
    sizes: list[tuple[int, int]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"image does not exist: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            sizes.append(rgb.size)
            array = np.asarray(rgb).copy()
        images.append(torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0))
    return images, paths, sizes


def load_state(path: Path | None, *, max_state_dim: int) -> torch.Tensor:
    if path is None:
        return torch.zeros(1, max_state_dim, dtype=torch.float32)
    if not path.is_file():
        raise FileNotFoundError(f"state file does not exist: {path}")

    if path.suffix.lower() == ".json":
        value: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("state", "observation.state"):
                if key in value:
                    value = value[key]
                    break
            else:
                raise ValueError(
                    "state JSON object must contain `state` or `observation.state`."
                )
        state = torch.as_tensor(value, dtype=torch.float32)
    elif path.suffix.lower() == ".npy":
        state = torch.from_numpy(np.load(path, allow_pickle=False)).to(torch.float32)
    else:
        raise ValueError("state file must use the .json or .npy extension.")

    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim != 2 or state.shape[0] != 1:
        raise ValueError(
            f"state must describe one vector with shape (D,) or (1,D), got {tuple(state.shape)}."
        )
    if state.shape[1] > max_state_dim:
        raise ValueError(
            f"state width {state.shape[1]} exceeds max_state_dim={max_state_dim}."
        )
    return state


def load_stats(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"stats file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("stats JSON must contain an object.")
    return value


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = load_config(args.checkpoint, LingBotVLA2Config)
    images, image_paths, image_sizes = load_camera_images(
        args.image,
        repeat_single_image=args.repeat_single_image,
    )
    state = load_state(args.state_file, max_state_dim=config.max_state_dim)
    dataset_stats = load_stats(args.stats_json)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(
        1,
        config.chunk_size,
        config.max_action_dim,
        generator=generator,
        dtype=torch.float32,
    )
    processor = LingBotV2Processor(
        processor_name=args.processor,
        num_images=3,
        num_channels=config.vision.in_channels,
        patch_vector_dim=config.vision.patch_vector_dim,
        tokenizer_max_length=config.tokenizer_max_length,
        max_state_dim=config.max_state_dim,
        action_dim=config.action_dim,
        dataset_stats=dataset_stats,
        device=device,
        params_dtype=torch.bfloat16,
    )
    processed = processor.preprocess(
        {
            "images": images,
            "task": [args.task],
            "state": state,
            "noise": noise,
        }
    )
    request = LingBotV2Request(**vars(processed))
    active_patch_counts = processed.image_grid_thw.prod(dim=-1)[processed.image_masks]
    max_vision_tokens = int(
        (active_patch_counts // config.vision.spatial_merge_unit).max()
    )

    print(f"torch                 : {torch.__version__}")
    print(f"cuda runtime          : {torch.version.cuda}")
    print(f"gpu                   : {torch.cuda.get_device_name(device)}")
    print(f"linear kernel         : {args.linear_kernel}")
    print(f"PatchEmbed backend    : {args.patch_embed_backend}")
    print(f"task                  : {args.task}")
    for index, (path, size) in enumerate(zip(image_paths, image_sizes)):
        print(f"camera[{index}]             : {path} ({size[0]}x{size[1]})")
    print(f"state input width     : {state.shape[1]}")
    print(f"dataset stats         : {args.stats_json or 'none (model-space output)'}")
    print(f"pixel_values          : {tuple(processed.pixel_values.shape)}")
    print(f"image_grid_thw        : {processed.image_grid_thw.cpu().tolist()}")
    print(f"input_ids             : {tuple(processed.input_ids.shape)}")
    print(f"vision token capacity : {max_vision_tokens}")

    engine = Engine(
        EngineArgs(
            plugin="lingbot_v2",
            plugin_args=LingBotV2Args(
                checkpoint_dir=args.checkpoint,
                config=config,
                max_batch_size=1,
                num_images=3,
                max_vision_tokens_per_image=max_vision_tokens,
                weight_strict=True,
                vision_patch_embed_backend=args.patch_embed_backend,
            ),
            config=EngineConfig(
                device=DeviceConfig(
                    target=str(device),
                    params_dtype=torch.bfloat16,
                ),
                runtime=RuntimeConfig(
                    use_cuda_graph=True,
                    force_linear_kernel=args.linear_kernel,
                ),
            ),
        )
    )
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                engine.step(request)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            raw_actions = engine.step(request)
            torch.cuda.synchronize(device)
            latency_ms = (time.perf_counter() - start) * 1000.0
        actions = processor.postprocess(raw_actions)
    finally:
        engine.close()

    if not bool(torch.isfinite(actions).all()):
        raise RuntimeError("inference returned non-finite actions.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actions": actions,
            "raw_actions": raw_actions.detach().float().cpu(),
            "pixel_values": processed.pixel_values.detach().cpu(),
            "image_grid_thw": processed.image_grid_thw.detach().cpu(),
            "image_masks": processed.image_masks.detach().cpu(),
            "input_ids": processed.input_ids.detach().cpu(),
            "lang_lens": processed.lang_lens.detach().cpu(),
            "model_state": processed.state.detach().cpu(),
            "model_noise": (
                None if processed.noise is None else processed.noise.detach().cpu()
            ),
            "task": args.task,
            "image_paths": [str(path) for path in image_paths],
            "state": state,
            "seed": args.seed,
            "used_dataset_stats": dataset_stats is not None,
        },
        args.output,
    )

    print(f"raw action            : {tuple(raw_actions.shape)} {raw_actions.dtype}")
    print(f"final action          : {tuple(actions.shape)} {actions.dtype}")
    print(f"all finite            : {bool(torch.isfinite(actions).all())}")
    print(f"action mean/std       : {actions.mean():.6f} / {actions.std():.6f}")
    print(f"first action          : {actions[0, 0].tolist()}")
    print(f"latency               : {latency_ms:.3f} ms")
    print(
        "peak CUDA allocated   : "
        f"{torch.cuda.max_memory_allocated(device) / 2**30:.3f} GiB"
    )
    print(f"saved                 : {args.output}")


if __name__ == "__main__":
    main()
