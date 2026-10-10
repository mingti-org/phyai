"""Run MolmoAct2 continuous actions from camera frames, a task, and robot state."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from phyai_utils_tools.models.molmoact2 import MolmoAct2Processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--norm-tag",
        required=True,
        help="Robot normalization tag from checkpoint norm_stats.json.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="Camera frame; repeat in the order required by the selected robot tag.",
    )
    parser.add_argument(
        "--state", type=Path, help="Robot state stored as a NumPy .npy array."
    )
    parser.add_argument(
        "--task", default="pick up the red block and place it in the bowl"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic images and the tag's mean state for a smoke run.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument(
        "--save-actions", type=Path, default=None, help="Output action array (.npy)."
    )
    args = parser.parse_args()
    if args.synthetic and (args.image or args.state is not None):
        parser.error("--synthetic cannot be combined with --image or --state.")
    if not args.synthetic and (not args.image or args.state is None):
        parser.error("Provide --image and --state, or select --synthetic.")

    from phyai.engine import Engine, EngineArgs
    from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
    from phyai.models.molmoact2 import MolmoAct2Args, MolmoAct2Request
    from phyai.utils import get_logger

    logger = get_logger(__name__)
    processor = MolmoAct2Processor.from_pretrained(
        args.checkpoint,
        norm_tag=args.norm_tag,
        n_action_steps=args.n_action_steps,
    )
    if args.synthetic:
        image_count = len(processor.metadata.get("camera_keys") or []) or 1
        rng = np.random.default_rng(args.seed)
        images = [
            rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
            for _ in range(image_count)
        ]
        state_stats = processor.metadata.get("state_stats") or {}
        if "mean" not in state_stats:
            raise ValueError("Synthetic observations require state mean statistics.")
        state = np.asarray(state_stats["mean"], dtype=np.float32)
        logger.info_rank0("Using synthetic observations for this smoke run.")
    else:
        images = []
        for path in args.image:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB")))
        state = np.load(args.state, allow_pickle=False)
    processed = processor.preprocess(
        {"images": images, "task": args.task, "state": state}
    )
    request = MolmoAct2Request(
        inputs=processed.tensors,
        action_horizon=processed.action_horizon,
        num_steps=args.num_steps,
        seed=args.seed,
    )
    engine = Engine(
        EngineArgs(
            plugin="molmoact2",
            plugin_args=MolmoAct2Args(checkpoint_dir=args.checkpoint),
            config=EngineConfig(
                device=DeviceConfig(target=args.device, params_dtype=torch.bfloat16),
                runtime=RuntimeConfig(use_cuda_graph=not args.no_cuda_graph),
            ),
        )
    )
    try:
        normalized_actions = engine.step(request)
        actions = processor.postprocess(normalized_actions)
    finally:
        engine.close()
    logger.info_rank0(
        "Action chunk shape=%s dtype=%s finite=%s",
        tuple(actions.shape),
        actions.dtype,
        bool(torch.isfinite(actions).all()),
    )
    logger.info_rank0("First action: %s", actions[0, 0].tolist())
    if args.save_actions is not None:
        args.save_actions.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_actions, actions.numpy())
        logger.info_rank0("Saved actions to %s", args.save_actions)


if __name__ == "__main__":
    main()
