"""Run a converted RLinf Tianji pi0.5 checkpoint with the existing PI05 engine.

Input NPZ: RGB uint8 HWC images ``head_left``, ``left_wrist``, ``right_wrist``,
a raw 16-dimensional ``state``, a scalar string ``task``, and optionally
``noise`` of shape (1, 50, 32). Output actions are absolute joint targets;
gripper coordinates remain absolute.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from phyai import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.kernel.config import KernelConfig
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_pi05 import PI05Request
from phyai.utils import get_logger, load_config
from phyai_utils_tools.models.pi05 import PI05Processor
from phyai_utils_tools.processing.steps import NormalizerStep, UnnormalizerStep

logger = get_logger(__name__)


def load_processor(checkpoint: Path) -> tuple[PI05Processor, dict]:
    metadata = json.loads((checkpoint / "rlinf_metadata.json").read_text())
    stats = json.loads((checkpoint / "norm_stats.json").read_text())["norm_stats"]
    processor = PI05Processor(
        tokenizer_name=str(checkpoint),
        dataset_stats={"observation.state": stats["state"], "action": stats["actions"]},
        action_dim=metadata["action_dim"],
        normalize_pixels=True,
        params_dtype=torch.float32,
    )
    # OpenPI uses NumPy float64 stats and epsilon 1e-6.
    for pipeline in (processor.preprocessor, processor.postprocessor):
        pipeline.steps = [
            replace(step, eps=metadata["norm_eps"], dtype=torch.float64)
            if isinstance(step, (NormalizerStep, UnnormalizerStep))
            else step
            for step in pipeline.steps
        ]
    return processor, metadata


def prepare_observation(processor: PI05Processor, metadata: dict, payload):
    images = []
    for camera in metadata["camera_names"]:
        image = np.asarray(payload[camera])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{camera} must be an RGB uint8 HWC image")
        height, width = image.shape[:2]
        size = processor.image_size
        ratio = max(width / size, height / size)
        resized_size = (int(width / ratio), int(height / ratio))
        resized = Image.fromarray(image).resize(resized_size, Image.Resampling.BILINEAR)
        padded = Image.new("RGB", (size, size))
        padded.paste(
            resized, ((size - resized.width) // 2, (size - resized.height) // 2)
        )
        images.append(
            torch.from_numpy(np.array(padded)).permute(2, 0, 1)[None].float() / 255.0
        )
    state = np.asarray(payload["state"], dtype=np.float32)
    if state.shape != (metadata["action_dim"],):
        raise ValueError(
            f"Expected state shape {(metadata['action_dim'],)}, got {state.shape}"
        )
    return processor.preprocess(
        {
            "images": images,
            "task": str(np.asarray(payload["task"]).item()),
            "state": torch.from_numpy(state.copy())[None],
        }
    )


def absolute_actions(
    processor: PI05Processor, metadata: dict, normalized: torch.Tensor, state
) -> torch.Tensor:
    # Slice padding before the existing processor applies the 16-dimensional stats.
    actions = processor.postprocess(
        normalized[..., : metadata["action_dim"]].float().cpu()
    )
    anchor = torch.as_tensor(np.asarray(state), dtype=actions.dtype)
    mask = torch.tensor(metadata["delta_mask"], dtype=torch.bool)
    return actions + torch.where(mask, anchor, 0)[..., None, :]


def make_engine(
    checkpoint: Path, *, use_cuda_graph: bool = True, max_batch_size: int = 1
) -> Engine:
    return Engine(
        EngineArgs(
            plugin="pi05",
            plugin_args=PI05Args(
                checkpoint_dir=checkpoint, max_batch_size=max_batch_size
            ),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                runtime=RuntimeConfig(use_cuda_graph=use_cuda_graph, num_threads=16),
                kernel=KernelConfig(
                    config_path=str(
                        Path(__file__).with_name("kernel_policy_rlinf_bf16.yaml")
                    )
                ),
            ),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()
    processor, metadata = load_processor(args.checkpoint)
    config = load_config(args.checkpoint, PI05Config)
    with np.load(args.input, allow_pickle=False) as payload:
        state = payload["state"].copy()
        processed = prepare_observation(processor, metadata, payload)
        noise = (
            torch.from_numpy(payload["noise"].copy())
            if "noise" in payload
            else torch.randn(
                1,
                config.chunk_size,
                config.max_action_dim,
                generator=torch.Generator().manual_seed(args.seed),
                dtype=torch.float32,
            )
        )
    engine = make_engine(args.checkpoint, use_cuda_graph=not args.no_cuda_graph)
    try:
        with torch.inference_mode():
            normalized = engine.step(
                PI05Request(
                    pixel_values=processed.pixel_values,
                    input_ids=processed.input_ids,
                    lang_lens=processed.lang_lens,
                    noise=noise,
                )
            )
        actions = absolute_actions(processor, metadata, normalized, state)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            actions=actions.numpy(),
            normalized_actions=normalized.float().cpu().numpy(),
        )
        logger.info_rank0(
            "Saved absolute joint targets %s to %s", tuple(actions.shape), args.output
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
