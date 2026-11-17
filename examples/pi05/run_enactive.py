"""Infer absolute joint targets from a converted Enactive pi0.5 checkpoint.

The input NPZ contains RGB uint8 arrays ``front``, ``left``, ``right`` (HWC),
the raw 14-dimensional joint ``state``, and a scalar string ``task``. An
optional ``noise`` array of shape (1, 50, 32) enables paired comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from phyai import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_pi05 import PI05Request
from phyai.utils import get_logger, load_config
from phyai_utils_tools.models.pi05 import PI05Processor

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    processor = PI05Processor.from_pretrained(
        args.checkpoint,
        tokenizer_name=str(args.checkpoint),
        image_resize_backend="pil",
        normalize_pixels=True,
        action_dim=14,
        params_dtype=torch.float32,
    )
    config = load_config(args.checkpoint, PI05Config)
    with np.load(args.input, allow_pickle=False) as payload:
        raw_state = payload["state"].copy()
        processed = processor.preprocess(
            {
                "images": [
                    torch.from_numpy(payload[view].copy()).permute(2, 0, 1).unsqueeze(0)
                    for view in ("front", "left", "right")
                ],
                "task": str(payload["task"].item()),
                "state": torch.from_numpy(raw_state).unsqueeze(0),
            }
        )
        noise = (
            torch.from_numpy(payload["noise"].copy())
            if "noise" in payload
            else torch.randn(
                1,
                config.chunk_size,
                config.max_action_dim,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
                dtype=torch.float32,
            )
        )

    engine = Engine(
        EngineArgs(
            plugin="pi05",
            plugin_args=PI05Args(checkpoint_dir=args.checkpoint),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                runtime=RuntimeConfig(use_cuda_graph=not args.no_cuda_graph),
            ),
        )
    )
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
        actions = processor.postprocess({"action": normalized, "state": raw_state})
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
