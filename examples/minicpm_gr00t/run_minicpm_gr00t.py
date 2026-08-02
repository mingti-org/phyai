"""Run MiniCPM-GR00T end-to-end from an original PTH checkpoint.

Demonstrates the full flow: a
``phyai_utils_tools.models.minicpm_gr00t.MiniCPMGR00TProcessor`` turns two raw
camera frames + a task instruction + an 80-D state into the canonical
``MiniCPMGR00TRequest`` tensors, the engine runs, and the processor's
``postprocess`` returns the CPU float32 action chunk.

Example::

    uv run python examples/minicpm_gr00t/run_minicpm_gr00t.py \
      --checkpoint /path/to/checkpoint.pth \
      --vlm-path /path/to/MiniCPM-V-4.6 \
      --instruction "open the middle drawer of the cabinet" \
      --image /path/to/base_camera.png --image /path/to/wrist_camera.png \
      --seed 123 \
      --save-actions /tmp/minicpm_gr00t_actions.pt

Without ``--image`` two blank 224x224 frames are used, which keeps the run
fully deterministic for a given ``--seed`` and makes results reproducible
across machines (useful for numerical comparisons against the reference
implementation).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from phyai.engine import Engine, EngineArgs
from phyai.models.minicpm_gr00t.main_minicpm_gr00t import MiniCPMGR00TArgs
from phyai.models.minicpm_gr00t.scheduler_ws1_minicpm_gr00t import (
    MiniCPMGR00TRequest,
)
from phyai_utils_tools.models.minicpm_gr00t import (
    MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE,
    MiniCPMGR00TProcessor,
)


def load_raw_images(paths: list[Path], size: int) -> list[np.ndarray]:
    """Return two raw HWC uint8 frames; the processor owns the model resize."""
    if not paths:
        blank = np.zeros((size, size, 3), dtype=np.uint8)
        return [blank, blank.copy()]
    if len(paths) != 2:
        raise ValueError(f"Expected exactly two --image paths, got {len(paths)}.")
    return [np.asarray(Image.open(path).convert("RGB")) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the original PTH checkpoint (safetensors also accepted).",
    )
    parser.add_argument(
        "--vlm-path",
        type=Path,
        required=True,
        help="Path to the MiniCPM-V processor/tokenizer directory.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="Camera frame path; pass exactly twice (base + wrist). "
        "Omit to use two blank frames.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square size each camera frame is resized to.",
    )
    parser.add_argument(
        "--instruction",
        default="open the middle drawer of the cabinet",
        help="Task instruction inserted into the prompt template.",
    )
    parser.add_argument(
        "--prompt-template",
        default=MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template; must contain an {instruction} placeholder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for the flow-matching starting noise.",
    )
    parser.add_argument(
        "--save-actions",
        type=Path,
        default=None,
        help="Optional path to save the predicted actions tensor (.pt).",
    )
    args = parser.parse_args()

    processor = MiniCPMGR00TProcessor(
        processor=AutoProcessor.from_pretrained(args.vlm_path),
        processor_name=str(args.vlm_path),
        image_size=(args.image_size, args.image_size),
        prompt_template=args.prompt_template,
    )
    processed = processor.preprocess(
        {
            "images": load_raw_images(args.image, args.image_size),
            "task": args.instruction,
            "state": np.zeros(80, dtype=np.float32),
        }
    )
    generator = torch.Generator().manual_seed(args.seed)
    request = MiniCPMGR00TRequest(
        input_ids=processed.input_ids,
        attention_mask=processed.attention_mask,
        pixel_values=processed.pixel_values,
        target_sizes=processed.target_sizes,
        state=processed.state,
        noise=torch.randn(1, 30, 80, generator=generator),
    )
    engine = Engine(
        EngineArgs(
            plugin="minicpm_gr00t",
            plugin_args=MiniCPMGR00TArgs(checkpoint=args.checkpoint),
        )
    )
    try:
        start = time.perf_counter()
        actions = engine.step(request)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        engine.close()
    actions = processor.postprocess(actions)

    print(
        f"actions={tuple(actions.shape)} dtype={actions.dtype} "
        f"finite={bool(torch.isfinite(actions).all())} elapsed={elapsed:.3f}s"
    )
    print(
        f"stats min={actions.min().item():.6f} max={actions.max().item():.6f} "
        f"mean={actions.mean().item():.6f} std={actions.std().item():.6f}"
    )
    print("step0 first10 dims:", actions[0, 0, :10].tolist())
    if args.save_actions is not None:
        args.save_actions.parent.mkdir(parents=True, exist_ok=True)
        torch.save(actions.cpu(), args.save_actions)
        print(f"actions saved to {args.save_actions}")


if __name__ == "__main__":
    main()
