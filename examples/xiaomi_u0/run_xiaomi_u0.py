"""Generate a fixed-size U0 image through the registered PhyAI engine."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from phyai_utils_tools.models.xiaomi_u0 import (
    U0TextProcessor,
    image_from_pixels,
    reference_image_pixels,
    visual_codes,
)


def engine_config(dtype: torch.dtype):
    """Configure one CUDA device for native U0 inference."""
    from phyai.engine_config import EngineConfig

    config = EngineConfig.from_env(base=None)
    return config.replace(
        device=replace(config.device, target="cuda:0", params_dtype=dtype),
        runtime=replace(
            config.runtime, use_cuda_graph=False, freeze_kernel_choices=False
        ),
    )


def encode_reference(image_path: Path, checkpoint: Path) -> torch.Tensor:
    """Encode the prepared reference and release the vision model before generation."""
    from phyai.engine import EngineArgs, EngineCore
    from phyai.models.xiaomi_u0 import U0VisionArgs

    pixels = reference_image_pixels(image_path)
    core = EngineCore(
        EngineArgs(
            plugin="xiaomi_u0_encode",
            config=engine_config(torch.float32),
            plugin_args=U0VisionArgs(checkpoint_dir=str(checkpoint)),
        )
    )
    try:
        codes = core.step(pixels).squeeze(0).cpu()
    finally:
        core.close()
    del core
    gc.collect()
    torch.cuda.empty_cache()
    return codes


def main() -> None:
    """Parse local checkpoint paths, generate tokens and save a decoded image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--vision-tokenizer-dir", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--reference-image", type=Path, help="Edit this RGB image using the prompt."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=4169)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 4169:
        parser.error("A complete fixed-grid image requires max-new-tokens >= 4169.")
    if (
        args.reference_image is not None or not args.prepare_only
    ) and args.vision_tokenizer_dir is None:
        parser.error(
            "--vision-tokenizer-dir is required to encode or generate an image."
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use an empty output directory to preserve earlier results.")
    processor = U0TextProcessor(args.checkpoint_dir)
    reference_codes = None
    if args.reference_image is None:
        ids, unc = processor.build_inputs(args.prompt)
    else:
        reference_codes = encode_reference(
            args.reference_image, args.vision_tokenizer_dir
        )
        ids, unc = processor.build_x2i_inputs(args.prompt, reference_codes)
    mapping = processor.header_mapping()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "input_ids.npy", ids.numpy())
    np.save(args.output_dir / "unconditional_ids.npy", unc.numpy())
    (args.output_dir / "tokenizer_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n"
    )
    record = {
        "prompt": args.prompt,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "checkpoint_dir": str(args.checkpoint_dir),
        "vision_tokenizer_dir": str(args.vision_tokenizer_dir)
        if args.vision_tokenizer_dir
        else None,
        "backend": "flashinfer",
        "vision_backend": "phyai",
        "input_tokens": ids.shape[1],
        "unconditional_tokens": unc.shape[1],
        "torch": torch.__version__,
        "prepared_only": args.prepare_only,
    }
    if reference_codes is not None:
        np.save(args.output_dir / "reference_codes.npy", reference_codes.numpy())
        record["reference_image"] = str(args.reference_image)
        record["reference_grid_shape"] = list(reference_codes.shape)
    report_path = args.output_dir / "generation.json"
    report_path.write_text(json.dumps(record, indent=2) + "\n")
    if args.prepare_only:
        print(
            f"Prepared input: {ids.shape[1]} conditional, {unc.shape[1]} unconditional tokens."
        )
        return

    from phyai.engine import EngineArgs, EngineCore
    from phyai.models.xiaomi_u0 import UNISArgs, UNISGenerationRequest, U0VisionArgs

    cfg = engine_config(torch.bfloat16)
    started = time.perf_counter()
    core = EngineCore(
        EngineArgs(
            plugin="xiaomi_u0",
            config=cfg,
            plugin_args=UNISArgs(
                checkpoint_dir=str(args.checkpoint_dir),
                tokenizer_mapping=mapping,
                capacity=max(ids.shape[1], unc.shape[1]) + args.max_new_tokens,
                paged_backend="flashinfer",
            ),
        )
    )
    torch.cuda.synchronize()
    record["engine_setup_seconds"] = time.perf_counter() - started
    try:
        started = time.perf_counter()
        result = core.step(
            UNISGenerationRequest(
                input_ids=ids,
                unconditional_ids=unc,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
            )
        )
        torch.cuda.synchronize()
        record["generation_seconds_including_prefill"] = time.perf_counter() - started
        np.save(args.output_dir / "generated_ids.npy", result.generated_ids.numpy())
        record.update(
            stop_reason=result.stop_reason,
            generated_tokens=result.generated_ids.numel(),
            conditional_cached_tokens=result.conditional_cached_tokens,
            unconditional_cached_tokens=result.unconditional_cached_tokens,
        )
        report_path.write_text(json.dumps(record, indent=2) + "\n")
        if result.stop_reason != "eos" or result.replay:
            raise RuntimeError("Generation did not finish with a sampled EOS.")
        visual_codes(result.generated_ids)
    finally:
        core.close()
    del core
    gc.collect()
    torch.cuda.empty_cache()

    started = time.perf_counter()
    vision_cfg = cfg.replace(device=replace(cfg.device, params_dtype=torch.float32))
    vision_core = EngineCore(
        EngineArgs(
            plugin="xiaomi_u0_decode",
            config=vision_cfg,
            plugin_args=U0VisionArgs(checkpoint_dir=str(args.vision_tokenizer_dir)),
        )
    )
    try:
        codes = visual_codes(result.generated_ids).unsqueeze(0)
        pixels = vision_core.step(codes)
        image = image_from_pixels(pixels)
        image.save(args.output_dir / "image.png")
        torch.cuda.synchronize()
    finally:
        vision_core.close()
    record["vision_load_and_decode_seconds"] = time.perf_counter() - started
    record["image_size"] = list(image.size)
    report_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved image: {args.output_dir / 'image.png'}")


if __name__ == "__main__":
    main()
