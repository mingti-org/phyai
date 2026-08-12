from __future__ import annotations

import argparse
import gc
import json
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.lingbot_v2 import LingBotV2Args, LingBotVLA2Config
from phyai.utils import load_config
from profile_lingbot_v2 import load_inputs, make_request, validate_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the official-compatible BF16 Conv3D PatchEmbed with the "
            "mathematically equivalent PHYAI BF16 GEMM path."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vision-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--linear-kernel",
        choices=("torch", "flashinfer"),
        default="torch",
    )
    parser.add_argument("--cosine-threshold", type=float, default=0.99)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark/lingbot_v2/patch_embed_gemm_parity.json"),
    )
    return parser.parse_args()


def tensor_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | bool | list[int] | str]:
    shape_match = reference.shape == candidate.shape
    dtype_match = reference.dtype == candidate.dtype
    if not shape_match:
        return {
            "shape_match": False,
            "dtype_match": dtype_match,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "reference_dtype": str(reference.dtype),
            "candidate_dtype": str(candidate.dtype),
            "exact": False,
        }

    reference_fp32 = reference.float().reshape(-1)
    candidate_fp32 = candidate.float().reshape(-1)
    difference = candidate_fp32 - reference_fp32
    reference_norm = torch.linalg.vector_norm(reference_fp32)
    relative_l2 = torch.linalg.vector_norm(difference) / reference_norm.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return {
        "shape_match": True,
        "dtype_match": dtype_match,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "exact": bool(torch.equal(reference, candidate)),
        "cosine": float(
            F.cosine_similarity(reference_fp32, candidate_fp32, dim=0).item()
        ),
        "max_abs": float(difference.abs().max().item()),
        "relative_l2": float(relative_l2.item()),
    }


def run_backend(
    *,
    backend: str,
    checkpoint: Path,
    config: LingBotVLA2Config,
    request,
    max_vision_tokens: int,
    device: torch.device,
    params_dtype: torch.dtype,
    vision_dtype: torch.dtype,
    linear_kernel: str,
) -> dict[str, torch.Tensor]:
    engine = Engine(
        EngineArgs(
            plugin="lingbot_v2",
            plugin_args=LingBotV2Args(
                checkpoint_dir=checkpoint,
                config=config,
                max_batch_size=1,
                num_images=3,
                max_vision_tokens_per_image=max_vision_tokens,
                weight_strict=True,
                vision_params_dtype=vision_dtype,
                vision_patch_embed_backend=backend,
            ),
            config=EngineConfig(
                device=DeviceConfig(target=str(device), params_dtype=params_dtype),
                runtime=RuntimeConfig(
                    use_cuda_graph=False,
                    force_linear_kernel=linear_kernel,
                ),
            ),
        )
    )
    captures: dict[str, torch.Tensor] = {}

    def capture_patch(_module, _inputs, output) -> None:
        captures["patch_embed"] = output.detach().cpu()

    def capture_vision(_module, _inputs, output) -> None:
        captures["vision"] = output[0].detach().cpu()

    model = engine.entry.model
    patch_embed = model.vision.patch_embed
    if backend == "conv3d":
        original_forward = patch_embed.forward

        def forward_without_cudnn(
            _self,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            with torch.backends.cudnn.flags(enabled=False):
                return original_forward(hidden_states)

        patch_embed.forward = types.MethodType(
            forward_without_cudnn,
            patch_embed,
        )
    patch_handle = model.vision.patch_embed.register_forward_hook(capture_patch)
    vision_handle = model.vision.register_forward_hook(capture_vision)
    try:
        captures["action"] = engine.step(request).detach().cpu()
    finally:
        patch_handle.remove()
        vision_handle.remove()
        engine.close()
    del model
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return captures


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PatchEmbed GEMM parity.")
    if not 0.0 <= args.cosine_threshold <= 1.0:
        raise ValueError("--cosine-threshold must be in [0, 1].")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    params_dtype = torch.bfloat16
    vision_dtype = torch.bfloat16 if args.vision_dtype == "bf16" else torch.float32
    config = load_config(args.checkpoint, LingBotVLA2Config)
    tensors, input_metadata = load_inputs(args.input)
    contract = validate_contract(tensors, config)
    active_patch_counts = tensors["input.image_grid_thw"].prod(dim=-1)[
        tensors["input.image_masks"].bool()
    ]
    max_vision_tokens = int(
        (active_patch_counts // config.vision.spatial_merge_unit).max()
    )
    request = make_request(tensors, device=device, params_dtype=params_dtype)

    conv3d = run_backend(
        backend="conv3d",
        checkpoint=args.checkpoint,
        config=config,
        request=request,
        max_vision_tokens=max_vision_tokens,
        device=device,
        params_dtype=params_dtype,
        vision_dtype=vision_dtype,
        linear_kernel=args.linear_kernel,
    )
    gemm = run_backend(
        backend="gemm",
        checkpoint=args.checkpoint,
        config=config,
        request=request,
        max_vision_tokens=max_vision_tokens,
        device=device,
        params_dtype=params_dtype,
        vision_dtype=vision_dtype,
        linear_kernel=args.linear_kernel,
    )

    comparisons = {
        name: tensor_metrics(conv3d[name], gemm[name])
        for name in ("patch_embed", "vision", "action")
    }
    passed = all(
        metrics.get("shape_match", False)
        and metrics.get("dtype_match", False)
        and metrics.get("cosine", 0.0) >= args.cosine_threshold
        for metrics in comparisons.values()
    )
    result = {
        "contract": contract,
        "input_metadata": input_metadata,
        "vision_dtype": str(vision_dtype),
        "linear_kernel": args.linear_kernel,
        "reference_backend": "official_compatible_bf16_conv3d_cudnn_disabled",
        "candidate_backend": "phyai_replicated_linear_gemm",
        "cosine_threshold": args.cosine_threshold,
        "comparisons": comparisons,
        "passed": bool(passed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Report: {args.out}")
    if not passed:
        raise RuntimeError(
            "BF16 GEMM PatchEmbed did not meet the requested parity threshold."
        )


if __name__ == "__main__":
    main()
