from __future__ import annotations

import argparse
import json
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
            "Require eager, CUDA-graph capture, and CUDA-graph replay to "
            "produce identical LingBot V2 actions for one fixed input."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vision-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--patch-embed-backend",
        choices=("conv3d", "gemm"),
        default="gemm",
    )
    parser.add_argument(
        "--linear-kernel",
        choices=("torch", "flashinfer"),
        default="torch",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark/lingbot_v2/cuda_graph_parity.json"),
    )
    return parser.parse_args()


def compare(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | bool]:
    left_fp32 = left.float().reshape(-1)
    right_fp32 = right.float().reshape(-1)
    return {
        "exact": bool(torch.equal(left, right)),
        "cosine": float(F.cosine_similarity(left_fp32, right_fp32, dim=0).item()),
        "max_abs": float((left_fp32 - right_fp32).abs().max().item()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA Graph parity.")
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
                vision_params_dtype=vision_dtype,
                vision_patch_embed_backend=args.patch_embed_backend,
            ),
            config=EngineConfig(
                device=DeviceConfig(target=str(device), params_dtype=params_dtype),
                runtime=RuntimeConfig(
                    use_cuda_graph=True,
                    force_linear_kernel=args.linear_kernel,
                ),
            ),
        )
    )
    try:
        runner = engine.entry.scheduler.expert_runner
        if not runner.use_cuda_graph:
            raise RuntimeError("The Expert runner rejected CUDA Graph before capture.")

        runner.use_cuda_graph = False
        eager = engine.step(request).detach().clone()
        torch.cuda.synchronize(device)

        runner.use_cuda_graph = True
        captured = engine.step(request).detach().clone()
        torch.cuda.synchronize(device)
        if not runner.cuda_graph_active:
            raise RuntimeError(
                "CUDA Graph capture fell back to eager execution; inspect logs."
            )

        replayed = engine.step(request).detach().clone()
        torch.cuda.synchronize(device)
        graph_status = {
            "active": bool(runner.cuda_graph_active),
            "captured_graphs": int(runner.cuda_graph_count),
            "fallback_layouts": int(runner.cuda_graph_fallback_count),
        }
    finally:
        engine.close()

    result = {
        "contract": contract,
        "input_metadata": input_metadata,
        "output_shape": list(eager.shape),
        "output_dtype": str(eager.dtype),
        "vision_patch_embed_backend": args.patch_embed_backend,
        "graph_status": graph_status,
        "eager_vs_capture": compare(eager, captured),
        "eager_vs_replay": compare(eager, replayed),
        "capture_vs_replay": compare(captured, replayed),
        "all_finite": bool(
            torch.isfinite(eager).all()
            and torch.isfinite(captured).all()
            and torch.isfinite(replayed).all()
        ),
    }
    passed = result["all_finite"] and all(
        result[name]["exact"]
        for name in (
            "eager_vs_capture",
            "eager_vs_replay",
            "capture_vs_replay",
        )
    )
    result["passed"] = bool(passed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Report: {args.out}")
    if not passed:
        raise RuntimeError(
            "CUDA Graph changed the LingBot V2 action output; parity failed."
        )


if __name__ == "__main__":
    main()
