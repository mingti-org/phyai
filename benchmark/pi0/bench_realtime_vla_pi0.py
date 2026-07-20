#!/usr/bin/env python3
"""realtime-vla PI0 latency benchmark using the PhyAI bench runner.

This script adapts realtime-vla's ``Pi0Inference.forward`` to the common
``benchmark/bench_n_batch.py`` harness. It accepts realtime-vla converted
``.pt``/``.pkl`` checkpoints directly, or a HuggingFace-style PI0
``model.safetensors`` checkpoint via FlashRT's tested PI0 conversion helper.

Run::

    python benchmark/pi0/bench_realtime_vla_pi0.py \
        --realtime-vla-root <REALTIME_VLA_REPO> \
        --flashrt-root <FLASHRT_REPO> \
        --checkpoint <PI0_CHECKPOINT_OR_CONVERTED_FILE> \
        --num-views 2 --chunk-size 50 --prompt-len 16 \
        --batch-sizes 1 --n-warmup 100 --n-timed 100 \
        --result-file results/realtime_vla_pi0.jsonl

Only batch size 1 is supported because realtime-vla's PI0 inference object
used here is allocated for one synthetic request.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import torch
import triton
import triton.language as tl

_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCHMARK_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _BENCHMARK_DIR.parent
sys.path.insert(0, str(_BENCHMARK_DIR))
for _src in (_REPO_ROOT / "src", _REPO_ROOT / "phyai" / "src"):
    if _src.exists():
        sys.path.insert(0, str(_src))
import bench_n_batch as bnb  # noqa: E402
from phyai.utils.profile import (  # noqa: E402
    add_profile_cli_args,
    install_profiler,
    profile_config_from_args,
)


@triton.jit
def safe_matmul_abt_scale(
    q_ptr,
    k_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    scale_factor: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 32,
    BLOCK_SIZE_N: tl.constexpr = 32,
    BLOCK_SIZE_K: tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)

    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        offs_i = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_j = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            q = tl.load(
                q_ptr + offs_i[:, None] * K + offs_k[None, :],
                mask=(offs_i[:, None] < M) & (offs_k[None, :] < K),
                other=0,
            )
            kk = tl.load(
                k_ptr + offs_j[:, None] * K + offs_k[None, :],
                mask=(offs_j[:, None] < N) & (offs_k[None, :] < K),
                other=0,
            )
            accumulator = tl.dot(q, tl.trans(kk), accumulator)
        accumulator = accumulator * scale_factor
        tl.store(
            out_ptr + offs_i[:, None] * N + offs_j[None, :],
            accumulator.to(tl.bfloat16),
            mask=(offs_i[:, None] < M) & (offs_j[None, :] < N),
        )
        pid += psize


def patch_realtime_vla_pi0_kernels(realtime_vla_pi0) -> None:
    """Patch a realtime-vla PI0 attention GEMM kernel with safe edge masks."""
    realtime_vla_pi0.matmul_abT_scale = safe_matmul_abt_scale


DIRECT_PI0_KEYS = (
    "vision_patch_embedding_w",
    "vision_patch_embedding_b",
    "vision_position_embedding",
    "vision_attn_qkv_w",
    "vision_attn_qkv_b",
    "vision_attn_o_w",
    "vision_attn_o_b",
    "vision_ffn_up_w",
    "vision_ffn_up_b",
    "vision_ffn_down_w",
    "vision_ffn_down_b",
    "vision_pre_attn_norm_w",
    "vision_pre_attn_norm_b",
    "vision_pre_ffn_norm_w",
    "vision_pre_ffn_norm_b",
    "vision_final_norm_w",
    "vision_final_norm_b",
    "encoder_multi_modal_projector_w",
    "encoder_multi_modal_projector_b",
    "encoder_attn_qkv_w",
    "encoder_attn_o_w",
    "encoder_ffn_gate_w",
    "encoder_ffn_up_w",
    "encoder_ffn_down_w",
    "decoder_attn_qkv_w",
    "decoder_attn_o_w",
    "decoder_ffn_gate_w",
    "decoder_ffn_up_w",
    "decoder_ffn_down_w",
)

REALTIME_PI0_REQUIRED_KEYS = DIRECT_PI0_KEYS + (
    "decoder_state_in_proj_w",
    "decoder_state_in_proj_b",
    "decoder_action_fused_in_proj_w",
    "decoder_action_fused_time_biases",
    "decoder_action_mlp_w",
    "decoder_action_mlp_b",
    "decoder_action_fused_out_proj_w",
    "decoder_action_fused_out_proj_b",
    "language_embeds",
)
REALTIME_PI0_REQUIRED_KEY_SET = frozenset(REALTIME_PI0_REQUIRED_KEYS)


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _bf16_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()


def _sinusoidal_time_embedding(
    time_value: float, dimension: int = 1024
) -> torch.Tensor:
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    scaling_factor = 1.0 / period * 2 * torch.pi
    sin_input = scaling_factor * float(time_value)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=0)


def adapt_flashrt_pi0_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Map FlashRT PI0 safetensors-converted weights to realtime-vla keys.

    realtime-vla's PI0 kernels expect the first action MLP to be pre-fused:
    ``action_in_proj`` followed by the action half of ``action_time_mlp_in``.
    FlashRT keeps those projections split, so this adapter mirrors
    realtime-vla's original ``convert_from_jax.py`` fusion on CPU.
    """
    out: dict[str, Any] = {}
    for key in DIRECT_PI0_KEYS:
        out[key] = _bf16_cpu(checkpoint[key])

    out["decoder_state_in_proj_w"] = _bf16_cpu(checkpoint["state_proj_w"])
    out["decoder_state_in_proj_b"] = _bf16_cpu(checkpoint["state_proj_b"])

    action_in_w = checkpoint["decoder_action_in_proj_w"].float().cpu()
    action_in_b = checkpoint["decoder_action_in_proj_b"].float().cpu()
    action_time_in_wa = checkpoint["action_time_mlp_in_wa_w"].float().cpu()
    action_time_in_wt_raw = checkpoint["_action_time_mlp_in_wt_raw"].float().cpu()
    action_time_in_b = checkpoint["_action_time_mlp_in_b"].float().cpu()

    out["decoder_action_fused_in_proj_w"] = _bf16_cpu(action_in_w @ action_time_in_wa)

    action_bias_contrib = action_time_in_wa.t() @ action_in_b
    time_biases = torch.empty(10, 1024, dtype=torch.float32)
    for step in range(10):
        time_emb = _sinusoidal_time_embedding(1.0 - step / 10.0)
        time_contrib = time_emb @ action_time_in_wt_raw.t()
        time_biases[step] = action_bias_contrib + time_contrib + action_time_in_b
    out["decoder_action_fused_time_biases"] = _bf16_cpu(time_biases)

    out["decoder_action_mlp_w"] = _bf16_cpu(checkpoint["action_time_mlp_out_w"])
    out["decoder_action_mlp_b"] = _bf16_cpu(checkpoint["action_time_mlp_out_b"])

    final_norm_w = checkpoint["decoder_final_norm_w"].float().cpu()
    action_out_w = checkpoint["decoder_action_out_proj_w"].float().cpu()
    out["decoder_action_fused_out_proj_w"] = _bf16_cpu(
        action_out_w * final_norm_w[:, None]
    )
    out["decoder_action_fused_out_proj_b"] = _bf16_cpu(
        checkpoint["decoder_action_out_proj_b"]
    )

    if "language_embeds" in checkpoint:
        out["language_embeds"] = _bf16_cpu(checkpoint["language_embeds"])
    if "embedding_weight" in checkpoint:
        out["embedding_weight"] = _bf16_cpu(checkpoint["embedding_weight"])

    return out


def maybe_adapt_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "state_proj_w" in checkpoint and "decoder_state_in_proj_w" not in checkpoint:
        return adapt_flashrt_pi0_checkpoint(checkpoint)
    return checkpoint


def load_checkpoint(path: Path, flashrt_root: Path | None, trust_pickle: bool):
    if path.is_dir():
        path = path / "model.safetensors"
    if path.suffix == ".safetensors":
        if flashrt_root is None:
            raise ValueError(
                "--flashrt-root or FLASHRT_ROOT is required when loading a safetensors checkpoint"
            )
        sys.path.insert(0, str(flashrt_root))
        from flash_rt.frontends.torch.pi0_rtx import convert_pi0_safetensors

        return adapt_flashrt_pi0_checkpoint(convert_pi0_safetensors(path))
    if path.suffix in {".pt", ".pth"}:
        return maybe_adapt_checkpoint(
            torch.load(path, map_location="cpu", weights_only=True)
        )
    if path.suffix in {".pkl", ".pickle"}:
        if not trust_pickle:
            raise ValueError(
                "Refusing to load pickle checkpoint without --trust-pickle-checkpoint. "
                "Only use that flag for checkpoints from a trusted source."
            )
        with path.open("rb") as f:
            return maybe_adapt_checkpoint(pickle.load(f))  # nosec B301: guarded.
    raise ValueError(
        f"Unsupported checkpoint suffix {path.suffix!r}; expected .safetensors, .pt, .pth, .pkl, or .pickle"
    )


def build_language_embeds(
    checkpoint: dict[str, Any],
    prompt: str,
    tokenizer_path: Path,
    max_length: int,
) -> torch.Tensor:
    from torch import nn
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    embedding_weight = checkpoint["embedding_weight"].to(
        device="cuda", dtype=torch.bfloat16
    )
    language_embedding = nn.Embedding(
        num_embeddings=embedding_weight.shape[0],
        embedding_dim=embedding_weight.shape[1],
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        language_embedding.weight.copy_(embedding_weight)
    token_ids = (
        tokenizer(
            [prompt.strip().replace("_", " ") + "\n"],
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        .to(device="cuda")
        .squeeze(0)
    )
    embeds = language_embedding(token_ids) * (embedding_weight.shape[1] ** 0.5)
    return embeds.cpu()


def ensure_language_embeds(
    checkpoint: dict[str, Any], args: argparse.Namespace
) -> None:
    if "language_embeds" in checkpoint:
        return
    if args.tokenizer is not None and "embedding_weight" in checkpoint:
        checkpoint["language_embeds"] = build_language_embeds(
            checkpoint,
            prompt=args.prompt,
            tokenizer_path=args.tokenizer,
            max_length=args.prompt_len,
        )
        return
    checkpoint["language_embeds"] = torch.randn(
        args.prompt_len, 2048, dtype=torch.bfloat16
    )


def validate_realtime_pi0_checkpoint(checkpoint: dict[str, Any]) -> None:
    missing = sorted(REALTIME_PI0_REQUIRED_KEY_SET.difference(checkpoint))
    if missing:
        raise KeyError(
            "Checkpoint is missing realtime-vla PI0 keys: " + ", ".join(missing)
        )


def filter_realtime_pi0_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    validate_realtime_pi0_checkpoint(checkpoint)
    return {key: checkpoint[key] for key in REALTIME_PI0_REQUIRED_KEYS}


def make_setup_fn(args: argparse.Namespace):
    sys.path.insert(0, str(args.realtime_vla_root))
    import pi0_infer as realtime_vla_pi0

    patch_realtime_vla_pi0_kernels(realtime_vla_pi0)
    Pi0Inference = realtime_vla_pi0.Pi0Inference

    def setup_fn(batch_size: int) -> bnb.BenchSpec:
        if batch_size != 1:
            raise ValueError("realtime-vla PI0 wrapper supports only batch_size=1")

        checkpoint = load_checkpoint(
            args.checkpoint, args.flashrt_root, args.trust_pickle_checkpoint
        )
        ensure_language_embeds(checkpoint, args)
        checkpoint = filter_realtime_pi0_checkpoint(checkpoint)

        infer = Pi0Inference(
            checkpoint=checkpoint,
            num_views=args.num_views,
            chunk_size=args.chunk_size,
        )
        torch.manual_seed(args.seed)
        input_image = torch.randn(
            args.num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda"
        )
        input_state = torch.randn(32, dtype=torch.bfloat16, device="cuda")
        input_noise = torch.randn(
            args.chunk_size, 32, dtype=torch.bfloat16, device="cuda"
        )

        def step() -> None:
            infer.forward(input_image, input_state, input_noise)

        return bnb.BenchSpec(
            name="realtime_vla_pi0",
            step_callable=step,
            teardown_callable=lambda: None,
        )

    return setup_fn


def make_extras_fn(args: argparse.Namespace):
    def extras_fn(batch_size: int, spec: bnb.BenchSpec) -> dict[str, Any]:
        return {
            "runtime": "realtime-vla",
            "model": "pi0",
            "checkpoint": str(args.checkpoint),
            "realtime_vla_root": str(args.realtime_vla_root),
            "flashrt_root": str(args.flashrt_root) if args.flashrt_root else None,
            "precision": "bf16",
            "batch_size_contract": 1,
            "num_views": args.num_views,
            "chunk_size": args.chunk_size,
            "prompt": args.prompt,
            "prompt_len": args.prompt_len,
            "seed": args.seed,
            "timing_scope": "Pi0Inference.forward hot path; common runner CUDA event wraps one forward call",
        }

    return extras_fn


def main() -> None:
    realtime_default = env_path("REALTIME_VLA_ROOT")
    flashrt_default = env_path("FLASHRT_ROOT")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--realtime-vla-root",
        type=Path,
        default=realtime_default,
        required=realtime_default is None,
        help="Path to the realtime-vla repository clone. Can also be set with REALTIME_VLA_ROOT.",
    )
    parser.add_argument(
        "--flashrt-root",
        type=Path,
        default=flashrt_default,
        help="Path to FlashRT. Required only when --checkpoint is a PI0 safetensors checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="realtime-vla .pkl/.pt checkpoint, PI0 model.safetensors, or a directory containing model.safetensors.",
    )
    parser.add_argument(
        "--trust-pickle-checkpoint",
        action="store_true",
        help="Allow loading .pkl/.pickle checkpoints. Only use with trusted checkpoint files.",
    )
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--prompt", default="do something")
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Optional tokenizer path. If set, language_embeds are built from the checkpoint embedding table and --prompt.",
    )

    bnb.add_bench_cli_args(parser)
    parser.set_defaults(bench_name="realtime_vla_pi0", batch_sizes=[1])
    add_profile_cli_args(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for realtime-vla PI0 benchmarking")

    install_profiler(profile_config_from_args(args))
    runner = bnb.NBatchBenchRunner(
        setup_fn=make_setup_fn(args),
        extras_fn=make_extras_fn(args),
        device=torch.device("cuda", torch.cuda.current_device()),
        **bnb.bench_runner_kwargs_from_args(args),
    )
    runner.run()


if __name__ == "__main__":
    main()
