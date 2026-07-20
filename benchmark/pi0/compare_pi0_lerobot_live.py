"""Compare PI0 accuracy and latency by importing LeRobot from PhyAI.

This script lives in the PhyAI tree and treats LeRobot as an optional runtime
dependency. Point ``--lerobot-root`` at a LeRobot checkout, or install LeRobot
in the active environment, then the script will:

1. Build one deterministic random observation batch.
2. Run LeRobot ``PI0Policy.predict_action_chunk`` and model-only
   ``sample_actions`` on that batch.
3. Run PhyAI ``Engine.step`` on equivalent tensors and the same sampling noise.
4. Report action error metrics and latency for both implementations.

Examples:

    uv run python benchmark/pi0/compare_pi0_lerobot_live.py \
        --checkpoint /data/share/pi0_base \
        --lerobot-root /path/to/lerobot \
        --tokenizer-name /data/share/paligemma-3b-pt-224 \
        --language-inputs tokenizer \
        --dtype bfloat16 \
        --num-steps 10 \
        --n-warmup 3 \
        --n-timed 10

    # Strict model-path compare with synthetic language tokens.
    uv run python benchmark/pi0/compare_pi0_lerobot_live.py \
        --checkpoint /data/share/pi0_base \
        --lerobot-root /path/to/lerobot \
        --language-inputs fixed \
        --save-output outputs/pi0_live_compare.pt
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import BackendConfig, DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi0.configuration_pi0 import PI0Config
from phyai.models.pi0.main_pi0 import PI0Args
from phyai.models.pi0.scheduler_ws1_pi0 import PI0Request
from phyai.utils import load_config as load_phyai_config


OBS_IMAGES_PREFIX = "observation.images."
OBS_STATE = "observation.state"
ACTION = "action"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def choose_attn_backend(dtype: torch.dtype, requested: str | None) -> str:
    if requested is not None and requested != "auto":
        return requested
    if dtype is torch.float32:
        return "eager"
    return "flashinfer"


def align_shapes(
    actual: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if actual.shape == reference.shape:
        return actual, reference

    if (
        actual.ndim == reference.ndim == 3
        and actual.shape[0] == reference.shape[0]
        and actual.shape[1] == reference.shape[1]
        and actual.shape[2] >= reference.shape[2]
    ):
        return actual[..., : reference.shape[2]], reference

    raise ValueError(
        f"Cannot compare tensors with shapes actual={tuple(actual.shape)} "
        f"reference={tuple(reference.shape)}."
    )


def compute_metrics(
    actual: torch.Tensor, reference: torch.Tensor, rtol: float, atol: float
) -> dict[str, Any]:
    actual, reference = align_shapes(actual.float().cpu(), reference.float().cpu())
    diff = actual - reference
    abs_diff = diff.abs()
    rel_diff = abs_diff / reference.abs().clamp_min(1e-6)

    actual_flat = actual.reshape(-1)
    reference_flat = reference.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(actual_flat, reference_flat, dim=0)

    return {
        "shape": list(actual.shape),
        "finite_actual": bool(torch.isfinite(actual).all().item()),
        "finite_reference": bool(torch.isfinite(reference).all().item()),
        "finite_diff": bool(torch.isfinite(diff).all().item()),
        "allclose": bool(torch.allclose(actual, reference, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "max_rel": float(rel_diff.max().item()),
        "mean_rel": float(rel_diff.mean().item()),
        "cosine": float(cosine.item()),
    }


def print_preview(actual: torch.Tensor, reference: torch.Tensor) -> None:
    actual, reference = align_shapes(actual.float().cpu(), reference.float().cpu())
    first_actual = actual[0, 0].tolist()
    first_reference = reference[0, 0].tolist()
    first_diff = (actual[0, 0] - reference[0, 0]).tolist()

    print("first action actual   :", first_actual)
    print("first action reference:", first_reference)
    print("first action diff     :", first_diff)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live PhyAI-vs-LeRobot PI0 action and latency comparison."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="PI0 checkpoint directory."
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=None,
        help="Optional LeRobot checkout root. Its src/ directory is prepended to sys.path.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "bf16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--vision-dtype",
        dest="vision_dtype",
        choices=("float32", "bf16", "bfloat16"),
        default="float32",
        help=(
            "PhyAI PI0 vision tower dtype. This only affects "
            "PI0Args.vision_params_dtype; the LeRobot path is unchanged."
        ),
    )
    parser.add_argument(
        "--phyai-attn-backend",
        default=None,
        choices=("auto", "flashinfer", "sdpa", "eager"),
        help="PhyAI attention backend. Default is eager for fp32 and flashinfer otherwise.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt", default="Pick up the red block and place it in the bin"
    )
    parser.add_argument(
        "--language-inputs",
        choices=("fixed", "tokenizer"),
        default="tokenizer",
        help="Use real PaliGemma tokenization or synthetic fixed tokens.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="/data/share/paligemma-3b-pt-224",
        help="Tokenizer repo id or local directory. Used with --language-inputs tokenizer.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        choices=(2, 3),
        default=None,
        help="Override camera count. Default follows the PhyAI checkpoint config.",
    )
    parser.add_argument(
        "--phyai-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA graph for PhyAI.",
    )
    parser.add_argument(
        "--weight-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strict checkpoint loading in PhyAI PI0Args.",
    )
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-timed", type=int, default=10)
    parser.add_argument("--save-output", type=Path, default=None)
    return parser.parse_args()


def add_lerobot_to_path(lerobot_root: Path | None) -> None:
    if lerobot_root is None:
        return
    root = lerobot_root.resolve()
    src = root / "src"
    if not src.is_dir():
        raise NotADirectoryError(f"--lerobot-root must contain src/, got {root}")
    sys.path.insert(0, str(src))


def import_lerobot_symbols() -> dict[str, Any]:
    try:
        pretrained_mod = importlib.import_module("lerobot.configs")
        pi0_mod = importlib.import_module("lerobot.policies.pi0")
    except Exception as exc:
        raise RuntimeError(
            "Could not import LeRobot. Install it in this environment or pass "
            "--lerobot-root /path/to/lerobot-main."
        ) from exc
    return {
        "PreTrainedConfig": pretrained_mod.PreTrainedConfig,
        "PI0Policy": pi0_mod.PI0Policy,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_callable(
    fn,
    *,
    device: torch.device,
    n_warmup: int,
    n_timed: int,
) -> dict[str, float | int]:
    if n_timed <= 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n_warmup": n_warmup,
            "n_timed": n_timed,
        }

    for _ in range(n_warmup):
        fn()
    sync_if_needed(device)

    times_ms: list[float] = []
    for _ in range(n_timed):
        sync_if_needed(device)
        start = time.perf_counter()
        fn()
        sync_if_needed(device)
        times_ms.append((time.perf_counter() - start) * 1000)

    return {
        "mean": statistics.fmean(times_ms),
        "median": statistics.median(times_ms),
        "stdev": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min": min(times_ms),
        "max": max(times_ms),
        "n_warmup": n_warmup,
        "n_timed": n_timed,
    }


def tensor_summary(tensor: torch.Tensor) -> dict[str, float | list[int]]:
    tensor = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "l2": float(torch.linalg.vector_norm(tensor)),
    }


def is_empty_camera_key(key: str) -> bool:
    return key.startswith(OBS_IMAGES_PREFIX) and ".empty_camera_" in key


def resolve_lerobot_image_keys(policy: Any, num_images: int) -> list[str]:
    image_keys = [
        key for key in policy.config.image_features if not is_empty_camera_key(key)
    ]
    if len(image_keys) < num_images:
        raise ValueError(
            f"Checkpoint declares {len(image_keys)} real image features, "
            f"but PhyAI config expects {num_images}: {image_keys}"
        )
    return image_keys[:num_images]


def make_raw_inputs(
    *,
    batch_size: int,
    image_keys: list[str],
    image_size: int,
    max_state_dim: int,
    max_action_dim: int,
    chunk_size: int,
    device: torch.device,
    prompt: str,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        OBS_STATE: torch.randn(
            batch_size, max_state_dim, dtype=torch.float32, device=device
        ),
        ACTION: torch.randn(
            batch_size, chunk_size, max_action_dim, dtype=torch.float32, device=device
        ),
        "task": [prompt for _ in range(batch_size)],
    }
    for key in image_keys:
        raw[key] = torch.rand(
            batch_size,
            3,
            image_size,
            image_size,
            dtype=torch.float32,
            device=device,
        )
    return raw


def make_language_inputs(
    *,
    raw: dict[str, Any],
    mode: str,
    tokenizer_name: str,
    tokenizer_max_length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(raw[OBS_STATE].shape[0])
    if mode == "fixed":
        token_id = min(2, vocab_size - 1)
        input_ids = torch.full(
            (batch_size, tokenizer_max_length),
            token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones(
            (batch_size, tokenizer_max_length), dtype=torch.bool, device=device
        )
        return input_ids, attention_mask

    if mode != "tokenizer":
        raise ValueError(f"Unsupported language input mode: {mode}")

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for --language-inputs tokenizer"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tasks = [task if task.endswith("\n") else f"{task}\n" for task in raw["task"]]
    tokenized = tokenizer(
        tasks,
        max_length=tokenizer_max_length,
        truncation=True,
        padding="max_length",
        padding_side="right",
        return_tensors="pt",
    )
    return (
        tokenized["input_ids"].to(device=device, dtype=torch.long),
        tokenized["attention_mask"].to(device=device, dtype=torch.bool),
    )


def make_lerobot_batch(
    raw: dict[str, Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    batch = {
        key: value.clone() if torch.is_tensor(value) else list(value)
        for key, value in raw.items()
    }
    batch[OBS_LANGUAGE_TOKENS] = input_ids
    batch[OBS_LANGUAGE_ATTENTION_MASK] = attention_mask
    return batch


def make_phyai_request(
    raw: dict[str, Any],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    sample_noise: torch.Tensor,
    *,
    image_keys: list[str],
    device: torch.device,
) -> PI0Request:
    images = []
    for image_key in image_keys:
        image = raw[image_key].detach().to(dtype=torch.float32)
        images.append(image.mul(2.0).sub(1.0))

    return PI0Request(
        pixel_values=torch.stack(images, dim=1).to(device=device),
        input_ids=input_ids.to(device=device, dtype=torch.long),
        lang_lens=attention_mask.to(dtype=torch.long).sum(dim=-1).to(device=device),
        state=raw[OBS_STATE].to(device=device, dtype=torch.float32),
        noise=sample_noise.to(device=device, dtype=torch.float32),
    )


def normalize_lerobot_pi0_state_key(key: str) -> tuple[str, bool, bool]:
    added_model_prefix = not key.startswith("model.")
    if added_model_prefix:
        key = f"model.{key}"

    remapped_vision_tower = ".vision_tower.vision_model." in key
    if remapped_vision_tower:
        key = key.replace(".vision_tower.vision_model.", ".vision_tower.")

    return key, added_model_prefix, remapped_vision_tower


def preview_keys(keys: list[str], *, limit: int = 8) -> str:
    if len(keys) <= limit:
        return "\n".join(f"  - {key}" for key in keys)
    head = "\n".join(f"  - {key}" for key in keys[:limit])
    return f"{head}\n  ... and {len(keys) - limit} more"


def load_lerobot_state_dict(policy: Any, checkpoint: Path) -> None:
    model_file = checkpoint / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"Expected LeRobot checkpoint weights at {model_file}")

    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError(
            "safetensors is required to load LeRobot PI0 weights"
        ) from exc

    original_state_dict = load_file(model_file, device="cpu")
    fixed_state_dict = policy._fix_pytorch_state_dict_keys(  # noqa: SLF001
        original_state_dict,
        policy.config,
    )

    remapped_state_dict = {}
    model_prefix_count = 0
    vision_tower_count = 0
    for key, value in fixed_state_dict.items():
        new_key, added_model_prefix, remapped_vision_tower = (
            normalize_lerobot_pi0_state_key(key)
        )
        model_prefix_count += int(added_model_prefix)
        vision_tower_count += int(remapped_vision_tower)
        if new_key in remapped_state_dict:
            raise RuntimeError(
                f"Duplicate LeRobot state dict key after remap: {new_key}"
            )
        remapped_state_dict[new_key] = value

    incompatible = policy.load_state_dict(
        remapped_state_dict, strict=False, assign=False
    )
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        details = []
        if missing_keys:
            details.append(f"Missing keys:\n{preview_keys(missing_keys)}")
        if unexpected_keys:
            details.append(f"Unexpected keys:\n{preview_keys(unexpected_keys)}")
        raise RuntimeError(
            "Could not strictly load LeRobot PI0 state dict after compatibility remap.\n"
            + "\n".join(details)
        )

    print(
        "Loaded LeRobot PI0 weights from "
        f"{model_file} (added model. prefix: {model_prefix_count}, "
        f"vision_tower.vision_model remaps: {vision_tower_count})"
    )


def load_lerobot_policy(
    *,
    symbols: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
    dtype_name: str,
    num_steps: int,
):
    config = symbols["PreTrainedConfig"].from_pretrained(checkpoint)
    config.device = str(device)
    config.dtype = "float32" if dtype_name == "float32" else "bfloat16"
    config.compile_model = False
    config.num_inference_steps = num_steps
    policy = symbols["PI0Policy"](config)
    load_lerobot_state_dict(policy, checkpoint)
    policy.to(device).eval()
    policy.config.device = str(device)
    return policy


def trim_lerobot_image_features(policy: Any, image_keys: list[str]) -> None:
    keep = set(image_keys)
    policy.config.input_features = {
        key: value
        for key, value in policy.config.input_features.items()
        if not key.startswith("observation.images.") or key in keep
    }


def load_phyai_engine(
    *,
    checkpoint: Path,
    config: PI0Config,
    batch_size: int,
    dtype: torch.dtype,
    vision_dtype: torch.dtype,
    attn_backend: str,
    device_target: str,
    use_cuda_graph: bool,
    weight_strict: bool,
) -> Engine:
    return Engine(
        EngineArgs(
            plugin="pi0",
            plugin_args=PI0Args(
                checkpoint_dir=checkpoint,
                config=config,
                max_batch_size=batch_size,
                weight_strict=weight_strict,
                vision_params_dtype=vision_dtype,
            ),
            config=EngineConfig(
                backends=BackendConfig(attn=attn_backend),
                device=DeviceConfig(target=device_target, params_dtype=dtype),
                runtime=RuntimeConfig(use_cuda_graph=use_cuda_graph),
            ),
        )
    )


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_dir():
        raise NotADirectoryError(
            f"--checkpoint must be a directory, got {args.checkpoint}"
        )

    add_lerobot_to_path(args.lerobot_root)
    symbols = import_lerobot_symbols()
    seed_everything(args.seed)

    dtype = dtype_from_name(args.dtype)
    vision_dtype = dtype_from_name(args.vision_dtype)
    attn_backend = choose_attn_backend(dtype, args.phyai_attn_backend)
    device = torch.device(args.device)

    phyai_cfg = load_phyai_config(args.checkpoint, PI0Config)
    phyai_cfg = replace(phyai_cfg, num_inference_steps=args.num_steps)
    if args.num_images is not None:
        phyai_cfg = replace(phyai_cfg, empty_cameras=3 - args.num_images)
    num_images = phyai_cfg.num_images

    lerobot_policy = load_lerobot_policy(
        symbols=symbols,
        checkpoint=args.checkpoint,
        device=device,
        dtype_name=args.dtype,
        num_steps=args.num_steps,
    )
    image_keys = resolve_lerobot_image_keys(lerobot_policy, num_images)
    # Keep the same real cameras PhyAI will pack into PI0Request, and drop
    # empty-camera placeholders so both runtimes execute the same vision inputs.
    trim_lerobot_image_features(lerobot_policy, image_keys)

    embed_tokens = lerobot_policy.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens

    raw = make_raw_inputs(
        batch_size=args.batch_size,
        image_keys=image_keys,
        image_size=phyai_cfg.vision.image_size,
        max_state_dim=phyai_cfg.max_state_dim,
        max_action_dim=phyai_cfg.max_action_dim,
        chunk_size=phyai_cfg.chunk_size,
        device=device,
        prompt=args.prompt,
    )
    input_ids, attention_mask = make_language_inputs(
        raw=raw,
        mode=args.language_inputs,
        tokenizer_name=args.tokenizer_name,
        tokenizer_max_length=phyai_cfg.tokenizer_max_length,
        vocab_size=embed_tokens.num_embeddings,
        device=device,
    )
    sample_noise = torch.randn(
        args.batch_size,
        phyai_cfg.chunk_size,
        phyai_cfg.max_action_dim,
        dtype=torch.float32,
        device=device,
    )
    lerobot_batch = make_lerobot_batch(raw, input_ids, attention_mask)
    phyai_request = make_phyai_request(
        raw,
        input_ids,
        attention_mask,
        sample_noise,
        image_keys=image_keys,
        device=device,
    )

    phyai_engine = load_phyai_engine(
        checkpoint=args.checkpoint,
        config=phyai_cfg,
        batch_size=args.batch_size,
        dtype=dtype,
        vision_dtype=vision_dtype,
        attn_backend=attn_backend,
        device_target=args.device,
        use_cuda_graph=args.phyai_cuda_graph and device.type == "cuda",
        weight_strict=args.weight_strict,
    )

    try:
        with torch.inference_mode():
            lerobot_actions = lerobot_policy.predict_action_chunk(
                lerobot_batch,
                noise=sample_noise,
                num_steps=args.num_steps,
            ).detach()
            phyai_actions = phyai_engine.step(phyai_request).detach()

            lerobot_model_images, lerobot_model_img_masks = (
                lerobot_policy._preprocess_images(  # noqa: SLF001
                    lerobot_batch
                )
            )
            lerobot_model_lang_tokens = lerobot_batch[OBS_LANGUAGE_TOKENS]
            lerobot_model_lang_masks = lerobot_batch[OBS_LANGUAGE_ATTENTION_MASK]
            lerobot_model_state = lerobot_policy.prepare_state(lerobot_batch)

            lerobot_timing = time_callable(
                lambda: lerobot_policy.predict_action_chunk(
                    lerobot_batch,
                    noise=sample_noise,
                    num_steps=args.num_steps,
                ),
                device=device,
                n_warmup=args.n_warmup,
                n_timed=args.n_timed,
            )
            lerobot_model_timing = time_callable(
                lambda: lerobot_policy.model.sample_actions(
                    lerobot_model_images,
                    lerobot_model_img_masks,
                    lerobot_model_lang_tokens,
                    lerobot_model_lang_masks,
                    lerobot_model_state,
                    noise=sample_noise,
                    num_steps=args.num_steps,
                ),
                device=device,
                n_warmup=args.n_warmup,
                n_timed=args.n_timed,
            )
            phyai_timing = time_callable(
                lambda: phyai_engine.step(phyai_request),
                device=device,
                n_warmup=args.n_warmup,
                n_timed=args.n_timed,
            )
    finally:
        close = getattr(phyai_engine, "close", None)
        if close is not None:
            close()

    metrics = compute_metrics(
        phyai_actions.float().cpu(),
        lerobot_actions.float().cpu(),
        rtol=args.rtol,
        atol=args.atol,
    )
    result = {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "lerobot_root": str(args.lerobot_root) if args.lerobot_root else None,
            "device": args.device,
            "dtype": args.dtype,
            "phyai_vision_dtype": args.vision_dtype,
            "phyai_attn_backend": attn_backend,
            "phyai_cuda_graph": args.phyai_cuda_graph and device.type == "cuda",
            "batch_size": args.batch_size,
            "num_images": num_images,
            "image_keys_order": image_keys,
            "num_steps": args.num_steps,
            "language_inputs": args.language_inputs,
            "tokenizer_name": args.tokenizer_name
            if args.language_inputs == "tokenizer"
            else None,
            "seed": args.seed,
        },
        "metrics": metrics,
        "timing_ms": {
            "lerobot_predict_action_chunk": lerobot_timing,
            "lerobot_model_sample_actions": lerobot_model_timing,
            "phyai_engine_step": phyai_timing,
        },
        "summary": {
            "lerobot_actions": tensor_summary(lerobot_actions),
            "phyai_actions": tensor_summary(phyai_actions),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print_preview(phyai_actions.float().cpu(), lerobot_actions.float().cpu())

    if args.save_output is not None:
        phyai_aligned, lerobot_aligned = align_shapes(
            phyai_actions.float().cpu(),
            lerobot_actions.float().cpu(),
        )
        raw_cpu = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in raw.items()
        }
        processed_batch = {
            **raw_cpu,
            OBS_LANGUAGE_TOKENS: input_ids.detach().cpu(),
            OBS_LANGUAGE_ATTENTION_MASK: attention_mask.detach().cpu(),
        }
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "meta": result["meta"],
                "raw_batch": raw_cpu,
                "processed_batch": processed_batch,
                "sample_noise": sample_noise.detach().cpu(),
                "actions": lerobot_aligned,
                "metrics": metrics,
                "timing_ms": result["timing_ms"],
                "input_ids": input_ids.detach().cpu(),
                "attention_mask": attention_mask.detach().cpu(),
                "phyai_actions": phyai_aligned,
                "lerobot_actions": lerobot_aligned,
                "diff": phyai_aligned - lerobot_aligned,
            },
            args.save_output,
        )
        print(f"saved output: {args.save_output}")


if __name__ == "__main__":
    main()
