from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import statistics
import sys
import types
from contextlib import redirect_stdout
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
import yaml
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig


REQUIRED_INPUTS = (
    "input.pixel_values",
    "input.image_grid_thw",
    "input.image_masks",
    "input.input_ids",
    "input.lang_mask",
    "input.state",
    "input.noise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure hook-free official LingBot V2 model latency on one fixed "
            "model-ready batch. Loading, host I/O, and preprocessing are excluded."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--official-config", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-timed", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(distribution: str, module_name: str | None = None) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        if module_name is None:
            return "unknown"
        try:
            module = __import__(module_name)
        except ImportError:
            return "not-installed"
        return str(getattr(module, "__version__", "unknown"))


def checkpoint_shards(checkpoint: Path) -> list[Path]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return [checkpoint / name for name in sorted(set(index["weight_map"].values()))]
    shards = sorted(checkpoint.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors checkpoint found in {checkpoint}")
    return shards


def stream_load_state_dict(model: torch.nn.Module, checkpoint: Path) -> None:
    expected = set(model.state_dict())
    loaded: set[str] = set()
    unexpected: set[str] = set()
    for shard in checkpoint_shards(checkpoint):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            state = {name: handle.get_tensor(name) for name in handle.keys()}
        incompatible = model.load_state_dict(state, strict=False)
        loaded.update(state)
        unexpected.update(incompatible.unexpected_keys)
        del state
    missing = sorted(expected - loaded)
    if missing or unexpected:
        raise RuntimeError(
            "Official weight load was not strict: "
            f"missing={missing[:20]}, unexpected={sorted(unexpected)[:20]}"
        )


def build_official_config(config_path: Path, processor_path: str) -> Any:
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values = dict(raw["model"])
    values.update(raw["train"])
    values["tokenizer_path"] = processor_path
    values["attention_implementation"] = "eager"
    values["vit_attn_implementation"] = "eager"
    values["moe_implementation"] = "fused"
    values["use_compile"] = False
    config = LingbotVLAV2Config(**values)

    qwen_config = AutoConfig.from_pretrained(processor_path)
    qwen_dict = qwen_config.to_dict()
    text_config = qwen_dict.get("text_config", {})
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
    ):
        if name in text_config:
            setattr(config, name, text_config[name])
        elif name in qwen_dict:
            setattr(config, name, qwen_dict[name])
    config.vision_config = qwen_config.vision_config
    config.use_cache = True
    return config


def install_reference_import_stubs(official_repo: Path) -> None:
    importlib.import_module("lingbotvla")
    package_paths = {
        "lingbotvla.models": official_repo / "lingbotvla" / "models",
        "lingbotvla.models.vla": official_repo / "lingbotvla" / "models" / "vla",
        "lingbotvla.ops": official_repo / "lingbotvla" / "ops",
    }
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module

    loader = types.ModuleType("lingbotvla.models.loader")

    class LingBotVLAWeightLoader:
        pass

    loader.LingBotVLAWeightLoader = LingBotVLAWeightLoader
    sys.modules[loader.__name__] = loader


def install_transformers_compatibility() -> None:
    if not hasattr(transformers, "AutoModelForVision2Seq"):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText
    qwen25_modeling = importlib.import_module(
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
    )
    if not hasattr(qwen25_modeling, "Qwen2RMSNorm"):
        qwen25_modeling.Qwen2RMSNorm = qwen25_modeling.Qwen2_5_VLRMSNorm


def install_rank0_logging_compatibility(module: types.ModuleType) -> None:
    if not hasattr(module.logger, "info_rank0"):
        module.logger.info_rank0 = module.logger.info


def install_tied_weights_compatibility(
    qwen2_action_expert: types.ModuleType,
    qwen3_patch: types.ModuleType,
) -> None:
    qwen3_patch.Qwen3VLForConditionalGeneration._tied_weights_keys = {
        "lm_head.weight": "model.language_model.embed_tokens.weight"
    }
    qwen2_action_expert.Qwen2ForCausalLM._tied_weights_keys = {
        "lm_head.weight": "model.embed_tokens.weight"
    }


def install_qwen_layout_compatibility(qwen3_patch: types.ModuleType) -> None:
    qwen3_cls = qwen3_patch.Qwen3VLForConditionalGeneration
    if not hasattr(qwen3_cls, "visual"):
        qwen3_cls.visual = property(lambda self: self.model.visual)

    model_cls = qwen3_patch.Qwen3VLModel
    original_get_rope_index = model_cls.get_rope_index
    parameters = inspect.signature(original_get_rope_index).parameters
    if "mm_token_type_ids" not in parameters or getattr(
        model_cls, "_lingbot_rope_index_compatibility", False
    ):
        return

    def get_rope_index_compatibility(self, *args, **kwargs):
        if kwargs.get("mm_token_type_ids") is None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is None:
                raise ValueError("input_ids are required to derive mm_token_type_ids")
            token_types = torch.zeros_like(input_ids, dtype=torch.int64)
            image_token_id = getattr(self.config, "image_token_id", None)
            video_token_id = getattr(self.config, "video_token_id", None)
            if image_token_id is not None:
                token_types.masked_fill_(input_ids == image_token_id, 1)
            if video_token_id is not None:
                token_types.masked_fill_(input_ids == video_token_id, 2)
            kwargs["mm_token_type_ids"] = token_types
        return original_get_rope_index(self, *args, **kwargs)

    model_cls.get_rope_index = get_rope_index_compatibility
    model_cls._lingbot_rope_index_compatibility = True


def verify_thor_compatible_source(official_repo: Path) -> dict[str, Any]:
    manifest_path = official_repo / "PHYAI_THOR_COMPAT_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "official source has not been prepared for Thor; run "
            "prepare_official_thor.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = official_repo / manifest["model_relative_path"]
    actual_hash = sha256_file(model_path)
    if actual_hash != manifest["model_sha256_after"]:
        raise RuntimeError(
            "prepared official source hash does not match its compatibility " "manifest"
        )
    text = model_path.read_text(encoding="utf-8")
    hardcoded = (
        'vlm_config._attn_implementation = "flash_attention_2"',
        'vlm_config.text_config._attn_implementation = "flash_attention_2"',
        (
            "self.config.qwen_expert_config._attn_implementation = "
            '"flash_attention_2"'
        ),
    )
    if any(line in text for line in hardcoded):
        raise RuntimeError("official source still hard-codes FlashAttention2")
    return manifest


def strict_native_robby_moe(model: torch.nn.Module) -> int:
    def fallback_is_forbidden(_self, *_args, **_kwargs):
        raise RuntimeError(
            "official Robby MoE failed or was not selected; refusing to time "
            "the fallback as the native official backend"
        )

    count = 0
    for layer in model.model.qwenvl_with_expert.qwen_expert.model.layers:
        experts = getattr(layer.mlp, "experts", None)
        if experts is not None and hasattr(experts, "gate_proj"):
            experts.forward = types.MethodType(fallback_is_forbidden, experts)
            count += 1
    if count == 0:
        raise RuntimeError("no official fused MoE expert layers were found")
    return count


def install_thor_patch_embed_compatibility(model: torch.nn.Module) -> str:
    """Keep the official BF16 Conv3D while bypassing unavailable cuDNN."""

    patch_embed = model.model.qwenvl_with_expert.qwenvl.visual.patch_embed
    original_forward = patch_embed.forward

    def forward_without_cudnn(_self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.backends.cudnn.flags(enabled=False):
            return original_forward(hidden_states)

    patch_embed.forward = types.MethodType(
        forward_without_cudnn,
        patch_embed,
    )
    return "official_bf16_conv3d_cudnn_disabled"


def load_inputs(path: Path) -> dict[str, torch.Tensor]:
    tensors = load_file(path, device="cpu")
    missing = [name for name in REQUIRED_INPUTS if name not in tensors]
    if missing:
        raise ValueError(f"input artifact is missing tensors: {missing}")
    return tensors


def validate_contract(
    tensors: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, int | list[int]]:
    pixel_values = tensors["input.pixel_values"]
    grids = tensors["input.image_grid_thw"]
    masks = tensors["input.image_masks"].bool()
    input_ids = tensors["input.input_ids"]
    lang_mask = tensors["input.lang_mask"].bool()
    state = tensors["input.state"]
    noise = tensors["input.noise"]

    if pixel_values.shape[:3] != (1, 3, 256):
        raise ValueError(
            "fixed benchmark pixel_values must start with shape (1,3,256), "
            f"got {tuple(pixel_values.shape)}"
        )
    if pixel_values.shape[-1] != 1536:
        raise ValueError("fixed benchmark patch width must be 1536")
    if grids.shape != (1, 3, 3) or not bool((grids.prod(-1) == 256).all()):
        raise ValueError("fixed benchmark requires three 256-patch image grids")
    if masks.shape != (1, 3) or not bool(masks.all()):
        raise ValueError("all three benchmark views must be active")
    if input_ids.shape != lang_mask.shape or input_ids.shape[0] != 1:
        raise ValueError("input_ids and lang_mask must have identical B=1 shape")
    lang_len = int(lang_mask.sum())
    if not bool(lang_mask[0, :lang_len].all()) or bool(lang_mask[0, lang_len:].any()):
        raise ValueError("fixed prompt mask must be a contiguous valid prefix")
    if state.shape != (1, int(config.max_state_dim)):
        raise ValueError("state shape does not match the official config")
    if noise.shape != (
        1,
        int(config.n_action_steps),
        int(config.max_action_dim),
    ):
        raise ValueError("noise shape does not match the official config")
    if int(config.n_action_steps) != 50 or int(config.num_steps) != 10:
        raise ValueError("fixed benchmark requires chunk=50 and Euler steps=10")
    return {
        "batch_size": 1,
        "num_images": 3,
        "max_patches_per_image": 256,
        "patches_per_image": 256,
        "patch_vector_dim": 1536,
        "lang_len": lang_len,
        "input_ids": input_ids[0, :lang_len].tolist(),
        "chunk_size": int(config.n_action_steps),
        "num_inference_steps": int(config.num_steps),
        "action_dim": int(config.action_dim),
    }


def latency_stats(times_ms: list[float]) -> dict[str, float]:
    ordered = sorted(times_ms)
    return {
        "mean": statistics.fmean(ordered),
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "p99": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
        "std": statistics.pstdev(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def main() -> None:
    args = parse_args()
    if args.n_warmup < 1 or args.n_timed < 1:
        raise ValueError("n_warmup and n_timed must both be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official latency measurement")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    official_repo = args.official_repo.resolve()
    compatibility_manifest = verify_thor_compatible_source(official_repo)
    sys.path.insert(0, str(official_repo))

    install_transformers_compatibility()
    install_reference_import_stubs(official_repo)
    qwen2_action_expert = importlib.import_module(
        "lingbotvla.models.vla.lingbot_vla.qwen2_action_expert"
    )
    if qwen2_action_expert.robby_moe_forward is None:
        raise RuntimeError("official Robby MoE backend is unavailable")
    modeling = importlib.import_module(
        "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2"
    )
    qwen_patch = importlib.import_module(
        "lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla"
    )
    install_rank0_logging_compatibility(qwen_patch)
    install_tied_weights_compatibility(qwen2_action_expert, qwen_patch)
    install_qwen_layout_compatibility(qwen_patch)
    qwen_patch.apply_lingbot_qwen3_vl_patch()

    config_path = args.official_config or (
        official_repo / "configs" / "vla" / "robotwin" / "robotwin.yaml"
    )
    config = build_official_config(config_path, args.processor)
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = modeling.LingbotVlaV2Policy(config, eval=True)
    finally:
        torch.set_default_dtype(previous_default)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    stream_load_state_dict(model, args.checkpoint)
    patch_embed_backend = install_thor_patch_embed_compatibility(model)
    strict_moe_layers = strict_native_robby_moe(model)

    inputs = load_inputs(args.input)
    contract = validate_contract(inputs, config)
    images = inputs["input.pixel_values"].to(device, dtype=torch.bfloat16)
    image_masks = inputs["input.image_masks"].to(device, dtype=torch.bool)
    input_ids = inputs["input.input_ids"].to(device, dtype=torch.int64)
    lang_mask = inputs["input.lang_mask"].to(device, dtype=torch.bool)
    state = inputs["input.state"].to(device, dtype=torch.bfloat16)
    noise = inputs["input.noise"].to(device, dtype=torch.bfloat16)
    image_grid_thw = inputs["input.image_grid_thw"].to(
        device,
        dtype=torch.int64,
    )

    with open(os.devnull, "w", encoding="utf-8") as devnull:

        def infer() -> torch.Tensor:
            with redirect_stdout(devnull):
                return model.model.sample_actions(
                    images,
                    image_masks,
                    input_ids,
                    lang_mask,
                    state,
                    noise=noise.clone(),
                    image_grid_thw=image_grid_thw,
                )

        with torch.inference_mode():
            print(
                f"Official warmup: {args.n_warmup} forwards "
                "(the first run may compile Robby MoE)"
            )
            output = infer()
            for _ in range(args.n_warmup - 1):
                output = infer()
            torch.cuda.synchronize(device)

            print("Official deterministic repeat check")
            repeat = infer()
            torch.cuda.synchronize(device)
            checks: dict[str, float | bool | int | list[int] | str] = {
                "exact": bool(torch.equal(output, repeat)),
                "cosine": float(
                    F.cosine_similarity(
                        output.float().flatten(),
                        repeat.float().flatten(),
                        dim=0,
                    )
                ),
                "max_abs": float((output.float() - repeat.float()).abs().max()),
                "all_finite": bool(torch.isfinite(repeat).all()),
            }
            if not checks["all_finite"]:
                raise RuntimeError("official benchmark output is non-finite")

            torch.cuda.reset_peak_memory_stats(device)
            times_ms: list[float] = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            print(f"Official timing: {args.n_timed} CUDA-event forwards")
            for index in range(args.n_timed):
                start.record()
                output = infer()
                end.record()
                torch.cuda.synchronize(device)
                times_ms.append(start.elapsed_time(end))
                if (index + 1) % 10 == 0 or index + 1 == args.n_timed:
                    print(f"  completed {index + 1}/{args.n_timed}")

    checks.update(
        allocated_peak_bytes=torch.cuda.max_memory_allocated(device),
        reserved_peak_bytes=torch.cuda.max_memory_reserved(device),
        output_shape=list(output.shape),
        output_dtype=str(output.dtype),
    )
    properties = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": 1,
        "meta": {
            "model": "lingbot_v2",
            "implementation": "official_thor_compat",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "checkpoint": str(args.checkpoint),
            "input": str(args.input),
            "input_sha256": sha256_file(args.input),
            "contract": contract,
            "params_dtype": str(torch.bfloat16),
            "vision_dtype": str(torch.bfloat16),
            "attention_backend": "official_eager",
            "vision_attention_backend": "official_eager",
            "vision_patch_embed_backend": patch_embed_backend,
            "moe_backend": "official_robby_triton_strict",
            "strict_moe_layers": strict_moe_layers,
            "linear_kernel": "official_torch_linear",
            "use_cuda_graph": False,
            "torch_compile": False,
            "n_warmup": args.n_warmup,
            "n_timed": args.n_timed,
            "timing_backend": "cuda_events",
            "forward_hooks": False,
            "stdout_suppressed": True,
            "compatibility_manifest": compatibility_manifest,
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": package_version("triton"),
            "transformers": package_version("transformers"),
            "safetensors": package_version("safetensors"),
        },
        "hardware": {
            "name": torch.cuda.get_device_name(device),
            "index": device.index or 0,
            "compute_capability": (f"{properties.major}.{properties.minor}"),
            "total_mem_gb": properties.total_memory / 2**30,
            "sm_count": properties.multi_processor_count,
        },
        "latency_ms": latency_stats(times_ms),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    latency = result["latency_ms"]
    print("Official LingBot V2 Thor latency")
    print(json.dumps(contract, indent=2))
    print(f"GPU                 : {result['hardware']['name']}")
    print(f"input SHA256        : {result['meta']['input_sha256']}")
    print("attention           : official eager (Thor compatibility)")
    print("vision PatchEmbed   : official BF16 Conv3D, cuDNN disabled")
    print("MoE                 : official Robby Triton, strict")
    print(
        "latency mean/p50/p99: "
        f"{latency['mean']:.3f} / {latency['p50']:.3f} / "
        f"{latency['p99']:.3f} ms"
    )
    print(
        "memory peak         : "
        f"{int(checks['allocated_peak_bytes']) / 2**30:.3f} GiB allocated, "
        f"{int(checks['reserved_peak_bytes']) / 2**30:.3f} GiB reserved"
    )
    print(
        "determinism         : "
        f"exact={checks['exact']} cosine={checks['cosine']:.9f} "
        f"max_abs={checks['max_abs']:.3e}"
    )
    print(f"report              : {args.output}")


if __name__ == "__main__":
    main()
