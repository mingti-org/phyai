from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

import hardware_probe as hp
import model_flops_lingbot_v2 as mf
import profile_metrics as pm

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
import phyai.models.lingbot_v2.scheduler_lingbotv2 as scheduler_module
from phyai.models.lingbot_v2 import (
    LingBotV2Args,
    LingBotV2Request,
    LingBotVLA2Config,
)
from phyai.utils import load_config


STAGES = (
    "lingbot_v2.vision",
    "lingbot_v2.prefix_pack",
    "lingbot_v2.prefix_plan",
    "lingbot_v2.prefix_forward",
    "lingbot_v2.expert_plan",
    "lingbot_v2.euler",
)

DETAIL_STAGES = (
    "lingbot_v2.detail.vision_patch_embed",
    "lingbot_v2.detail.vision_blocks",
    "lingbot_v2.detail.vision_mergers",
    "lingbot_v2.detail.text_attention",
    "lingbot_v2.detail.text_mlp",
    "lingbot_v2.detail.expert_attention",
    "lingbot_v2.detail.expert_moe",
    "lingbot_v2.detail.expert_adanorm",
    "lingbot_v2.detail.expert_heads",
)

DETAIL_PARENT = {
    "lingbot_v2.detail.vision_patch_embed": "lingbot_v2.vision",
    "lingbot_v2.detail.vision_blocks": "lingbot_v2.vision",
    "lingbot_v2.detail.vision_mergers": "lingbot_v2.vision",
    "lingbot_v2.detail.text_attention": "lingbot_v2.prefix_forward",
    "lingbot_v2.detail.text_mlp": "lingbot_v2.prefix_forward",
    "lingbot_v2.detail.expert_attention": "lingbot_v2.euler",
    "lingbot_v2.detail.expert_moe": "lingbot_v2.euler",
    "lingbot_v2.detail.expert_adanorm": "lingbot_v2.euler",
    "lingbot_v2.detail.expert_heads": "lingbot_v2.euler",
}

REQUIRED_INPUTS = (
    "input.pixel_values",
    "input.image_grid_thw",
    "input.image_masks",
    "input.input_ids",
    "input.lang_lens",
    "input.state",
    "input.noise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile one fixed LingBot V2 batch with model-ready inputs. "
            "The primary latency excludes checkpoint loading, preprocessing, "
            "tokenization, and host file I/O."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vision-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--patch-embed-backend",
        choices=("conv3d", "gemm"),
        default="conv3d",
    )
    parser.add_argument(
        "--linear-kernel",
        choices=("torch", "flashinfer"),
        default="torch",
    )
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-timed", type=int, default=50)
    parser.add_argument("--n-prof-steps", type=int, default=5)
    cuda_graph_group = parser.add_mutually_exclusive_group()
    cuda_graph_group.add_argument(
        "--use-cuda-graph",
        dest="use_cuda_graph",
        action="store_true",
        help="Capture and replay the fixed-shape Expert Euler loop (default).",
    )
    cuda_graph_group.add_argument(
        "--no-cuda-graph",
        dest="use_cuda_graph",
        action="store_false",
        help="Disable CUDA Graph for diagnostics.",
    )
    parser.set_defaults(use_cuda_graph=True)
    parser.add_argument("--no-roofline", action="store_true")
    parser.add_argument("--roofline-iters", type=int, default=30)
    parser.add_argument("--gemm-size", type=int, default=8192)
    parser.add_argument("--copy-bytes", type=int, default=4 * 1024**3)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("benchmark/lingbot_v2/traces"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark/lingbot_v2/lingbot_v2_profile.json"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(checkpoint: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("config.json", "model.safetensors.index.json"):
        path = checkpoint / name
        if path.is_file():
            result[name] = sha256_file(path)
    return result


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


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


def load_inputs(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"input artifact does not exist: {path}")
    tensors = load_file(path, device="cpu")
    missing = [name for name in REQUIRED_INPUTS if name not in tensors]
    if missing:
        raise ValueError(f"input artifact is missing tensors: {missing}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        file_metadata = dict(handle.metadata() or {})
    return tensors, file_metadata


def validate_contract(
    tensors: dict[str, torch.Tensor],
    config: LingBotVLA2Config,
) -> dict[str, int | list]:
    pixel_values = tensors["input.pixel_values"]
    image_grid_thw = tensors["input.image_grid_thw"]
    image_masks = tensors["input.image_masks"].bool()
    input_ids = tensors["input.input_ids"]
    lang_lens = tensors["input.lang_lens"]
    state = tensors["input.state"]
    noise = tensors["input.noise"]

    if pixel_values.ndim != 4:
        raise ValueError("pixel_values must have shape (B,N,P,D).")
    batch_size, num_images, max_patches, patch_dim = pixel_values.shape
    if batch_size != 1:
        raise ValueError(
            f"the official single-batch benchmark requires B=1, got {batch_size}."
        )
    if num_images != 3:
        raise ValueError(
            f"the official benchmark requires exactly three views, got {num_images}."
        )
    if patch_dim != config.vision.patch_vector_dim:
        raise ValueError(
            f"patch width {patch_dim} != config {config.vision.patch_vector_dim}."
        )
    if image_grid_thw.shape != (batch_size, num_images, 3):
        raise ValueError("image_grid_thw must have shape (1,3,3).")
    if image_masks.shape != (batch_size, num_images) or not bool(image_masks.all()):
        raise ValueError("all three benchmark camera views must be active.")
    patch_counts = image_grid_thw.prod(dim=-1)
    if not bool((patch_counts == 256).all()):
        raise ValueError(
            "the fixed 256x256 benchmark requires 256 packed patches per view; "
            f"got {patch_counts.tolist()}."
        )
    if max_patches < 256:
        raise ValueError(f"pixel storage holds only {max_patches} patches per view.")
    if input_ids.shape != (1, config.tokenizer_max_length):
        raise ValueError(
            f"input_ids shape {tuple(input_ids.shape)} does not match "
            f"(1,{config.tokenizer_max_length})."
        )
    if lang_lens.shape != (1,):
        raise ValueError("lang_lens must have shape (1,).")
    lang_len = int(lang_lens[0])
    if not 0 < lang_len <= config.tokenizer_max_length:
        raise ValueError(f"invalid fixed prompt length: {lang_len}.")
    if state.shape != (1, config.max_state_dim):
        raise ValueError(
            f"state shape {tuple(state.shape)} != (1,{config.max_state_dim})."
        )
    if noise.shape != (1, config.chunk_size, config.max_action_dim):
        raise ValueError(
            f"noise shape {tuple(noise.shape)} != "
            f"(1,{config.chunk_size},{config.max_action_dim})."
        )
    if config.chunk_size != 50 or config.num_inference_steps != 10:
        raise ValueError(
            "the fixed LingBot V2 benchmark requires chunk_size=50 and "
            f"num_inference_steps=10, got {config.chunk_size} and "
            f"{config.num_inference_steps}."
        )
    return {
        "batch_size": batch_size,
        "num_images": num_images,
        "max_patches_per_image": max_patches,
        "patches_per_image": 256,
        "patch_vector_dim": patch_dim,
        "lang_len": lang_len,
        "input_ids": input_ids[0, :lang_len].tolist(),
        "chunk_size": config.chunk_size,
        "num_inference_steps": config.num_inference_steps,
        "action_dim": config.action_dim,
    }


def dims_from_config(
    config: LingBotVLA2Config,
    contract: dict[str, int | list],
) -> mf.LingBotV2Dims:
    return mf.LingBotV2Dims(
        vision_hidden=config.vision.hidden_size,
        vision_layers=config.vision.depth,
        vision_heads=config.vision.num_heads,
        vision_intermediate=config.vision.intermediate_size,
        patch_vector_dim=config.vision.patch_vector_dim,
        spatial_merge_unit=config.vision.spatial_merge_unit,
        vision_out_hidden=config.vision.out_hidden_size,
        deepstack_mergers=len(config.vision.deepstack_visual_indexes),
        text_hidden=config.text.hidden_size,
        text_layers=config.text.num_hidden_layers,
        text_heads=config.text.num_attention_heads,
        text_kv_heads=config.text.num_key_value_heads,
        text_head_dim=config.text.head_dim,
        text_intermediate=config.text.intermediate_size,
        expert_hidden=config.expert.hidden_size,
        expert_layers=config.expert.num_hidden_layers,
        expert_heads=config.expert.num_attention_heads,
        expert_kv_heads=config.expert.num_key_value_heads,
        expert_head_dim=config.expert.head_dim,
        num_experts=config.moe.num_experts,
        top_k=config.moe.top_k,
        moe_intermediate=config.moe.moe_intermediate_size,
        shared_intermediate=config.moe.shared_expert_intermediate_size,
        num_images=int(contract["num_images"]),
        patches_per_image=int(contract["patches_per_image"]),
        vision_boundaries_per_image=2 if config.use_vision_boundaries else 0,
        current_query_tokens=config.dual_query.current_query_token_count,
        future_query_tokens=config.dual_query.future_query_token_count,
        chunk_size=config.chunk_size,
        action_dim=config.action_dim,
        max_state_dim=config.max_state_dim,
        num_inference_steps=config.num_inference_steps,
    )


def make_request(
    tensors: dict[str, torch.Tensor],
    *,
    device: torch.device,
    params_dtype: torch.dtype,
) -> LingBotV2Request:
    """Place weight-bearing inputs on CUDA and keep scalar metadata on CPU."""

    return LingBotV2Request(
        pixel_values=tensors["input.pixel_values"].to(
            device=device,
            dtype=params_dtype,
        ),
        image_grid_thw=tensors["input.image_grid_thw"].to(
            device="cpu",
            dtype=torch.int64,
        ),
        image_masks=tensors["input.image_masks"].to(
            device="cpu",
            dtype=torch.bool,
        ),
        input_ids=tensors["input.input_ids"].to(
            device=device,
            dtype=torch.int64,
        ),
        lang_lens=tensors["input.lang_lens"].to(
            device="cpu",
            dtype=torch.int64,
        ),
        state=tensors["input.state"].to(
            device=device,
            dtype=params_dtype,
        ),
        noise=tensors["input.noise"].to(
            device=device,
            dtype=params_dtype,
        ),
    )


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


def time_engine(
    engine: Engine,
    request: LingBotV2Request,
    *,
    n_warmup: int,
    n_timed: int,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor, dict[str, float | bool]]:
    if n_warmup < 1 or n_timed < 1:
        raise ValueError("n_warmup and n_timed must both be positive.")
    output = engine.step(request)
    for _ in range(n_warmup - 1):
        output = engine.step(request)
    torch.cuda.synchronize(device)

    repeat = engine.step(request)
    torch.cuda.synchronize(device)
    deterministic = {
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
    if not deterministic["all_finite"]:
        raise RuntimeError("warm benchmark output contains non-finite values.")

    torch.cuda.reset_peak_memory_stats(device)
    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(n_timed):
        start.record()
        output = engine.step(request)
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))
    memory = {
        "allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
        "reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
    }
    return latency_stats(times), output, {**deterministic, **memory}


class CudaEventRecorder:
    """Collect low-overhead elapsed times on the current CUDA stream."""

    def __init__(self, labels: tuple[str, ...]) -> None:
        self.labels = frozenset(labels)
        self.pairs: dict[
            str,
            list[tuple[torch.cuda.Event, torch.cuda.Event]],
        ] = defaultdict(list)

    def start(self, label: str) -> torch.cuda.Event | None:
        if label not in self.labels:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def finish(self, label: str, start: torch.cuda.Event | None) -> None:
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.pairs[label].append((start, end))

    @contextmanager
    def scope(self, label: str, **_metadata: object) -> Iterator[None]:
        start = self.start(label)
        try:
            yield
        finally:
            self.finish(label, start)

    def average_ms(
        self,
        labels: tuple[str, ...],
        *,
        n_steps: int,
    ) -> dict[str, float]:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        return {
            label: (
                sum(start.elapsed_time(end) for start, end in self.pairs[label])
                / n_steps
            )
            for label in labels
        }


def install_detail_scopes(
    engine: Engine,
    recorder: CudaEventRecorder,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Install temporary CUDA-event hooks without changing model math."""

    model = engine.entry.model
    if model is None:
        raise RuntimeError("LingBot V2 model is unavailable for profiling.")
    handles: list[torch.utils.hooks.RemovableHandle] = []
    event_stacks: dict[int, list[torch.cuda.Event | None]] = defaultdict(list)

    def register(module: torch.nn.Module, label: str) -> None:
        module_id = id(module)

        def before(_module, _args) -> None:
            event_stacks[module_id].append(recorder.start(label))

        def after(_module, _args, _output) -> None:
            recorder.finish(label, event_stacks[module_id].pop())

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after, always_call=True))

    register(
        model.vision.patch_embed,
        "lingbot_v2.detail.vision_patch_embed",
    )
    for block in model.vision.blocks:
        register(block, "lingbot_v2.detail.vision_blocks")
    register(model.vision.merger, "lingbot_v2.detail.vision_mergers")
    for merger in model.vision.deepstack_merger_list:
        register(merger, "lingbot_v2.detail.vision_mergers")

    for layer in model.text.layers:
        register(layer.qkv_proj, "lingbot_v2.detail.text_attention")
        register(layer.attn, "lingbot_v2.detail.text_attention")
        register(layer.o_proj, "lingbot_v2.detail.text_attention")
        register(layer.mlp, "lingbot_v2.detail.text_mlp")

    for layer in model.expert_stack.layers:
        register(layer.qkv_proj, "lingbot_v2.detail.expert_attention")
        register(layer.attn, "lingbot_v2.detail.expert_attention")
        register(layer.o_proj, "lingbot_v2.detail.expert_attention")
        register(layer.mlp, "lingbot_v2.detail.expert_moe")
        register(
            layer.input_layernorm,
            "lingbot_v2.detail.expert_adanorm",
        )
        register(
            layer.post_attention_layernorm,
            "lingbot_v2.detail.expert_adanorm",
        )

    for module in (
        model.heads.state_proj,
        model.heads.action_in_proj,
        model.heads.action_time_mlp_in,
        model.heads.action_time_mlp_out,
        model.heads.action_out_proj,
        model.expert_stack.norm,
    ):
        register(module, "lingbot_v2.detail.expert_heads")
    return handles


def run_cuda_event_pass(
    engine: Engine,
    request: LingBotV2Request,
    *,
    n_steps: int,
    device: torch.device,
    include_details: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    labels = STAGES + DETAIL_STAGES if include_details else STAGES
    recorder = CudaEventRecorder(labels)
    handles = install_detail_scopes(engine, recorder) if include_details else []
    original_event_scope = scheduler_module.event_scope
    scheduler_module.event_scope = recorder.scope
    try:
        for _ in range(n_steps):
            engine.step(request)
        torch.cuda.synchronize(device)
    finally:
        scheduler_module.event_scope = original_event_scope
        for handle in handles:
            handle.remove()

    stages = recorder.average_ms(STAGES, n_steps=n_steps)
    details = (
        recorder.average_ms(DETAIL_STAGES, n_steps=n_steps)
        if include_details
        else {stage: 0.0 for stage in DETAIL_STAGES}
    )
    return stages, details


def profile_stages(
    engine: Engine,
    request: LingBotV2Request,
    *,
    trace_dir: Path,
    n_steps: int,
    device: torch.device,
    include_details: bool = True,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    str | None,
]:
    del trace_dir
    engine.step(request)
    torch.cuda.synchronize(device)

    stages, _ = run_cuda_event_pass(
        engine,
        request,
        n_steps=n_steps,
        device=device,
        include_details=False,
    )
    if include_details:
        detail_parent_stages, raw_details = run_cuda_event_pass(
            engine,
            request,
            n_steps=n_steps,
            device=device,
            include_details=True,
        )
        details = pm.scale_details_to_parent_stages(
            raw_details,
            detail_parent_stages,
            stages,
            DETAIL_PARENT,
        )
    else:
        # Forward hooks execute during capture, not replay. Reporting their
        # graph-on values would therefore be incomplete and misleading.
        details = {stage: 0.0 for stage in DETAIL_STAGES}
        raw_details = dict(details)
        detail_parent_stages = {}
    return stages, details, raw_details, detail_parent_stages, None


def achieved_tflops(flop: float, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return (flop / 1e12) / (latency_ms / 1e3)


def cuda_graph_status(engine: Engine) -> dict[str, bool | int]:
    """Report the Expert runner's actual graph state, not just the CLI flag."""

    runner = engine.entry.scheduler.expert_runner
    return {
        "requested": bool(runner.cuda_graph_requested),
        "enabled": bool(runner.use_cuda_graph),
        "active": bool(runner.cuda_graph_active),
        "captured_graphs": int(runner.cuda_graph_count),
        "fallback_layouts": int(runner.cuda_graph_fallback_count),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the LingBot V2 benchmark.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    params_dtype = torch.bfloat16
    vision_dtype = torch.bfloat16 if args.vision_dtype == "bf16" else torch.float32

    config = load_config(args.checkpoint, LingBotVLA2Config)
    tensors, input_metadata = load_inputs(args.input)
    contract = validate_contract(tensors, config)
    dims = dims_from_config(config, contract)
    flop = mf.stage_flops(dims, lang_len=int(contract["lang_len"]))

    if args.no_roofline:
        hardware = hp.probe_device(device.index or 0)
    else:
        hardware = hp.measure_roofline(
            device=device.index or 0,
            gemm_size=args.gemm_size,
            copy_bytes=args.copy_bytes,
            iters=args.roofline_iters,
        )
        torch.cuda.empty_cache()

    active_patch_counts = tensors["input.image_grid_thw"].prod(dim=-1)[
        tensors["input.image_masks"].bool()
    ]
    max_vision_tokens = int(
        (active_patch_counts // config.vision.spatial_merge_unit).max()
    )
    request = make_request(
        tensors,
        device=device,
        params_dtype=params_dtype,
    )

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
                device=DeviceConfig(
                    target=str(device),
                    params_dtype=params_dtype,
                ),
                runtime=RuntimeConfig(
                    use_cuda_graph=args.use_cuda_graph,
                    force_linear_kernel=args.linear_kernel,
                ),
            ),
        )
    )
    try:
        latency, output, checks = time_engine(
            engine,
            request,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
            device=device,
        )
        graph_status = cuda_graph_status(engine)
        if args.use_cuda_graph and not graph_status["active"]:
            raise RuntimeError(
                "CUDA Graph was requested, but the Expert runner did not retain "
                "a captured graph. Inspect the preceding capture-fallback warning."
            )
        (
            stages,
            details,
            raw_details,
            detail_parent_stages,
            trace_path,
        ) = profile_stages(
            engine,
            request,
            trace_dir=args.trace_dir,
            n_steps=args.n_prof_steps,
            device=device,
            include_details=not args.use_cuda_graph,
        )
    finally:
        engine.close()

    compute_stage_flop = {
        "lingbot_v2.vision": flop["vision"],
        "lingbot_v2.prefix_forward": flop["text_prefix"],
        "lingbot_v2.euler": flop["expert_loop"],
    }
    detail_flop = {
        "lingbot_v2.detail.vision_patch_embed": flop["vision_patch_embed"],
        "lingbot_v2.detail.vision_blocks": flop["vision_blocks"],
        "lingbot_v2.detail.vision_mergers": flop["vision_mergers"],
        "lingbot_v2.detail.text_attention": flop["text_attention"],
        "lingbot_v2.detail.text_mlp": flop["text_mlp"],
        "lingbot_v2.detail.expert_attention": (
            flop["expert_attention_1step"] * config.num_inference_steps
        ),
        "lingbot_v2.detail.expert_moe": (
            (flop["expert_moe_1step"] + flop["expert_router_1step"])
            * config.num_inference_steps
        ),
        "lingbot_v2.detail.expert_adanorm": (
            flop["expert_adanorm_1step"] * config.num_inference_steps
        ),
        "lingbot_v2.detail.expert_heads": (
            flop["expert_heads_1step"] * config.num_inference_steps
        ),
    }
    achieved = {
        stage: achieved_tflops(stage_flop, stages[stage])
        for stage, stage_flop in compute_stage_flop.items()
    }
    peak = hardware.get("peak_bf16_tflops")
    stage_mfu = {
        stage: (100.0 * value / peak if peak else None)
        for stage, value in achieved.items()
    }
    detail_achieved = {
        stage: achieved_tflops(stage_flop, details[stage])
        for stage, stage_flop in detail_flop.items()
    }
    detail_mfu = {
        stage: (100.0 * value / peak if peak else None)
        for stage, value in detail_achieved.items()
    }
    e2e_achieved = achieved_tflops(flop["e2e_compute"], latency["mean"])
    e2e_mfu = 100.0 * e2e_achieved / peak if peak else None
    diagnostics = pm.profile_diagnostics(latency["mean"], stages)
    detail_profile_available = not args.use_cuda_graph
    detail_diagnostics = (
        pm.profile_diagnostics(latency["mean"], detail_parent_stages)
        if detail_profile_available
        else None
    )
    profile_valid = bool(diagnostics["valid"])
    detail_profile_valid = bool(
        detail_diagnostics is not None and detail_diagnostics["valid"]
    )
    stage_share = pm.normalized_share_pct(stages)
    profiled_stage_sum = float(diagnostics["profiled_stage_sum_ms"])
    detail_share = {
        stage: (100.0 * value / profiled_stage_sum if profiled_stage_sum > 0 else 0.0)
        for stage, value in details.items()
    }
    stage_sum = profiled_stage_sum

    result = {
        "schema_version": 3,
        "meta": {
            "model": "lingbot_v2",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(),
            "command": sys.argv,
            "checkpoint": str(args.checkpoint),
            "checkpoint_fingerprint": checkpoint_fingerprint(args.checkpoint),
            "input": str(args.input),
            "input_sha256": sha256_file(args.input),
            "input_metadata": input_metadata,
            "contract": contract,
            "params_dtype": str(params_dtype),
            "vision_dtype": str(vision_dtype),
            "attention_backend": "vision=flashinfer; prefix/expert=official_eager",
            "vision_patch_embed_backend": args.patch_embed_backend,
            "linear_kernel": args.linear_kernel,
            "use_cuda_graph": args.use_cuda_graph,
            "cuda_graph_status": graph_status,
            "torch_compile": False,
            "n_warmup": args.n_warmup,
            "n_timed": args.n_timed,
            "n_prof_steps": args.n_prof_steps,
            "profile_backend": "cuda_events",
            "profile_with_stack": False,
            "profile_passes": 1 if args.use_cuda_graph else 2,
            "detail_profile_available": detail_profile_available,
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flashinfer": package_version("flashinfer-python", "flashinfer"),
            "transformers": package_version("transformers"),
            "safetensors": package_version("safetensors"),
        },
        "hardware": hardware,
        "latency_ms": latency,
        "stage_gpu_ms": stages,
        "stage_latency_share_pct": stage_share,
        "detail_gpu_ms": details if detail_profile_available else None,
        "detail_cuda_event_raw_ms": (raw_details if detail_profile_available else None),
        "detail_profile_stage_gpu_ms": (
            detail_parent_stages if detail_profile_available else None
        ),
        "detail_latency_share_pct": (
            detail_share if detail_profile_available else None
        ),
        "stage_sum_ms": stage_sum,
        "unattributed_ms": latency["mean"] - stage_sum,
        "profile_diagnostics": diagnostics,
        "detail_profile_diagnostics": detail_diagnostics,
        "euler_step_mean_ms": stages["lingbot_v2.euler"] / config.num_inference_steps,
        "flop_per_sample": flop,
        "achieved_tflops": {
            **achieved,
            **(
                detail_achieved
                if detail_profile_available
                else {stage: None for stage in detail_achieved}
            ),
            "e2e": e2e_achieved,
        },
        "mfu_pct_bf16_peak": {
            **{
                name: value if profile_valid else None
                for name, value in stage_mfu.items()
            },
            **{
                name: value if profile_valid and detail_profile_valid else None
                for name, value in detail_mfu.items()
            },
            "e2e": e2e_mfu,
        },
        "checks": {
            **checks,
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
        },
        "trace": trace_path,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nLingBot V2 benchmark contract")
    print(json.dumps(contract, indent=2))
    print(f"\nGPU                 : {hardware['name']}")
    print(f"input SHA256        : {result['meta']['input_sha256']}")
    print(f"vision dtype        : {vision_dtype}")
    print(f"attention backend   : {result['meta']['attention_backend']}")
    print(f"PatchEmbed backend  : {args.patch_embed_backend}")
    print(f"linear kernel       : {args.linear_kernel}")
    print(
        "CUDA Graph          : "
        f"requested={graph_status['requested']} "
        f"enabled={graph_status['enabled']} "
        f"active={graph_status['active']} "
        f"captured={graph_status['captured_graphs']} "
        f"fallbacks={graph_status['fallback_layouts']}"
    )
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
        "profile stage/baseline: "
        f"{stage_sum:.3f} / {latency['mean']:.3f} ms  "
        f"ratio={float(diagnostics['overhead_ratio']):.3f}  "
        f"valid={profile_valid}"
    )
    if detail_diagnostics is not None:
        print(
            "detail stage/baseline : "
            f"{float(detail_diagnostics['profiled_stage_sum_ms']):.3f} / "
            f"{latency['mean']:.3f} ms  "
            f"ratio={float(detail_diagnostics['overhead_ratio']):.3f}  "
            f"valid={detail_profile_valid}"
        )
    if not profile_valid:
        print(
            "WARNING: component latency and component MFU are diagnostic only; "
            "instrumentation perturbation exceeded the 15% validity limit."
        )
    if detail_profile_available and not detail_profile_valid:
        print(
            "WARNING: detailed latency and detailed MFU are diagnostic only; "
            "detail-hook perturbation exceeded the 15% validity limit."
        )
    for stage in STAGES:
        stage_mfu_value = result["mfu_pct_bf16_peak"].get(stage)
        mfu_text = (
            f"  MFU={stage_mfu_value:>7.2f}%" if stage_mfu_value is not None else ""
        )
        print(
            f"{stage:<29}: {stages[stage]:>10.3f} ms  "
            f"{stage_share[stage]:>7.2f}%{mfu_text}"
        )
    if detail_profile_available:
        print("\nDetailed model components")
        for stage in DETAIL_STAGES:
            label = stage.removeprefix("lingbot_v2.detail.")
            detail_mfu_value = result["mfu_pct_bf16_peak"][stage]
            mfu_text = (
                f"  MFU={detail_mfu_value:>7.2f}%"
                if detail_mfu_value is not None
                else ""
            )
            print(
                f"{label:<29}: {details[stage]:>10.3f} ms  "
                f"{detail_share[stage]:>7.2f}%{mfu_text}"
            )
    print(f"{'unattributed':<29}: {result['unattributed_ms']:>10.3f} ms")
    print("Euler per denoise step: " f"{result['euler_step_mean_ms']:.3f} ms")
    if peak:
        print(f"measured BF16 peak  : {peak:.1f} TFLOPS")
        for stage in compute_stage_flop:
            print(f"{stage} MFU".ljust(29) + f": {stage_mfu[stage]:.2f}%")
        print(f"{'E2E effective MFU':<29}: {e2e_mfu:.2f}%")
    print(
        "determinism         : "
        f"exact={checks['exact']} cosine={checks['cosine']:.9f} "
        f"max_abs={checks['max_abs']:.3e}"
    )
    print(f"report              : {args.out}")
    print(f"trace               : {trace_path}")


if __name__ == "__main__":
    main()
