from __future__ import annotations

import argparse
import gzip
import json
import statistics
import subprocess
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

import hardware_probe as hp
import model_flops as mf

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi0.configuration_pi0 import PI0Config
from phyai.models.pi0.main_pi0 import PI0Args
from phyai.models.pi0.scheduler_ws1_pi0 import PI0Request
from phyai.utils import load_config
from phyai.utils.profile import ProfilerConfig, get_profiler, install_profiler


STAGES = [
    "pi0.vision_loop",
    "pi0.lang_pack",
    "pi0.llm_prefix_plan",
    "pi0.llm_prefix_fwd",
    "pi0.expert_plan",
    "pi0.expert_loop",
]
EXPERT_STEP_STAGE = "pi0.expert_step"


def dims_from_config(cfg: PI0Config) -> mf.Pi0Dims:
    """Build the FLOP-model dims from a live :class:`PI0Config`."""
    return mf.Pi0Dims(
        v_hidden=cfg.vision.hidden_size,
        v_layers=cfg.vision.num_hidden_layers,
        v_heads=cfg.vision.num_attention_heads,
        v_intermediate=cfg.vision.intermediate_size,
        image_size=cfg.vision.image_size,
        patch_size=cfg.vision.patch_size,
        num_channels=cfg.vision.num_channels,
        l_hidden=cfg.text.hidden_size,
        l_layers=cfg.text.num_hidden_layers,
        l_heads=cfg.text.num_attention_heads,
        l_kv_heads=cfg.text.num_key_value_heads,
        l_head_dim=cfg.text.head_dim,
        l_intermediate=cfg.text.intermediate_size,
        e_hidden=cfg.expert.hidden_size,
        e_layers=cfg.expert.num_hidden_layers,
        e_heads=cfg.expert.num_attention_heads,
        e_kv_heads=cfg.expert.num_key_value_heads,
        e_head_dim=cfg.expert.head_dim,
        e_intermediate=cfg.expert.intermediate_size,
        chunk_size=cfg.chunk_size,
        max_state_dim=cfg.max_state_dim,
        max_action_dim=cfg.max_action_dim,
        num_inference_steps=cfg.num_inference_steps,
        tokenizer_max_length=cfg.tokenizer_max_length,
        num_images=cfg.num_images,
    )


def _module_bytes(
    module: torch.nn.Module, *, exclude_prefixes: tuple[str, ...] = ()
) -> int:
    total = 0
    for name, p in module.named_parameters():
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        total += p.numel() * p.element_size()
    return total


def stage_weight_bytes(model) -> dict[str, float]:
    """Exact resident weight bytes streamed by each pi0 stage."""
    n_steps = model.config.num_inference_steps
    vision_b = _module_bytes(model.vision)
    llm_b = _module_bytes(model.paligemma_lm, exclude_prefixes=("embed_tokens",))
    expert_b = _module_bytes(model.expert_stack) + _module_bytes(model.heads)
    return {
        "vision": float(vision_b),
        "llm_prefix": float(llm_b),
        "expert_1step": float(expert_b),
        "expert_loop": float(expert_b * n_steps),
    }


def make_request(
    bs: int,
    cfg: PI0Config,
    device: torch.device,
    dtype: torch.dtype,
    lang_len: int,
    *,
    fixed_noise: bool,
) -> PI0Request:
    """Random pixels, prompt ids, state, and optional fixed noise for ``bs`` robots."""
    pixel_values = torch.rand(
        bs,
        cfg.num_images,
        cfg.vision.num_channels,
        cfg.vision.image_size,
        cfg.vision.image_size,
        dtype=dtype,
        device=device,
    )
    input_ids = torch.zeros(
        bs, cfg.tokenizer_max_length, dtype=torch.int64, device=device
    )
    input_ids[:, :lang_len] = 2
    lang_lens = torch.full((bs,), lang_len, dtype=torch.int64, device=device)
    state = torch.rand(bs, cfg.max_state_dim, dtype=dtype, device=device)

    noise = None
    if fixed_noise:
        gen = torch.Generator(device=device).manual_seed(0)
        noise = torch.randn(
            bs,
            cfg.chunk_size,
            cfg.max_action_dim,
            dtype=dtype,
            device=device,
            generator=gen,
        )

    return PI0Request(
        pixel_values=pixel_values,
        input_ids=input_ids,
        lang_lens=lang_lens,
        state=state,
        noise=noise,
    )


def time_e2e(engine, request: PI0Request, n_warmup: int, n_timed: int) -> dict:
    """Cuda-event end-to-end ``step`` latency stats."""
    for _ in range(n_warmup):
        engine.step(request)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        engine.step(request)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {
        "mean": statistics.fmean(times),
        "p50": times[len(times) // 2],
        "p90": times[int(len(times) * 0.9)],
        "p99": times[min(len(times) - 1, int(len(times) * 0.99))],
        "std": statistics.pstdev(times),
        "min": times[0],
        "max": times[-1],
    }


def parse_stage_gpu_ms(trace_path: Path, n_steps: int) -> dict[str, float]:
    """Per-profiled-step GPU ms for each pi0 scope from a chrome trace."""
    opener = gzip.open if str(trace_path).endswith(".gz") else open
    with opener(trace_path, "rt") as f:
        trace = json.load(f)
    events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
    agg: dict[str, float] = defaultdict(float)
    expert_step_us = 0.0
    for ev in events:
        if ev.get("cat") != "gpu_user_annotation":
            continue
        name = ev.get("name")
        dur = float(ev.get("dur", 0.0))
        if name in STAGES:
            agg[name] += dur
        elif name == EXPERT_STEP_STAGE:
            expert_step_us += dur

    result = {stage: agg.get(stage, 0.0) / 1e3 / max(n_steps, 1) for stage in STAGES}
    # CUDA graph replay can make the outer loop annotation only cover replay overhead.
    # The ten expert_step annotations are the reliable GPU sum for the denoising loop.
    expert_from_steps = expert_step_us / 1e3 / max(n_steps, 1)
    if expert_from_steps > result["pi0.expert_loop"]:
        result["pi0.expert_loop"] = expert_from_steps
    return result


def profile_stages(engine, request: PI0Request, prof_dir: Path, n_prof_steps: int):
    """Run the torch profiler and parse stage-level GPU ms."""
    prof_dir.mkdir(parents=True, exist_ok=True)
    install_profiler(ProfilerConfig(backend="torch", output_dir=prof_dir))
    prof = get_profiler()
    engine.step(request)
    torch.cuda.synchronize()
    prof.start()
    for _ in range(n_prof_steps):
        engine.step(request)
    torch.cuda.synchronize()
    prof.stop()
    traces = sorted(prof_dir.glob("*.trace.json*"), key=lambda p: p.stat().st_mtime)
    if not traces:
        return {stage: 0.0 for stage in STAGES}, None
    latest = traces[-1]
    return parse_stage_gpu_ms(latest, n_prof_steps), str(latest)


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def is_oom(exc: Exception) -> bool:
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in str(exc).lower()
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="HF-style pi0 checkpoint folder. Omit for random-weight smoke timing.",
    )
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--lang-len", type=int, default=1)
    ap.add_argument(
        "--num-images",
        type=int,
        choices=(2, 3),
        default=None,
        help="Override camera count. Default comes from checkpoint config.",
    )
    ap.add_argument(
        "--vision-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Vision tower compute dtype. pi0 defaults to fp32 for parity.",
    )
    ap.add_argument("--fixed-noise", action="store_true")
    ap.add_argument(
        "--workspace-bytes",
        type=int,
        default=512 * 1024**2,
        help="flashinfer workspace bytes.",
    )
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-timed", type=int, default=50)
    ap.add_argument("--n-prof-steps", type=int, default=5)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--no-roofline", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("benchmark/pi0/pi0_profile.json"))
    ap.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("benchmark/pi0/traces"),
        help="Where to write per-batch chrome traces.",
    )
    args = ap.parse_args()

    if args.checkpoint is not None and not args.checkpoint.is_dir():
        raise NotADirectoryError(f"--checkpoint must be a directory: {args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark.")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    vision_dtype = torch.float32 if args.vision_dtype == "float32" else torch.bfloat16
    if args.checkpoint is not None:
        cfg = load_config(args.checkpoint, PI0Config)
    else:
        cfg = PI0Config()
    if args.num_images is not None:
        cfg = replace(cfg, empty_cameras=3 - args.num_images)
    if not 0 < args.lang_len <= cfg.tokenizer_max_length:
        raise ValueError(
            f"--lang-len must be in [1, {cfg.tokenizer_max_length}], got {args.lang_len}."
        )
    dims = dims_from_config(cfg)

    print("[hw] probing device + measuring roofline microbench...")
    hardware = (
        hp.probe_device(args.gpu_index)
        if args.no_roofline
        else hp.measure_roofline(device=args.gpu_index)
    )
    print(
        f"[hw] {hardware['name']}: "
        + (
            f"BF16 peak {hardware.get('peak_bf16_tflops')} TFLOPS, "
            f"HBM {hardware.get('hbm_tb_s')} TB/s, ridge "
            f"{hardware.get('ridge_point_flop_per_byte')} FLOP/byte"
            if not args.no_roofline
            else "(roofline microbench skipped)"
        )
    )

    flop = mf.stage_flops(dims, lang_len=args.lang_len)
    peak_tflops = hardware.get("peak_bf16_tflops")

    sweep = []
    weight_bytes = None
    for bs in args.batch_sizes:
        print(
            f"\n{'=' * 64}\n[bs={bs}] building engine "
            f"(workspace={args.workspace_bytes / 2**20:.0f} MiB, "
            f"vision={args.vision_dtype})..."
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            engine = Engine(
                EngineArgs(
                    plugin="pi0",
                    plugin_args=PI0Args(
                        checkpoint_dir=args.checkpoint,
                        config=cfg,
                        max_batch_size=bs,
                        vision_params_dtype=vision_dtype,
                    ),
                    config=EngineConfig(
                        device=DeviceConfig(target="cuda", params_dtype=dtype),
                        runtime=RuntimeConfig(
                            use_cuda_graph=True,
                            flashinfer_workspace_bytes=args.workspace_bytes,
                        ),
                    ),
                )
            )
            if weight_bytes is None:
                weight_bytes = stage_weight_bytes(engine.entry.model)
            request = make_request(
                bs,
                cfg,
                device,
                dtype,
                args.lang_len,
                fixed_noise=args.fixed_noise,
            )

            lat = time_e2e(engine, request, args.n_warmup, args.n_timed)
            mem_alloc = torch.cuda.max_memory_allocated() / 2**20
            mem_resv = torch.cuda.max_memory_reserved() / 2**20
            stages, trace_file = profile_stages(
                engine, request, args.trace_dir / f"bs{bs}", args.n_prof_steps
            )
        except Exception as exc:  # noqa: BLE001
            if "engine" in locals() and engine is not None:
                engine.close()
            if is_oom(exc):
                print(f"[bs={bs}] OOM -- stopping sweep here.")
                torch.cuda.empty_cache()
                break
            raise

        def achieved_tflops(stage_flop: float, stage_ms: float) -> float:
            return (bs * stage_flop / 1e12) / (stage_ms / 1e3) if stage_ms > 0 else 0.0

        ach = {
            "vision": achieved_tflops(flop["vision"], stages["pi0.vision_loop"]),
            "llm_prefix": achieved_tflops(
                flop["llm_prefix"], stages["pi0.llm_prefix_fwd"]
            ),
            "expert": achieved_tflops(flop["expert_loop"], stages["pi0.expert_loop"]),
        }
        ai = {
            "vision": bs * flop["vision"] / weight_bytes["vision"],
            "llm_prefix": bs * flop["llm_prefix"] / weight_bytes["llm_prefix"],
            "expert": bs * flop["expert_loop"] / weight_bytes["expert_loop"],
        }
        expert_mfu = (100.0 * ach["expert"] / peak_tflops) if peak_tflops else None
        throughput = bs * 1000.0 / lat["mean"]

        sweep.append(
            {
                "bs": bs,
                "latency_ms": lat,
                "per_sample_ms": lat["mean"] / bs,
                "throughput_sps": throughput,
                "action_steps_per_s": cfg.chunk_size * throughput,
                "mem_alloc_mib": mem_alloc,
                "mem_reserved_mib": mem_resv,
                "stage_gpu_ms": stages,
                "stage_sum_ms": sum(stages.values()),
                "achieved_tflops": ach,
                "arithmetic_intensity": ai,
                "expert_mfu_pct": expert_mfu,
                "trace": trace_file,
            }
        )
        print(
            f"[bs={bs}] e2e mean={lat['mean']:.2f}ms p99={lat['p99']:.2f}ms "
            f"tp={throughput:.1f}/s per_sample={lat['mean'] / bs:.3f}ms "
            f"mem={mem_alloc:.0f}MiB"
        )
        print(
            f"[bs={bs}] stages(ms): "
            + "  ".join(
                f"{stage.split('.')[1]}={stages[stage]:.2f}" for stage in STAGES
            )
        )
        print(
            f"[bs={bs}] expert: {ach['expert']:.1f} TFLOPS"
            + (f" ({expert_mfu:.1f}% MFU)" if expert_mfu else "")
        )
        engine.close()
        del engine
        torch.cuda.empty_cache()

    out = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(),
            "model": "pi0",
            "scheduler": "ws1",
            "checkpoint": args.checkpoint.name if args.checkpoint else None,
            "dtype": "bfloat16",
            "vision_dtype": args.vision_dtype,
            "use_cuda_graph": True,
            "num_images": cfg.num_images,
            "lang_len": args.lang_len,
            "tokenizer_max_length": cfg.tokenizer_max_length,
            "image_tokens": dims.image_tokens,
            "n_per_sample": dims.n_per_sample,
            "chunk_size": cfg.chunk_size,
            "max_state_dim": cfg.max_state_dim,
            "max_action_dim": cfg.max_action_dim,
            "num_inference_steps": cfg.num_inference_steps,
            "fixed_noise": args.fixed_noise,
            "n_warmup": args.n_warmup,
            "n_timed": args.n_timed,
            "n_prof_steps": args.n_prof_steps,
            "flashinfer_workspace_mib": args.workspace_bytes // 2**20,
        },
        "hardware": hardware,
        "stages_flop": {
            "lang_len": args.lang_len,
            "flop_per_sample": flop,
            "weight_bytes": weight_bytes,
        },
        "sweep": sweep,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    print(
        f"\n{'bs':>4}{'e2e_ms':>10}{'p99':>8}{'tp/s':>9}{'per_smpl':>10}"
        f"{'mem_MiB':>10}{'vision':>9}{'llm_fwd':>9}{'expert':>9}{'exp_MFU':>9}"
    )
    for row in sweep:
        st = row["stage_gpu_ms"]
        mfu = f"{row['expert_mfu_pct']:.1f}%" if row["expert_mfu_pct"] else "-"
        print(
            f"{row['bs']:>4}{row['latency_ms']['mean']:>10.2f}"
            f"{row['latency_ms']['p99']:>8.2f}{row['throughput_sps']:>9.1f}"
            f"{row['per_sample_ms']:>10.3f}{row['mem_alloc_mib']:>10.0f}"
            f"{st['pi0.vision_loop']:>9.2f}{st['pi0.llm_prefix_fwd']:>9.2f}"
            f"{st['pi0.expert_loop']:>9.2f}{mfu:>9}"
        )


if __name__ == "__main__":
    main()
