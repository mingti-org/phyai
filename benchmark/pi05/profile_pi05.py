from __future__ import annotations

import argparse
import gzip
import json
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch

import model_flops as mf
import hardware_probe as hp

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request
from phyai.utils import load_config
from phyai.utils.profile import ProfilerConfig, get_profiler, install_profiler


# The six GPU-annotation scopes the pi0.5 scheduler emits per step. Kept in
# pipeline order; plan scopes are ~0 once the layout is cached (verified).
STAGES = [
    "pi05.vision_loop",
    "pi05.lang_pack",
    "pi05.llm_prefix_plan",
    "pi05.llm_prefix_fwd",
    "pi05.expert_plan",
    "pi05.expert_loop",
]


def dims_from_config(cfg: PI05Config) -> mf.Pi05Dims:
    """Build the FLOP-model dims from a live :class:`PI05Config`.

    Mirrors the scheduler's language buckets so the FLOP denominator matches
    the prefix length the captured graph actually ran at.
    """
    tok_max = cfg.tokenizer_max_length
    buckets = tuple(sorted({b for b in (16, 48, 112) if 0 < b < tok_max} | {tok_max}))
    return mf.Pi05Dims(
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
        num_inference_steps=cfg.num_inference_steps,
        tokenizer_max_length=tok_max,
        lang_buckets=buckets,
    )


def _module_bytes(
    module: torch.nn.Module, *, exclude_prefixes: tuple[str, ...] = ()
) -> int:
    """Sum ``numel * element_size`` over a module's params, skipping prefixes."""
    total = 0
    for name, p in module.named_parameters():
        if any(name.startswith(pre) for pre in exclude_prefixes):
            continue
        total += p.numel() * p.element_size()
    return total


def stage_weight_bytes(model) -> dict[str, float]:
    """Exact resident weight bytes streamed by each stage, from the loaded model.

    * ``vision`` — the whole vision tower + multi-modal projector.
    * ``llm_prefix`` — PaliGemma decoder layers + final norm. The tied
      ``embed_tokens`` is a gather (no GEMM read of the 0.5 B-param table), so
      it is excluded from the prefix's streamed bytes.
    * ``expert_1step`` — the expert decoder stack + action/time heads, read
      once per Euler step. ``expert_loop`` multiplies by ``num_inference_steps``.
    """
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


def make_request(bs, cfg, num_images, device, dtype, lang_len) -> PI05Request:
    """Random-pixel, single-(or lang_len)-token-prompt request for ``bs`` robots."""
    pixel_values = torch.rand(
        bs,
        num_images,
        3,
        cfg.vision.image_size,
        cfg.vision.image_size,
        dtype=dtype,
        device=device,
    )
    input_ids = torch.zeros(
        bs, cfg.tokenizer_max_length, dtype=torch.int64, device=device
    )
    input_ids[:, :lang_len] = 2  # any non-pad token id
    lang_lens = torch.full((bs,), lang_len, dtype=torch.int64, device=device)
    return PI05Request(
        pixel_values=pixel_values, input_ids=input_ids, lang_lens=lang_lens
    )


def time_e2e(engine, request, n_warmup, n_timed) -> dict:
    """Cuda-event end-to-end ``step`` latency stats over ``n_timed`` runs."""
    for _ in range(n_warmup):
        engine.step(request)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_timed):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        engine.step(request)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
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
    """Per-step GPU ms for each pi05.* scope from a chrome trace's GPU track."""
    op = gzip.open if str(trace_path).endswith(".gz") else open
    with op(trace_path, "rt") as f:
        trace = json.load(f)
    events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
    agg: dict[str, float] = defaultdict(float)
    for ev in events:
        if ev.get("cat") == "gpu_user_annotation" and ev.get("name") in STAGES:
            agg[ev["name"]] += float(ev.get("dur", 0.0))  # microseconds
    return {s: agg.get(s, 0.0) / 1e3 / max(n_steps, 1) for s in STAGES}  # → ms/step


def profile_stages(engine, request, prof_dir: Path, n_prof_steps: int):
    """Run the torch profiler over ``n_prof_steps`` and parse per-stage GPU ms."""
    prof_dir.mkdir(parents=True, exist_ok=True)
    install_profiler(ProfilerConfig(backend="torch", output_dir=prof_dir))
    prof = get_profiler()
    engine.step(request)  # one more warm step so capture is steady-state
    torch.cuda.synchronize()
    prof.start()
    for _ in range(n_prof_steps):
        engine.step(request)
    torch.cuda.synchronize()
    prof.stop()
    traces = sorted(prof_dir.glob("*.trace.json*"), key=lambda p: p.stat().st_mtime)
    if not traces:
        return {s: 0.0 for s in STAGES}, None
    latest = traces[-1]
    return parse_stage_gpu_ms(latest, n_prof_steps), str(latest)


def git_commit() -> str:
    """Short git commit of the repo, or 'unknown' if not resolvable."""
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="pi05_base checkpoint folder (config.json + safetensors).",
    )
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument(
        "--lang-len",
        type=int,
        default=1,
        help="Prompt token count (rounds up to a scheduler bucket).",
    )
    ap.add_argument("--num-images", type=int, default=3)
    ap.add_argument(
        "--workspace-bytes",
        type=int,
        default=4 * 1024**3,
        help="flashinfer workspace (apply-only-raises; 4 GiB clears "
        "FA2 split-tmp at bs>=8).",
    )
    ap.add_argument(
        "--bf16-gemm-backend",
        choices=("auto", "cudnn", "cutlass", "tgv", "cublaslt", "tinygemm"),
        default="cudnn",
        help="FlashInfer mm_bf16 backend. 'auto' enables per-shape runner selection.",
    )
    ap.add_argument(
        "--flashinfer-autotune",
        action="store_true",
        help="Enable FlashInfer autotuning during capture-safe eager warmup.",
    )
    ap.add_argument(
        "--flashinfer-autotune-cache",
        type=Path,
        default=None,
        help="Precomputed FlashInfer autotuner cache loaded during engine initialization.",
    )
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-timed", type=int, default=50)
    ap.add_argument("--n-prof-steps", type=int, default=5)
    ap.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Local device index (after CUDA_VISIBLE_DEVICES) to probe.",
    )
    ap.add_argument(
        "--no-roofline",
        action="store_true",
        help="Skip the microbench (hardware block omits peak/bw).",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("benchmark/pi05/pi05_profile.json")
    )
    ap.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("benchmark/pi05/traces"),
        help="Where to drop per-batch chrome traces (gitignored scratch).",
    )
    args = ap.parse_args()
    flashinfer_tune_cache = args.flashinfer_autotune_cache

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark.")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = load_config(args.checkpoint, PI05Config)
    dims = dims_from_config(cfg)

    # --- Hardware facts + measured roofline (once, before the model fills VRAM). ---
    print("[hw] probing device + measuring roofline microbench...")
    if args.no_roofline:
        hardware = hp.probe_device(args.gpu_index)
    else:
        hardware = hp.measure_roofline(device=args.gpu_index)
    gpu_name = hardware["name"]
    print(
        f"[hw] {gpu_name}: "
        + (
            f"BF16 peak {hardware.get('peak_bf16_tflops')} TFLOPS, "
            f"HBM {hardware.get('hbm_tb_s')} TB/s, ridge "
            f"{hardware.get('ridge_point_flop_per_byte')} FLOP/byte"
            if not args.no_roofline
            else "(roofline microbench skipped)"
        )
    )

    flop = mf.stage_flops(dims, lang_len=args.lang_len, num_images=args.num_images)
    peak_tflops = hardware.get("peak_bf16_tflops")

    sweep = []
    weight_bytes = None
    for bs in args.batch_sizes:
        print(
            f"\n{'=' * 64}\n[bs={bs}] building engine "
            f"(workspace={args.workspace_bytes / 2**20:.0f} MiB)..."
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        engine = Engine(
            EngineArgs(
                plugin="pi05",
                plugin_args=PI05Args(
                    checkpoint_dir=args.checkpoint,
                    max_batch_size=bs,
                    inputs_image_shape=[
                        [cfg.vision.image_size, cfg.vision.image_size, 3]
                        for _ in range(args.num_images)
                    ],
                ),
                config=EngineConfig(
                    device=DeviceConfig(target="cuda", params_dtype=dtype),
                    runtime=RuntimeConfig(
                        use_cuda_graph=True,
                        flashinfer_workspace_bytes=args.workspace_bytes,
                        flashinfer_bf16_backend=args.bf16_gemm_backend,
                        flashinfer_autotune=args.flashinfer_autotune,
                        flashinfer_autotune_cache=(
                            str(flashinfer_tune_cache)
                            if flashinfer_tune_cache is not None
                            else None
                        ),
                    ),
                ),
            )
        )
        if weight_bytes is None:  # batch-independent; sum once from the live model
            weight_bytes = stage_weight_bytes(engine.entry.model)

        request = make_request(bs, cfg, args.num_images, device, dtype, args.lang_len)

        lat = time_e2e(engine, request, args.n_warmup, args.n_timed)
        mem_alloc = torch.cuda.max_memory_allocated() / 2**20
        mem_resv = torch.cuda.max_memory_reserved() / 2**20
        tp = bs * 1000.0 / lat["mean"]

        stages, trace_file = profile_stages(
            engine, request, args.trace_dir / f"bs{bs}", args.n_prof_steps
        )

        # Achieved TFLOPS + AI per stage. Vision/LLM run once per sample; the
        # expert loop runs num_inference_steps. achieved = B*flop / stage_seconds.
        def achieved_tflops(stage_flop, stage_ms):
            return (bs * stage_flop / 1e12) / (stage_ms / 1e3) if stage_ms > 0 else 0.0

        ach = {
            "vision": achieved_tflops(flop["vision"], stages["pi05.vision_loop"]),
            "llm_prefix": achieved_tflops(
                flop["llm_prefix"], stages["pi05.llm_prefix_fwd"]
            ),
            "expert": achieved_tflops(flop["expert_loop"], stages["pi05.expert_loop"]),
        }
        ai = {
            "vision": bs * flop["vision"] / weight_bytes["vision"],
            "llm_prefix": bs * flop["llm_prefix"] / weight_bytes["llm_prefix"],
            "expert": bs * flop["expert_loop"] / weight_bytes["expert_loop"],
        }
        expert_mfu = (100.0 * ach["expert"] / peak_tflops) if peak_tflops else None

        sweep.append(
            {
                "bs": bs,
                "latency_ms": lat,
                "per_sample_ms": lat["mean"] / bs,
                "throughput_sps": tp,
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
            f"tp={tp:.1f}/s per_sample={lat['mean'] / bs:.3f}ms "
            f"mem={mem_alloc:.0f}MiB"
        )
        print(
            f"[bs={bs}] stages(ms): "
            + "  ".join(f"{s.split('.')[1]}={stages[s]:.2f}" for s in STAGES)
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
            "model": "pi05",
            "scheduler": "ws1",
            "checkpoint": args.checkpoint.name,  # basename only — no host path
            "dtype": "bfloat16",
            "use_cuda_graph": True,
            "num_images": args.num_images,
            "lang_len": args.lang_len,
            "lang_bucket": mf.bucket_lang_len(args.lang_len, dims),
            "n_per_sample": mf.n_per_sample(args.lang_len, dims, args.num_images),
            "n_warmup": args.n_warmup,
            "n_timed": args.n_timed,
            "n_prof_steps": args.n_prof_steps,
            "flashinfer_workspace_mib": args.workspace_bytes // 2**20,
            "flashinfer_bf16_backend": args.bf16_gemm_backend,
            "flashinfer_autotune": args.flashinfer_autotune,
            "flashinfer_autotune_cache": (
                flashinfer_tune_cache.name
                if flashinfer_tune_cache is not None
                else None
            ),
        },
        "hardware": hardware,
        "stages_flop": {
            "lang_len": args.lang_len,
            "num_images": args.num_images,
            "flop_per_sample": flop,
            "weight_bytes": weight_bytes,
        },
        "sweep": sweep,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    # Summary table.
    print(
        f"\n{'bs':>4}{'e2e_ms':>10}{'p99':>8}{'tp/s':>9}{'per_smpl':>10}"
        f"{'mem_MiB':>10}{'vision':>9}{'llm_fwd':>9}{'expert':>9}{'exp_MFU':>9}"
    )
    for r in sweep:
        st = r["stage_gpu_ms"]
        mfu = f"{r['expert_mfu_pct']:.1f}%" if r["expert_mfu_pct"] else "-"
        print(
            f"{r['bs']:>4}{r['latency_ms']['mean']:>10.2f}{r['latency_ms']['p99']:>8.2f}"
            f"{r['throughput_sps']:>9.1f}{r['per_sample_ms']:>10.3f}"
            f"{r['mem_alloc_mib']:>10.0f}{st['pi05.vision_loop']:>9.2f}"
            f"{st['pi05.llm_prefix_fwd']:>9.2f}{st['pi05.expert_loop']:>9.2f}{mfu:>9}"
        )


if __name__ == "__main__":
    main()
