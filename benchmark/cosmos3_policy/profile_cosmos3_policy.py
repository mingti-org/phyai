from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

import model_flops_cosmos3 as mf
import hardware_probe as hp


# The released DROID policy's recommended denoise-step count (cosmos-framework
# ``sample_args.json``: num_steps=4). The checkpoint is a distilled few-step
# policy, so this is fixed — the swept axis is batch size, not steps.
NUM_STEPS = 4


def register_single_gpu_mesh() -> None:
    """Single-GPU ws=1 mesh + flashinfer linear init (mirrors the parity run).

    The Engine plugin does this internally; a direct-construction profiler must
    replicate it before building any phyai layer.
    """
    from unittest.mock import MagicMock

    import phyai.layers.linear as linear_mod
    from phyai.parallel.mesh import Mesh
    from phyai.parallel.state import register_mesh

    tm = MagicMock()
    tm.mesh_dim_names = ("tp",)
    tm.size.side_effect = lambda axis: 1
    tm.get_local_rank.side_effect = lambda axis: 0
    tm.get_group.side_effect = lambda axis: MagicMock()
    register_mesh(Mesh(tm, name="model"))
    linear_mod.init(register_flashinfer=True, validate=False)


def module_bytes(module: torch.nn.Module) -> int:
    """Sum ``numel * element_size`` over a module's parameters."""
    return sum(p.numel() * p.element_size() for p in module.parameters())


def stage_weight_bytes(transformer, dims: mf.Cosmos3Dims, dtype_bytes: int) -> dict:
    """Exact resident weight bytes streamed by each phase, from the loaded model.

    * ``cond_encode`` — the UND tower's decoder layers. ``embed_tokens`` is a
      gather and ``language_model.norm`` is never called in the condition forward,
      so both are excluded.
    * ``gen_1step`` — the GEN decoder stack + ``proj_in`` / ``proj_out`` /
      ``norm_moe_gen`` / ``time_embedder``, read once per denoise forward (shared
      across the whole batch). The domain-aware action adapters stream only the
      active embodiment row, counted as a single
      ``(hidden·action_dim + action_dim·hidden)`` slice.
    """
    und_b = module_bytes(transformer.language_model.layers)
    gen_b = module_bytes(transformer.gen_layers)
    gen_b += module_bytes(transformer.proj_in)
    gen_b += module_bytes(transformer.proj_out)
    gen_b += module_bytes(transformer.norm_moe_gen)
    gen_b += module_bytes(transformer.time_embedder)
    gen_b += (
        dims.hidden * dims.action_dim + dims.action_dim * dims.hidden
    ) * dtype_bytes
    return {"cond_encode": float(und_b), "gen_1step": float(gen_b)}


def _stats(times_ms: list[float]) -> dict:
    """Latency stats over a list of per-iteration ms (cuda-event timed)."""
    t = sorted(times_ms)
    return {
        "mean": statistics.fmean(t),
        "p50": t[len(t) // 2],
        "p90": t[int(len(t) * 0.9)],
        "p99": t[min(len(t) - 1, int(len(t) * 0.99))],
        "std": statistics.pstdev(t),
        "min": t[0],
        "max": t[-1],
    }


def time_call(fn, n_warmup: int, n_timed: int) -> dict:
    """Cuda-event latency stats for ``fn`` over ``n_timed`` runs after warmup."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_timed):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return _stats(times)


def dims_from_config(cfg) -> mf.Cosmos3Dims:
    """Build the FLOP-model dims from a live :class:`Cosmos3Config`."""
    return mf.Cosmos3Dims(
        hidden=cfg.hidden_size,
        layers=cfg.num_hidden_layers,
        heads=cfg.num_attention_heads,
        kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        intermediate=cfg.intermediate_size,
        latent_channel=cfg.latent_channel,
        latent_patch_size=cfg.latent_patch_size,
        patch_latent_dim=cfg.patch_latent_dim,
        action_dim=cfg.action_dim,
    )


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
        required=True,
        help="Cosmos3-Nano-Policy checkpoint (transformer/ + vae/ subdirs).",
    )
    ap.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Batch sizes to sweep (observations / robots per forward).",
    )
    ap.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="Observation video frames. 0 = auto = action_chunk + 1 (cosmos-framework "
        "hard-couples them: inference/action.py `target_frames = action_chunk_size + 1`, "
        "and the RoboLab policy server `t_frames = action_chunk_size + 1`). The video is "
        "chunk+1 frames with only frame 0 the real observation; the rest are denoised.",
    )
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument(
        "--action-chunk",
        type=int,
        default=32,
        help="Action horizon (tech report §4.2.5: 32 future actions @ 15Hz).",
    )
    ap.add_argument(
        "--raw-action-dim", type=int, default=10, help="droid_lerobot = 10."
    )
    ap.add_argument("--domain-id", type=int, default=8, help="droid_lerobot = 8.")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument(
        "--guidance",
        type=float,
        default=3.0,
        help="CFG scale; >1 runs a cond + uncond branch per step (2x the GEN cost).",
    )
    ap.add_argument("--s-text", type=int, default=96, help="Prompt token count.")
    ap.add_argument("--n-warmup", type=int, default=2)
    ap.add_argument("--n-timed", type=int, default=5)
    ap.add_argument(
        "--no-vae",
        action="store_true",
        help="Skip loading/timing the VAE (profile the transformer phases only).",
    )
    ap.add_argument(
        "--decode-video",
        action="store_true",
        help="Also time VAE decode (NOT on the action path; reported separately).",
    )
    ap.add_argument("--flow-shift", type=float, default=5.0, help="DROID default 5.0.")
    ap.add_argument("--no-roofline", action="store_true", help="Skip the microbench.")
    ap.add_argument(
        "--gpu-index", type=int, default=0, help="Local device index to probe."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark/cosmos3_policy/cosmos3_policy_profile.json"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this benchmark.")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    dtype_bytes = 2

    # --- Hardware facts + measured roofline (once, before the model fills VRAM). ---
    print("[hw] probing device + measuring roofline microbench...")
    hardware = (
        hp.probe_device(args.gpu_index)
        if args.no_roofline
        else hp.measure_roofline(device=args.gpu_index)
    )
    peak_tflops = hardware.get("peak_bf16_tflops")
    print(
        f"[hw] {hardware['name']}: "
        + (
            f"BF16 peak {peak_tflops} TFLOPS, HBM {hardware.get('hbm_tb_s')} TB/s, "
            f"ridge {hardware.get('ridge_point_flop_per_byte')} FLOP/byte"
            if not args.no_roofline
            else "(roofline microbench skipped)"
        )
    )

    # --- Engine config + single-GPU mesh, THEN build (order matters). ---
    from phyai.engine_config import DeviceConfig, get_engine_config, set_engine_config

    set_engine_config(
        get_engine_config().replace(
            device=DeviceConfig(target="cuda", params_dtype=dtype)
        )
    )
    register_single_gpu_mesh()

    from phyai.models.cosmos3 import (
        Cosmos3ActionRequest,
        Cosmos3Config,
        Cosmos3PolicyScheduler,
        Cosmos3Transformer,
        Cosmos3WanVAE,
        Cosmos3WanVAEConfig,
        cosmos3_vae_weight_remap,
        cosmos3_weight_remap,
        pixel_to_latent_shape,
    )
    from phyai.utils import load_config
    from phyai.weights import load_pretrained

    ckpt = args.checkpoint
    print(f"[load] transformer from {ckpt / 'transformer'} ...")
    cfg = load_config(ckpt / "transformer", Cosmos3Config)
    transformer = Cosmos3Transformer(cfg, params_dtype=dtype, device=device).eval()
    load_pretrained(
        transformer, str(ckpt / "transformer"), remap=cosmos3_weight_remap, strict=False
    )

    vae = None
    if not args.no_vae:
        print(f"[load] vae from {ckpt / 'vae'} ...")
        vae_cfg = load_config(ckpt / "vae", Cosmos3WanVAEConfig)
        vae = Cosmos3WanVAE(vae_cfg)
        load_pretrained(
            vae, str(ckpt / "vae"), remap=cosmos3_vae_weight_remap, strict=False
        )
        vae = vae.to(device, dtype).eval()

    scheduler = Cosmos3PolicyScheduler(
        transformer, vae=vae, device=device, flow_shift=args.flow_shift
    )
    scheduler.setup()

    dims = dims_from_config(cfg)
    # cosmos-framework hard-couples the observation/rollout video length to the
    # action horizon: video_length = action_chunk + 1. Honor an explicit override,
    # else derive it (0 = auto).
    num_frames = args.num_frames if args.num_frames > 0 else args.action_chunk + 1
    t_lat, h_lat, w_lat = pixel_to_latent_shape(num_frames, args.height, args.width)
    print(
        f"[shape] action_chunk={args.action_chunk} -> video {num_frames} frames "
        f"-> latent grid = ({t_lat}, {h_lat}, {w_lat})"
    )

    flop = mf.stage_flops(
        dims,
        s_text=args.s_text,
        t_lat=t_lat,
        h_lat=h_lat,
        w_lat=w_lat,
        action_chunk=args.action_chunk,
        num_inference_steps=NUM_STEPS,
        guidance_scale=args.guidance,
    )
    weight_bytes = stage_weight_bytes(transformer, dims, dtype_bytes)
    branches = int(flop["cfg_branches"])
    n_steps = NUM_STEPS

    def achieved_tflops(stage_flop, stage_ms):
        return (stage_flop / 1e12) / (stage_ms / 1e3) if stage_ms > 0 else 0.0

    # ----------------------------------------------------------------------- #
    # Sweep batch size at the fixed denoise-step count.                        #
    # ----------------------------------------------------------------------- #
    sweep = []
    vae_decode_ms = None
    for bs in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gen = torch.Generator(device=device).manual_seed(0)
        try:
            cond_latents = torch.randn(
                bs,
                dims.latent_channel,
                t_lat,
                h_lat,
                w_lat,
                generator=gen,
                device=device,
                dtype=dtype,
            )
            obs_pixels = (
                torch.rand(
                    bs,
                    3,
                    num_frames,
                    args.height,
                    args.width,
                    generator=gen,
                    device=device,
                    dtype=dtype,
                )
                * 2.0
                - 1.0
            )
            text_ids = torch.full(
                (bs, args.s_text), 100, dtype=torch.long, device=device
            )
            text_mask = torch.ones(bs, args.s_text, dtype=torch.long, device=device)

            req = Cosmos3ActionRequest(
                text_ids=text_ids,
                text_mask=text_mask,
                neg_text_ids=text_ids,
                neg_text_mask=text_mask,
                video_shape=(t_lat, h_lat, w_lat),
                mode="policy",
                domain_id=args.domain_id,
                action_chunk=args.action_chunk,
                raw_action_dim=args.raw_action_dim,
                action_dim=dims.action_dim,
                cond_video_latents=cond_latents,
                fps=args.fps,
                num_inference_steps=n_steps,
                guidance_scale=args.guidance,
                seed=0,
            )

            # vae_encode (observation cost; not inside step since latents pre-set).
            vae_encode_ms = None
            if vae is not None:
                vae_encode_ms = time_call(
                    lambda: scheduler.vae_runner.encode(obs_pixels),
                    args.n_warmup,
                    args.n_timed,
                )["mean"]

            # Authoritative e2e denoise: the real batch-capable scheduler step.
            denoise = time_call(
                lambda: scheduler.step(req), args.n_warmup, args.n_timed
            )

            # Primitives for the per-phase split + roofline (batched).
            cond_1branch_ms = time_call(
                lambda: transformer.encode_condition(
                    text_ids,
                    text_mask,
                    (t_lat, h_lat, w_lat),
                    args.fps,
                    action_len=args.action_chunk,
                ),
                args.n_warmup,
                args.n_timed,
            )["mean"]

            domain = torch.full((bs,), args.domain_id, device=device, dtype=torch.long)
            action_latents = torch.randn(
                bs,
                args.action_chunk,
                dims.action_dim,
                generator=gen,
                device=device,
                dtype=dtype,
            )
            video_mask = torch.ones(bs, t_lat, dtype=torch.bool, device=device)
            action_mask = torch.ones(
                bs, args.action_chunk, dtype=torch.bool, device=device
            )
            tval = torch.tensor([500.0], device=device, dtype=dtype)
            scheduler.runner.reset()
            gen_step_ms = time_call(
                lambda: scheduler.runner.forward(
                    "cond",
                    cond_latents,
                    tval,
                    text_ids=text_ids,
                    text_mask=text_mask,
                    video_shape=(t_lat, h_lat, w_lat),
                    fps=args.fps,
                    noisy_frame_mask=video_mask,
                    action_latents=action_latents,
                    action_domain_id=domain,
                    action_noisy_mask=action_mask,
                ),
                args.n_warmup,
                args.n_timed,
            )["mean"]

            if args.decode_video and vae is not None and vae_decode_ms is None:
                vae_decode_ms = time_call(
                    lambda: scheduler.vae_runner.decode(cond_latents),
                    args.n_warmup,
                    args.n_timed,
                )["mean"]
        except Exception as exc:  # noqa: BLE001 - want to catch OOM and stop cleanly
            if is_oom(exc):
                print(
                    f"[bs={bs}] OOM — stopping sweep here (largest fitting batch < {bs})."
                )
                torch.cuda.empty_cache()
                break
            raise

        mem_alloc = torch.cuda.max_memory_allocated() / 2**20
        mem_resv = torch.cuda.max_memory_reserved() / 2**20

        cond_total_ms = cond_1branch_ms * branches
        gen_loop_ms = max(denoise["mean"] - cond_total_ms, 0.0)
        e2e_ms = denoise["mean"] + (vae_encode_ms or 0.0)
        chunks_per_s = bs * 1000.0 / e2e_ms
        per_sample_ms = e2e_ms / bs

        ach = {
            "cond_encode": achieved_tflops(bs * flop["cond_encode"], cond_1branch_ms),
            "gen_step": achieved_tflops(bs * flop["gen_1step"], gen_step_ms),
        }
        ai = {
            "cond_encode": bs * flop["cond_encode"] / weight_bytes["cond_encode"],
            "gen_step": bs * flop["gen_1step"] / weight_bytes["gen_1step"],
        }
        gen_mfu = (100.0 * ach["gen_step"] / peak_tflops) if peak_tflops else None

        sweep.append(
            {
                "bs": bs,
                "vae_encode_ms": vae_encode_ms,
                "cond_encode_1branch_ms": cond_1branch_ms,
                "cond_encode_total_ms": cond_total_ms,
                "gen_step_ms": gen_step_ms,
                "denoise_ms": denoise,
                "gen_loop_ms": gen_loop_ms,
                "e2e_ms": e2e_ms,
                "per_sample_ms": per_sample_ms,
                "action_chunks_per_s": chunks_per_s,
                "action_steps_per_s": args.action_chunk * chunks_per_s,
                "mem_alloc_mib": mem_alloc,
                "mem_reserved_mib": mem_resv,
                "achieved_tflops": ach,
                "arithmetic_intensity": ai,
                "gen_mfu_pct": gen_mfu,
            }
        )
        print(
            f"[bs={bs:>2}] e2e={e2e_ms:.0f}ms per_sample={per_sample_ms:.0f}ms  "
            f"{chunks_per_s:.2f} chunks/s  gen_step={gen_step_ms:.1f}ms "
            f"({ach['gen_step']:.0f} TFLOPS"
            + (f", {gen_mfu:.1f}% MFU)" if gen_mfu else ")")
            + f"  mem={mem_alloc:.0f}MiB"
        )

    out = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(),
            "model": "cosmos3_policy",
            "mode": "policy",
            "sweep_axis": "batch_size",
            "checkpoint": ckpt.name,  # basename only — no host path
            "dtype": "bfloat16",
            "use_cuda_graph": False,
            "num_steps": n_steps,
            "domain_id": args.domain_id,
            "raw_action_dim": args.raw_action_dim,
            "action_dim": dims.action_dim,
            "action_chunk": args.action_chunk,
            "num_frames": num_frames,
            "height": args.height,
            "width": args.width,
            "latent_grid": [t_lat, h_lat, w_lat],
            "s_text": args.s_text,
            "fps": args.fps,
            "flow_shift": args.flow_shift,
            "guidance_scale": args.guidance,
            "cfg_branches": branches,
            "use_vae": vae is not None,
            "vae_decode_ms": vae_decode_ms,
            "n_warmup": args.n_warmup,
            "n_timed": args.n_timed,
        },
        "hardware": hardware,
        "phases_flop": {
            "s_text": flop["s_text"],
            "s_video": flop["s_video"],
            "s_action": flop["s_action"],
            "s_gen": flop["s_gen"],
            "gen_kv_len": flop["gen_kv_len"],
            "flop_per_sample": {
                "cond_encode": flop["cond_encode"],
                "gen_1step": flop["gen_1step"],
                "gen_loop": flop["gen_loop"],
            },
            "weight_bytes": weight_bytes,
        },
        "sweep": sweep,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    # Summary table.
    print(
        f"\n{'bs':>4}{'e2e_ms':>10}{'per_smpl':>10}{'chunks/s':>10}"
        f"{'gen_step':>10}{'gen_MFU':>9}{'mem_MiB':>10}"
    )
    for r in sweep:
        mfu = f"{r['gen_mfu_pct']:.1f}%" if r["gen_mfu_pct"] else "-"
        print(
            f"{r['bs']:>4}{r['e2e_ms']:>10.0f}{r['per_sample_ms']:>10.1f}"
            f"{r['action_chunks_per_s']:>10.2f}{r['gen_step_ms']:>10.1f}"
            f"{mfu:>9}{r['mem_alloc_mib']:>10.0f}"
        )


if __name__ == "__main__":
    main()
