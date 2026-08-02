"""End-to-end Cosmos3 tensor-parallel generation demo — T2V / T2AV under torchrun.

Multi-GPU ("wn" / world-size-N) sibling of ``run_cosmos3.py``. Drives the
``cosmos3_wn`` engine plugin: the transformer is sharded across ``--tp`` GPUs
(tensor parallelism); with ``--cfg 2`` the conditional and unconditional CFG
branches additionally run concurrently on two ``tp`` groups (CFG parallelism), so
each denoise step does one forward per group instead of two sequential. The WAN VAE
decode is split spatially across all ``cfg*tp`` ranks. Only rank 0 writes the mp4.

Launch under torchrun with ``--nproc_per_node`` equal to ``cfg * tp``::

    # TP only (4 GPUs)
    torchrun --nproc_per_node=4 examples/cosmos3/run_cosmos3_wn.py --tp 4 \\
        --checkpoint /path/to/Cosmos3-Nano \\
        --prompt "A red sports car driving along a coastal road at sunset." \\
        --out .cache/cosmos3_t2v_wn

    # CFG=2 x TP=4 (8 GPUs) — cond/uncond branches run concurrently
    torchrun --nproc_per_node=8 examples/cosmos3/run_cosmos3_wn.py --cfg 2 --tp 4 \\
        --checkpoint /path/to/Cosmos3-Nano --prompt "..." --out .cache/cosmos3_t2v_wn

Cosmos3-Nano (32 attention heads / 8 KV heads) supports ``--tp`` in {1, 2, 4, 8};
``--cfg`` is 1 or 2 (cosmos3 has exactly two CFG branches) and only helps when
``--guidance-scale > 1``. Add ``--sound`` for a jointly denoised audio stream (T2AV).
Defaults follow the cosmos-framework native generation config, same as ``run_cosmos3.py``.

Requires CUDA + NCCL.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import time

import torch


def _resolve_topology(cfg: int, tp: int) -> tuple[int, int, int, bool]:
    """Reconcile ``--cfg``/``--tp`` with the torchrun launch env.

    Returns ``(local_rank, cfg, tp, is_main)``. The total process count is
    ``world = cfg * tp`` (one process per rank); it must equal torchrun's
    ``WORLD_SIZE``. ``cfg=tp=1`` runs in-process (no torchrun needed). Rank 0 is in
    the cond CFG group and holds the final combined media.
    """
    if cfg not in (1, 2):
        raise SystemExit("--cfg must be 1 or 2 (cosmos3 has exactly 2 CFG branches).")
    world = cfg * tp
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and env_world != world:
        raise SystemExit(
            f"--cfg {cfg} x --tp {tp} = {world} requires torchrun "
            f"--nproc_per_node={world} (saw WORLD_SIZE={env_world}). Example:\n"
            f"  torchrun --nproc_per_node={world} examples/cosmos3/run_cosmos3_wn.py "
            f"--cfg {cfg} --tp {tp} --checkpoint <ckpt> ..."
        )
    if world == 1 and env_world != 1:
        raise SystemExit(
            f"launched under torchrun (WORLD_SIZE={env_world}) but --cfg*--tp is 1; "
            f"set --cfg/--tp to use all ranks."
        )
    return local_rank, cfg, tp, local_rank == 0


@contextlib.contextmanager
def _timed(label: str, store: dict):
    """Time a region in seconds (CUDA-synchronized) into ``store[label]``."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        store[label] = time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Cosmos3-Nano checkpoint dir"
    )
    parser.add_argument(
        "--prompt", default="A red sports car driving along a coastal road at sunset."
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt. Omit to use the native structured default; pass '' for none.",
    )
    parser.add_argument("--num-frames", type=int, default=189)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=10.0)
    parser.add_argument(
        "--use-karras-sigmas",
        choices=("auto", "true", "false"),
        default="false",
        help="UniPC sigma schedule. 'false' (default) = native linear-flow + flow_shift; "
        "'true' = Karras (diffusers); 'auto' reads the checkpoint scheduler_config.json.",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sound",
        action="store_true",
        help="Also generate a joint audio stream (T2AV).",
    )
    parser.add_argument("--out", default=".cache/cosmos3_t2v_wn")
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor-parallel degree; world_size = cfg * tp must equal "
        "torchrun --nproc_per_node.",
    )
    parser.add_argument(
        "--cfg",
        type=int,
        default=1,
        help="CFG-parallel degree (1 or 2). cfg=2 runs the cond/uncond branches "
        "concurrently on two tp groups (needs 2*tp GPUs); only helps when "
        "--guidance-scale > 1.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    from phyai.engine import Engine, EngineArgs
    from phyai.engine_config import (
        DeviceConfig,
        EngineConfig,
        ParallelConfig,
        RuntimeConfig,
    )
    from phyai.models.cosmos3 import Cosmos3T2VRequest, pixel_to_latent_shape
    from phyai.models.cosmos3.main_cosmos3_wn import Cosmos3WNArgs
    from phyai_utils_tools.models.cosmos3 import (
        Cosmos3GenerationPostProcessor,
        Cosmos3Processor,
    )

    local_rank, cfg_size, tp_size, is_main = _resolve_topology(args.cfg, args.tp)
    device = f"cuda:{local_rank}"
    dtype = torch.bfloat16
    use_karras = {"auto": None, "true": True, "false": False}[args.use_karras_sigmas]

    def log(*a, **k):
        if is_main:
            print(*a, **k)

    timings: dict[str, float] = {}

    log(
        f"[engine] creating cosmos3 tensor-parallel generation engine "
        f"(cfg={cfg_size}, tp={tp_size})..."
    )
    engine = Engine(
        EngineArgs(
            plugin="cosmos3_wn",
            plugin_args=Cosmos3WNArgs(
                checkpoint_dir=args.checkpoint,
                flow_shift=args.flow_shift,
                use_karras_sigmas=use_karras,
                load_sound=(True if args.sound else None),
            ),
            config=EngineConfig(
                device=DeviceConfig(target=device, params_dtype=dtype),
                parallel=ParallelConfig(
                    world_size=cfg_size * tp_size,
                    cfg_size=cfg_size,
                    tp_size=tp_size,
                ),
                runtime=RuntimeConfig(use_cuda_graph=False),
            ),
        )
    )

    try:
        with _timed("preprocess", timings):
            processor = Cosmos3Processor(
                f"{args.checkpoint}/text_tokenizer",
                fps=args.fps,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                append_metadata=True,
            )
            cond, uncond = processor.tokenize_pair(
                args.prompt, negative_prompt=args.negative_prompt, device=device
            )
            video_shape = pixel_to_latent_shape(
                args.num_frames, args.height, args.width
            )
            sound_frames = (
                math.ceil(args.num_frames / args.fps * 25.0) if args.sound else None
            )
            request = Cosmos3T2VRequest(
                text_ids=cond.text_ids,
                text_mask=cond.text_mask,
                neg_text_ids=uncond.text_ids,
                neg_text_mask=uncond.text_mask,
                video_shape=video_shape,
                fps=args.fps,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                sound_frames=sound_frames,
            )

        log(
            f"[run] T2{'AV' if args.sound else 'V'} latent={video_shape} "
            f"steps={args.steps} guidance={args.guidance_scale} shift={args.flow_shift} "
            f"cfg={cfg_size} tp={tp_size}"
        )
        # Every rank runs the denoise loop (the collectives must fire on all ranks);
        # the result is identical across ranks, so only rank 0 writes it out.
        with _timed("inference", timings):
            result = engine.step(request)

        if is_main:
            postprocessor = Cosmos3GenerationPostProcessor(fps=args.fps)
            out_mp4 = f"{args.out}.mp4"
            with _timed("to_cpu", timings):
                media = postprocessor.postprocess(result)
            with _timed("encode", timings):
                postprocessor.save_mp4(media, out_mp4)
            print(
                f"[saved] -> {out_mp4}"
                + (
                    f" (+{media.sample_rate} Hz audio)"
                    if media.waveform is not None and media.sample_rate is not None
                    else ""
                )
            )

        log("\n=== timing (seconds) ===")
        for label in ("preprocess", "inference", "to_cpu", "encode"):
            if label in timings:
                log(f"  {label:<11s}{timings[label]:9.2f}")
        if timings.get("inference") and args.steps > 0:
            log(
                f"  {'per-step':<11s}{timings['inference'] / args.steps:9.3f}"
                f"   ({args.steps} steps)"
            )
        log(f"  {'TOTAL':<11s}{sum(timings.values()):9.2f}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
