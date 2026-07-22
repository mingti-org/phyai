"""End-to-end Cosmos3 tensor-parallel action/policy demo — under torchrun.

Multi-GPU ("wn" / world-size-N) sibling of ``run_cosmos3_policy.py``. Drives the
``cosmos3_policy_wn`` engine plugin: the transformer is sharded across ``--tp``
GPUs; with ``--cfg 2`` the cond/uncond branches additionally run concurrently on two
``tp`` groups (CFG parallelism, only useful at ``--guidance-scale > 1``). The
optional rollout-video decode is split spatially across all ``cfg*tp`` ranks. Only
rank 0 writes the action JSON / mp4.

Launch under torchrun with ``--nproc_per_node`` equal to ``cfg * tp``::

    torchrun --nproc_per_node=4 examples/cosmos3/run_cosmos3_policy_wn.py --tp 4 \\
        --checkpoint /path/to/Cosmos3-Nano-Policy-DROID --image observation.png \\
        --prompt "robot picks up the cup" --domain-name droid_lerobot \\
        --out .cache/cosmos3_policy_wn

    # CFG=2 x TP=4 (8 GPUs) — needs --guidance-scale > 1 to be useful:
    # torchrun --nproc_per_node=8 ... --cfg 2 --tp 4 --guidance-scale 4

Cosmos3-Nano (32 attention heads / 8 KV heads) supports ``--tp`` in {1, 2, 4, 8};
``--cfg`` is 1 or 2. Modes / conventions match ``run_cosmos3_policy.py``.

Requires CUDA + NCCL.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def _resolve_topology(cfg: int, tp: int) -> tuple[int, int, int, bool]:
    """Reconcile ``--cfg``/``--tp`` with the torchrun launch env.

    Returns ``(local_rank, cfg, tp, is_main)``. Total processes ``= cfg * tp`` (one
    per rank) must equal torchrun's ``WORLD_SIZE``. ``cfg=tp=1`` runs in-process.
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
            f"  torchrun --nproc_per_node={world} "
            f"examples/cosmos3/run_cosmos3_policy_wn.py --cfg {cfg} --tp {tp} "
            f"--checkpoint <ckpt> ..."
        )
    if world == 1 and env_world != 1:
        raise SystemExit(
            f"launched under torchrun (WORLD_SIZE={env_world}) but --cfg*--tp is 1; "
            f"set --cfg/--tp to use all ranks."
        )
    return local_rank, cfg, tp, local_rank == 0


def _read_video_frames(path: str, num_frames: int) -> list:
    """Read the first ``num_frames`` frames of a video as a list of HxWx3 uint8."""
    import av

    frames: list = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= num_frames:
                break
    if not frames:
        raise SystemExit(
            f"could not decode any frames from {path!r}; pass --image instead."
        )
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


def _save_video(video: torch.Tensor, path: str, fps: float) -> None:
    """Save ``[1, 3, T, H, W]`` or ``[3, T, H, W]`` in [0,1] to mp4 (PyAV) or ``.pt``."""
    if video.ndim == 5:
        video = video[0]
    frames = (video.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 3, 0).cpu()
    if path.endswith(".pt"):
        torch.save(frames, path)
        return
    from fractions import Fraction

    import av

    arr = frames.numpy()
    with av.open(path, mode="w") as container:
        stream = container.add_stream(
            "h264", rate=Fraction(fps).limit_denominator(10000)
        )
        stream.width = int(arr.shape[2])
        stream.height = int(arr.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for frame_data in arr:
            for pkt in stream.encode(av.VideoFrame.from_ndarray(frame_data, "rgb24")):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


def _save_action(action: torch.Tensor, path: str) -> None:
    """Save ``[1, chunk, dim]`` action tensor as JSON."""
    data = {
        "shape": list(action.shape),
        "dtype": str(action.dtype).replace("torch.", ""),
        "data": action.squeeze(0).tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_action_from_file(path: str, chunk_index: int = 0) -> torch.Tensor:
    """Load action from the Cosmos3 example JSON format."""
    with open(path) as f:
        data = json.load(f)
    if "action_chunks" in data:
        chunk = data["action_chunks"][chunk_index]
        return torch.tensor(chunk, dtype=torch.float32)
    if "data" in data:
        shape = data.get("shape", None)
        t = torch.tensor(data["data"], dtype=torch.float32)
        if shape:
            t = t.reshape(shape)
        return t
    raise ValueError(f"Unrecognized action file format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Cosmos3-Nano(-Policy) checkpoint dir"
    )
    parser.add_argument(
        "--image", default=None, help="Single observation image (-> t_lat=1)"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Observation video (mp4): reads the first action_chunk_size+1 frames.",
    )
    parser.add_argument("--prompt", default="robot manipulates objects")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--mode",
        choices=("policy", "forward_dynamics", "inverse_dynamics"),
        default="policy",
    )
    parser.add_argument(
        "--condition-frames",
        default=None,
        help="Comma-separated clean latent-frame indices (e.g. '0,1').",
    )
    parser.add_argument("--prompt-format", choices=("plain", "json"), default="json")
    parser.add_argument("--view-point", default="ego_view")
    parser.add_argument("--domain-name", default="agibotworld")
    parser.add_argument(
        "--action-file",
        default=None,
        help="JSON file with action chunks (required for forward_dynamics)",
    )
    parser.add_argument("--action-chunk-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument(
        "--image-size",
        type=int,
        default=480,
        help="Snap the observation to the closest aspect ratio in this tier. "
        "Pass 0 to use the explicit --height/--width instead.",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument(
        "--policy-modeling-mode",
        choices=("reference", "fused"),
        default="reference",
        help="Reference matches official policy arithmetic; fused favors speed.",
    )
    parser.add_argument(
        "--use-karras-sigmas", choices=("auto", "true", "false"), default="false"
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--raw-action-dim", type=int, default=None)
    parser.add_argument("--action-stats-path", default=None)
    parser.add_argument(
        "--action-normalization",
        choices=("minmax", "meanstd", "quantile", "quantile_rot"),
        default="minmax",
    )
    parser.add_argument("--no-prompt-metadata", action="store_true")
    parser.add_argument("--out", default=".cache/cosmos3_policy_wn")
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
        help="CFG-parallel degree (1 or 2). cfg=2 runs cond/uncond on two tp groups "
        "(needs 2*tp GPUs); only helps when --guidance-scale > 1 (policy default 1).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    if (args.image is None) == (args.video is None):
        raise SystemExit("pass exactly one of --image or --video.")

    if args.video is not None:
        observation = _read_video_frames(args.video, args.action_chunk_size + 1)
        default_cond = (0, 1)
    else:
        observation = args.image
        default_cond = (0,)
    if args.condition_frames is not None:
        cond_frames = tuple(int(x) for x in args.condition_frames.split(",") if x != "")
    else:
        cond_frames = default_cond

    from phyai.engine import Engine, EngineArgs
    from phyai.engine_config import (
        DeviceConfig,
        EngineConfig,
        ParallelConfig,
        RuntimeConfig,
    )
    from phyai.models.cosmos3 import Cosmos3ActionRequest, pixel_to_latent_shape
    from phyai.models.cosmos3.main_cosmos3_policy_wn import Cosmos3PolicyWNArgs
    from phyai_utils_tools.models.cosmos3 import Cosmos3PolicyProcessor

    local_rank, cfg_size, tp_size, is_main = _resolve_topology(args.cfg, args.tp)
    device = f"cuda:{local_rank}"
    dtype = torch.bfloat16

    def log(*a, **k):
        if is_main:
            print(*a, **k)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"[engine] creating cosmos3_policy tensor-parallel engine "
        f"(cfg={cfg_size}, tp={tp_size}) ..."
    )
    use_karras = {"auto": None, "true": True, "false": False}[args.use_karras_sigmas]
    engine = Engine(
        EngineArgs(
            plugin="cosmos3_policy_wn",
            plugin_args=Cosmos3PolicyWNArgs(
                checkpoint_dir=args.checkpoint,
                flow_shift=args.flow_shift,
                use_karras_sigmas=use_karras,
                policy_modeling_mode=args.policy_modeling_mode,
                decode_video=True,
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
        log("[processor] preprocessing ...")
        processor = Cosmos3PolicyProcessor(
            tokenizer_name_or_path=f"{args.checkpoint}/text_tokenizer",
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            mode=args.mode,
            domain_name=args.domain_name,
            action_chunk_size=args.action_chunk_size,
            raw_action_dim=args.raw_action_dim,
            fps=args.fps,
            image_size=(args.image_size if args.image_size > 0 else None),
            append_metadata=not args.no_prompt_metadata,
            prompt_format=args.prompt_format,
            view_point=args.view_point,
            cond_frame_indexes=cond_frames,
            action_stats_path=args.action_stats_path,
            action_normalization=args.action_normalization,
            negative_prompt=args.negative_prompt,
            device=device,
            params_dtype=dtype,
        )

        raw_input: dict = {"images": observation, "task": args.prompt}
        if args.mode == "forward_dynamics":
            if args.action_file is None:
                raise ValueError("--action-file is required for forward_dynamics mode.")
            raw_input["cond_action"] = _load_action_from_file(
                args.action_file, args.action_chunk_index
            )

        processed = processor.preprocess(raw_input)
        video_shape = pixel_to_latent_shape(
            processed.video_num_frames,
            processed.content_size[0],
            processed.content_size[1],
        )
        request = Cosmos3ActionRequest(
            text_ids=processed.text_ids.to(device),
            text_mask=processed.text_mask.to(device),
            neg_text_ids=processed.neg_text_ids.to(device),
            neg_text_mask=processed.neg_text_mask.to(device),
            video_shape=video_shape,
            mode=processed.mode,
            domain_id=processed.domain_id,
            action_chunk=processed.action_chunk,
            raw_action_dim=processed.raw_action_dim,
            cond_video_pixels=processed.pixel_values.to(device=device, dtype=dtype),
            cond_action=(
                processed.cond_action.to(device=device, dtype=dtype)
                if processed.cond_action is not None
                else None
            ),
            cond_frame_indexes=processed.cond_frame_indexes,
            cond_action_indexes=processed.cond_action_indexes,
            action_start_frame_offset=processed.action_start_frame_offset,
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )

        log(
            f"[run] mode={args.mode} domain={args.domain_name} "
            f"latent={video_shape} clean_frames={list(cond_frames)} "
            f"steps={args.steps} action_chunk={args.action_chunk_size}x"
            f"{args.raw_action_dim} cfg={cfg_size} tp={tp_size}"
        )
        # All ranks run the denoise loop (collectives must fire everywhere); the
        # action / video are identical across ranks, so only rank 0 writes output.
        result = engine.step(request)

        if is_main:
            output = processor.postprocess(result)
            action = output["action"]
            print(
                f"[done] action shape={tuple(action.shape)} "
                f"range=[{action.min():.4f}, {action.max():.4f}]"
            )
            action_path = f"{args.out}_action.json"
            _save_action(action, action_path)
            print(f"[saved] action -> {action_path}")
            if "pixels" in output:
                video_path = f"{args.out}.mp4"
                _save_video(output["pixels"], video_path, args.fps)
                print(f"[saved] video -> {video_path}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
