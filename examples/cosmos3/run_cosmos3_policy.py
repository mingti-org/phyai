"""End-to-end Cosmos3 action/policy demo.

Drives the ``cosmos3_policy`` engine plugin on a Cosmos3-Nano checkpoint:
preprocesses an observation (single ``--image`` or a multi-frame ``--video``) +
task prompt, runs the diffusion policy, and writes the predicted action (JSON) plus
an optional rollout video (mp4).

Requires CUDA and a Cosmos3-Nano checkpoint.

Example (policy mode, the default)::

    uv run python examples/cosmos3/run_cosmos3_policy.py \\
        --checkpoint /path/to/Cosmos3-Nano \\
        --image observation.png \\
        --prompt "robot picks up the cup" \\
        --domain-name agibotworld \\
        --out .cache/cosmos3_policy_out

Video observation with the linear-flow sampler::

    uv run python examples/cosmos3/run_cosmos3_policy.py \\
        --checkpoint /path/to/Cosmos3-Nano \\
        --video obs.mp4 --domain-name bridge_orig_lerobot --image-size 480 \\
        --mode forward_dynamics --action-file action.json \\
        --condition-frames 0,1 --prompt-format json \\
        --use-karras-sigmas false --flow-shift 10 --fps 5

Modes (``--mode``):
  * policy            (default) observation + prompt -> action chunk [+ video]
  * inverse_dynamics  observation video + prompt -> action explaining the transition
  * forward_dynamics  observation + prompt + action -> rollout video; pass
                      ``--action-file <chunks.json>``

Conventions:
  * ``--video`` reads ``action_chunk_size + 1`` frames -> correct t_lat; ``--image``
    is single-frame (t_lat=1). Clean frames default to ``0,1`` (video) / ``0`` (image).
  * ``--prompt-format json`` = structured caption; ``plain`` =
    duration/FPS/resolution sentences.
  * ``--use-karras-sigmas auto`` reads the checkpoint scheduler_config; ``false`` =
    linear-flow + ``--flow-shift``.

Outputs (prefix set by ``--out``, default ``.cache/cosmos3_policy_out``):
  * ``<out>_action.json``  predicted action (always written)
  * ``<out>.mp4``          decoded rollout video (when the plugin returns pixels)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _read_video_frames(path: str, num_frames: int) -> list:
    """Read the first ``num_frames`` frames of a video as a list of HxWx3 uint8.

    Decodes in-process via PyAV (libav). Repeats the last frame if the clip is
    shorter (pad to ``num_frames``).
    """
    import av

    frames: list = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))  # [H, W, 3] uint8
            if len(frames) >= num_frames:
                break
    if not frames:
        raise SystemExit(
            f"could not decode any frames from {path!r}; pass --image instead."
        )
    while len(frames) < num_frames:  # repeat last frame to num_frames
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

    arr = frames.numpy()  # [T, H, W, 3] uint8 RGB
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
        "--checkpoint", required=True, help="Cosmos3-Nano checkpoint dir"
    )
    parser.add_argument(
        "--image", default=None, help="Single observation image (-> t_lat=1)"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Observation video (mp4): reads the first action_chunk_size+1 frames "
        "as a multi-frame observation (-> correct t_lat). Decoded via PyAV.",
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
        help="Comma-separated clean latent-frame indices (e.g. '0,1'). Default: "
        "'0,1' for --video, '0' for --image.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("plain", "json"),
        default="json",
        help="'json' = structured caption; 'plain' = duration/FPS/resolution "
        "sentences.",
    )
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
        "--use-karras-sigmas",
        choices=("auto", "true", "false"),
        default="false",
        help="UniPC sigma schedule. 'auto' reads use_karras_sigmas from the "
        "checkpoint scheduler_config.json. 'false' (default) matches native "
        "linear-flow + flow_shift.",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument(
        "--raw-action-dim",
        type=int,
        default=None,
        help="Embodiment raw action width. Auto-resolved from --domain-name when "
        "omitted (e.g. droid_lerobot=10, agibotworld=29).",
    )
    parser.add_argument(
        "--action-stats-path",
        default=None,
        help="JSON stats file to denormalize the output action to physical units.",
    )
    parser.add_argument(
        "--action-normalization",
        choices=("minmax", "meanstd", "quantile", "quantile_rot"),
        default="minmax",
        help="Denormalization method for --action-stats-path.",
    )
    parser.add_argument("--no-prompt-metadata", action="store_true")
    parser.add_argument("--out", default=".cache/cosmos3_policy_out")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    if (args.image is None) == (args.video is None):
        raise SystemExit("pass exactly one of --image or --video.")

    # Observation frames + default clean-frame conditioning per input type
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
    from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
    from phyai.models.cosmos3 import Cosmos3ActionRequest, pixel_to_latent_shape
    from phyai.models.cosmos3.main_cosmos3_policy import Cosmos3PolicyArgs
    from phyai_utils_tools.models.cosmos3 import Cosmos3PolicyProcessor

    device = "cuda"
    dtype = torch.bfloat16

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[engine] creating cosmos3_policy engine ...")
    use_karras = {"auto": None, "true": True, "false": False}[args.use_karras_sigmas]
    engine = Engine(
        EngineArgs(
            plugin="cosmos3_policy",
            plugin_args=Cosmos3PolicyArgs(
                checkpoint_dir=args.checkpoint,
                flow_shift=args.flow_shift,
                use_karras_sigmas=use_karras,
                policy_modeling_mode=args.policy_modeling_mode,
                decode_video=True,
            ),
            config=EngineConfig(
                device=DeviceConfig(target=device, params_dtype=dtype),
                runtime=RuntimeConfig(use_cuda_graph=False),
            ),
        )
    )

    try:
        print("[processor] preprocessing ...")
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

        raw_input: dict = {
            "images": observation,
            "task": args.prompt,
        }
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

        print(
            f"[run] mode={args.mode} domain={args.domain_name} "
            f"latent={video_shape} clean_frames={list(cond_frames)} "
            f"steps={args.steps} action_chunk={args.action_chunk_size}x{args.raw_action_dim}"
        )
        result = engine.step(request)

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
