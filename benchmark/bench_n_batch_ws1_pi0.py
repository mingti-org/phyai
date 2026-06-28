"""End-to-end pi0 ws1 (single-card) latency benchmark, swept over batch sizes.

Builds the pi0 engine (phyai's own implementation, *not* lerobot's) via the
standard plugin path once per batch size, feeds a dummy request (random
pixels + one-token "prompt" + random robot state, optionally a fixed noise
tensor), and hands it to the generic :class:`NBatchBenchRunner` from
:mod:`bench_n_batch` for timing + optional Nsight Systems / Perfetto
profile capture.

The numbers are end-to-end ``Engine.step`` latency -- they include the
vision tower replay, the LLM prefix forward, the state-token pass, and all
``num_inference_steps`` Euler steps of the action expert. The action chunks
themselves are garbage (inputs are random); the script exists for
performance measurement, not correctness.

Run::

    uv run python \\
        benchmark/bench_n_batch_ws1_pi0.py \\
        --batch-sizes 1 2 4 --n-warmup 5 --n-timed 30 \\
        --result-file ./pi0_ws1_results.jsonl

    # random weights, exercise engine wiring + timing (no checkpoint needed)
    uv run python benchmark/bench_n_batch_ws1_pi0.py \\
        --batch-sizes 1 --n-warmup 3 --n-timed 10

Profile a tight window with the torch profiler (Perfetto-loadable)::

    ... --profile-backend torch --profile-output-dir ./prof \\
        --profile-start-step 5 --profile-num-steps 3

Profile under Nsight Systems::

    nsys profile --capture-range=cudaProfilerApi \\
        --capture-range-end=stop -o ./prof/pi0_ws1 \\
        uv run python benchmark/bench_n_batch_ws1_pi0.py \\
            --batch-sizes 4 --profile-backend nsys \\
            --profile-start-step 5 --profile-num-steps 3

``torch`` and ``nsys`` are exclusive -- pick one per run. NVTX ranges
emitted by ``nsys`` mode are only captured by an enclosing
``nsys profile``; nothing happens if you select ``--profile-backend
nsys`` without that wrapper.

The pi0 scheduler is already instrumented with named event scopes
(``pi0.vision_loop`` / ``pi0.lang_pack`` / ``pi0.llm_prefix_plan``
/ ``pi0.llm_prefix_fwd`` / ``pi0.expert_plan`` / ``pi0.expert_loop``
with per-Euler-step ``pi0.expert_step``), and the bench runner wraps
each timed step in ``bench.step`` -- every profile backend sees these
as named ranges with no extra wiring.

Note: pi0 runs the vision tower in fp32 by default (OpenPI parity), so
``--vision-dtype`` defaults to ``float32``; the rest of the model follows
``--dtype`` (bf16 by default).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

import bench_n_batch as bnb
from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi0.configuration_pi0 import PI0Config
from phyai.models.pi0.main_pi0 import PI0Args
from phyai.models.pi0.scheduler_ws1_pi0 import PI0Request
from phyai.utils import load_config
from phyai.utils.profile import (
    add_profile_cli_args,
    install_profiler,
    profile_config_from_args,
)


_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def make_dummy_request(
    *,
    batch_size: int,
    plugin_cfg: PI0Config,
    device: torch.device,
    dtype: torch.dtype,
    fixed_noise: bool = False,
) -> PI0Request:
    """Random pixels + single-token prompt + random state PI0Request.

    ``pixel_values`` is ``(B, num_images, C, H, W)``; ``state`` is
    ``(B, max_state_dim)`` (pi0 folds the robot state into an expert-side
    suffix token). When ``fixed_noise`` is set, the flow-matching noise is
    drawn under a fixed seed and passed explicitly so two runs start from
    identical noise -- the basis for an OpenPI parity check.
    """
    pixel_values = torch.rand(
        batch_size,
        plugin_cfg.num_images,
        plugin_cfg.vision.num_channels,
        plugin_cfg.vision.image_size,
        plugin_cfg.vision.image_size,
        dtype=dtype,
        device=device,
    )
    input_ids = torch.zeros(
        batch_size, plugin_cfg.tokenizer_max_length, dtype=torch.int64, device=device
    )
    input_ids[:, 0] = 2  # any non-pad token id
    lang_lens = torch.ones(batch_size, dtype=torch.int64, device=device)
    state = torch.rand(
        batch_size, plugin_cfg.max_state_dim, dtype=dtype, device=device
    )

    noise = None
    if fixed_noise:
        gen = torch.Generator(device=device).manual_seed(0)
        noise = torch.randn(
            batch_size,
            plugin_cfg.chunk_size,
            plugin_cfg.max_action_dim,
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


def make_setup_fn(
    *,
    checkpoint: Path | None,
    dtype: torch.dtype,
    device_target: str,
    use_cuda_graph: bool,
    num_images: int | None,
    vision_params_dtype: torch.dtype | None,
    fixed_noise: bool,
    flashinfer_workspace_bytes: int,
):
    """Build the per-batch-size ``setup_fn`` closure for :class:`NBatchBenchRunner`.

    Reads the plugin config once (from ``checkpoint`` when given, else the
    pi0 defaults); the engine itself is rebuilt per batch size so each gets
    a fresh ``max_batch_size``-sized scheduler / KV pool.
    """
    if checkpoint is not None:
        plugin_cfg = load_config(checkpoint, PI0Config)
    else:
        plugin_cfg = PI0Config()
    if num_images is not None:
        plugin_cfg = replace(plugin_cfg, empty_cameras=3 - num_images)
    device = torch.device(device_target)

    def setup_fn(batch_size: int) -> bnb.BenchSpec:
        engine = Engine(
            EngineArgs(
                plugin="pi0",
                plugin_args=PI0Args(
                    checkpoint_dir=checkpoint,
                    config=plugin_cfg,
                    max_batch_size=batch_size,
                    vision_params_dtype=vision_params_dtype,
                ),
                config=EngineConfig(
                    device=DeviceConfig(target=device_target, params_dtype=dtype),
                    runtime=RuntimeConfig(
                        use_cuda_graph=use_cuda_graph,
                        flashinfer_workspace_bytes=flashinfer_workspace_bytes,
                    ),
                ),
            )
        )
        request = make_dummy_request(
            batch_size=batch_size,
            plugin_cfg=plugin_cfg,
            device=device,
            dtype=dtype,
            fixed_noise=fixed_noise,
        )
        return bnb.BenchSpec(
            name="ws1_pi0",
            step_callable=lambda: engine.step(request),
            teardown_callable=engine.close,
        )

    return setup_fn


def make_extras_fn(
    *,
    dtype_name: str,
    device_target: str,
    use_cuda_graph: bool,
    num_images: int,
):
    def extras_fn(batch_size: int, spec: bnb.BenchSpec) -> dict[str, Any]:
        return {
            "model": "pi0",
            "scheduler": "ws1",
            "dtype": dtype_name,
            "device": device_target,
            "use_cuda_graph": use_cuda_graph,
            "max_batch_size": batch_size,
            "num_images": num_images,
        }

    return extras_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional HF-style pi0 pytorch checkpoint folder (config.json + "
            "model.safetensors[.index.json]). Omit to run with random weights "
            "for a pure wiring/timing smoke test."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="bf16",
        help="Engine params_dtype (default bf16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Engine device target (default "cuda"; pass "cpu" for CPU-only debug).',
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA graph capture (engine still runs, just no replay).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        choices=(2, 3),
        default=None,
        help=(
            "Override the camera count. By default this is inferred from "
            "checkpoint config empty_cameras (3 - empty_cameras)."
        ),
    )
    parser.add_argument(
        "--vision-dtype",
        choices=("bfloat16", "float32"),
        default="float32",
        help=(
            "Vision tower compute precision. pi0 runs SigLIP + projector in "
            "fp32 by default (OpenPI parity) while the rest stays at --dtype; "
            "pass 'bfloat16' to lower the vision tower to bf16."
        ),
    )
    parser.add_argument(
        "--fixed-noise",
        action="store_true",
        help="Use a seed-0 noise tensor for a reproducible run.",
    )
    parser.add_argument(
        "--flashinfer-workspace-mib",
        type=int,
        default=512,
        help=(
            "flashinfer split-k scratch size in MiB "
            "(RuntimeConfig.flashinfer_workspace_bytes). pi0 has no "
            "recommended-engine-config to auto-bump this (unlike pi0.5, "
            "which floors at 256 MiB), so the default 128 MiB overflows the "
            "action-expert paged prefill at larger batch sizes "
            "(bs=4 needs ~206 MiB). 512 MiB covers bs up to ~8."
        ),
    )

    bnb.add_bench_cli_args(parser)
    add_profile_cli_args(parser)

    args = parser.parse_args()

    if args.checkpoint is not None and not args.checkpoint.is_dir():
        raise NotADirectoryError(
            f"--checkpoint must be a directory, got: {args.checkpoint}"
        )

    dtype = _DTYPES[args.dtype]
    use_cuda_graph = not args.no_cuda_graph and args.device == "cuda"
    vision_params_dtype = (
        torch.float32 if args.vision_dtype == "float32" else torch.bfloat16
    )

    # Resolve the effective camera count for the extras record. Mirror the
    # logic in make_setup_fn so the logged value matches what actually ran.
    if args.num_images is not None:
        effective_num_images = args.num_images
    elif args.checkpoint is not None:
        effective_num_images = load_config(args.checkpoint, PI0Config).num_images
    else:
        effective_num_images = PI0Config().num_images

    # Install whatever profiler the CLI requested. NoOp is the default
    # when --profile-backend is "none" (or rank is excluded).
    profile_cfg = profile_config_from_args(args)
    install_profiler(profile_cfg)

    setup_fn = make_setup_fn(
        checkpoint=args.checkpoint,
        dtype=dtype,
        device_target=args.device,
        use_cuda_graph=use_cuda_graph,
        num_images=args.num_images,
        vision_params_dtype=vision_params_dtype,
        fixed_noise=args.fixed_noise,
        flashinfer_workspace_bytes=args.flashinfer_workspace_mib * 1024 * 1024,
    )
    extras_fn = make_extras_fn(
        dtype_name=args.dtype,
        device_target=args.device,
        use_cuda_graph=use_cuda_graph,
        num_images=effective_num_images,
    )

    runner = bnb.NBatchBenchRunner(
        setup_fn=setup_fn,
        extras_fn=extras_fn,
        **bnb.bench_runner_kwargs_from_args(args),
    )
    runner.run()


if __name__ == "__main__":
    main()
