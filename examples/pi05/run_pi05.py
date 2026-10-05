"""Run PI0.5 inference inline on one GPU, or on managed multi-GPU replicas.

Spins up the pi0.5 plugin behind ``Engine`` and runs ``--batch-size`` robots.
``--replica-count N`` spawns one worker process per replica on the first ``N``
visible GPUs (pick physical GPUs with ``CUDA_VISIBLE_DEVICES``) and routes each
step to one of them.
By default it demonstrates the full preprocessing flow: a
``phyai_utils_tools.models.pi05.PI05Processor`` turns raw cameras + a task
string + a state vector into the canonical ``PI05Request`` tensors, the engine
runs, and the processor's ``postprocess`` converts the raw action chunk back.
With ``--raw`` it skips the processor and feeds canonical random tensors
directly (useful for pure engine timing without the tokenizer load). Action
numbers are meaningless (inputs are random); this verifies wiring + timing.

Run::

    uv run python examples/pi05/run_pi05.py --checkpoint /path/to/pi05_base/

The argument is a HuggingFace-style checkpoint **folder** (or a HuggingFace
repo id, downloaded on first use): it must contain ``config.json`` and either
``model.safetensors`` or ``model.safetensors.index.json`` plus its shards.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from phyai import DeploymentConfig, Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_pi05 import PI05Request
from phyai.server import WorkerSupervisorConfig
from phyai.utils import load_config


def make_raw_request(
    *,
    batch_size: int,
    num_images: int,
    plugin_cfg: PI05Config,
    device: torch.device,
    dtype: torch.dtype,
) -> PI05Request:
    """Build a canonical ``PI05Request`` directly (the ``--raw`` path).

    Bypasses the processor: random already-resized pixels + a one-token prompt.
    Used for pure engine timing so the tokenizer load doesn't pollute it.
    """
    pixel_values = torch.rand(
        batch_size,
        num_images,
        3,
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
    return PI05Request(
        pixel_values=pixel_values,
        input_ids=input_ids,
        lang_lens=lang_lens,
    )


def make_processed_request(
    processor,
    *,
    batch_size: int,
    num_images: int,
    plugin_cfg: PI05Config,
    device: torch.device,
    dtype: torch.dtype,
) -> PI05Request:
    """Build a ``PI05Request`` via the full preprocessing pipeline.

    Feeds raw (native-resolution) random cameras + task strings + a random
    state through ``processor.preprocess`` and adapts the resulting
    ``PI05ProcessedInputs`` into the canonical request. This is the real-world
    entry path: ``phyai`` itself never resizes or tokenizes.
    """
    # Native-resolution cameras (deliberately not 224) to see if auto resize is applied.
    images = [
        torch.rand(batch_size, 3, 480, 640, device=device) for _ in range(num_images)
    ]
    tasks = ["pick up the cup"] * batch_size
    state = torch.rand(batch_size, 7, device=device) * 2 - 1  # already in [-1, 1]
    processed = processor.preprocess({"images": images, "task": tasks, "state": state})
    return PI05Request(
        pixel_values=processed.pixel_values.to(device=device, dtype=dtype),
        input_ids=processed.input_ids.to(device),
        lang_lens=processed.lang_lens.to(device),
    )


def benchmark(
    engine: Engine,
    request: PI05Request,
    *,
    n_warmup: int,
    n_timed: int,
    synchronize_cuda: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Warm + ``n_timed`` ``engine.step`` calls; return last action + ms stats."""
    actions: torch.Tensor | None = None
    for _ in range(n_warmup):
        actions = engine.step(request)
    if synchronize_cuda:
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(n_timed):
        started = time.perf_counter()
        actions = engine.step(request)
        if synchronize_cuda:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - started) * 1000)

    assert actions is not None
    return actions, {
        "mean": statistics.fmean(times_ms),
        "median": statistics.median(times_ms),
        "stdev": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min": min(times_ms),
        "max": max(times_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "pi05_base checkpoint: a local folder, or a HuggingFace repo id "
            "(downloaded on first use). Must contain config.json and either "
            "model.safetensors or model.safetensors.index.json with its shards."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of robots per request (also sets the engine's max_batch_size).",
    )
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-timed", type=int, default=30)
    parser.add_argument(
        "--replica-count",
        type=int,
        default=1,
        help="Number of independent serving replicas.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for managed workers to load the model.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=3,
        help="Number of cameras per robot (default 3, the pi05_base contract).",
    )
    parser.add_argument(
        "--vision-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help=(
            "Vision tower compute precision. 'float32' runs SigLIP + projector "
            "in fp32 (openpi/lerobot parity) while the rest stays bf16."
        ),
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Skip the PI05Processor and feed canonical random tensors directly "
            "(pure engine timing; no tokenizer load)."
        ),
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help=(
            "Enable debug tensor dumping to this directory: every leaf "
            "operator's output is written to <dir>/rank{R}_pid{P}/pass{N}.pt, "
            "one file per engine.step(). Forces use_cuda_graph=False (forward "
            "hooks can't fire under a captured graph), so timing here reflects "
            "eager mode. Load a pass with phyai.runtime.tensor_dump.load_pass."
        ),
    )
    parser.add_argument(
        "--dump-filter",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Restrict tensor dumping to operators whose dotted name matches "
            "any of these regexes (e.g. --dump-filter 'expert_stack\\.layers\\.0\\.' "
            "'\\.heads\\.'). Omit to dump every operator. No effect without "
            "--dump-dir; mutually exclusive with --dump-filter-fn."
        ),
    )
    parser.add_argument(
        "--dump-filter-fn",
        type=str,
        default=None,
        help=(
            "Path to a (name, module) -> bool predicate for tensor-dump "
            "selection, as 'pkg.module:func' or '/path/to/file.py:func'. For "
            "logic a regex can't express. No effect without --dump-dir; "
            "mutually exclusive with --dump-filter."
        ),
    )
    args = parser.parse_args()

    if args.replica_count < 1:
        raise SystemExit("--replica-count must be positive.")
    if args.batch_size < 1 or args.n_warmup < 0 or args.n_timed < 1:
        raise SystemExit(
            "--batch-size and --n-timed must be positive; --n-warmup cannot be negative."
        )

    plugin_cfg = load_config(args.checkpoint, PI05Config)
    dtype = torch.bfloat16
    vision_dtype = torch.float32 if args.vision_dtype == "float32" else None
    inputs_image_shape = [
        [plugin_cfg.vision.image_size, plugin_cfg.vision.image_size, 3]
        for _ in range(args.num_images)
    ]

    engine_args = EngineArgs(
        plugin="pi05",
        plugin_args=PI05Args(
            checkpoint_dir=args.checkpoint,
            max_batch_size=args.batch_size,
            vision_params_dtype=vision_dtype,
            inputs_image_shape=inputs_image_shape,
        ),
        config=EngineConfig(
            device=DeviceConfig(target="cuda", params_dtype=dtype),
            runtime=RuntimeConfig(
                use_cuda_graph=args.dump_dir is None,
                debug_tensor_dump_dir=(
                    str(args.dump_dir) if args.dump_dir is not None else None
                ),
                debug_tensor_dump_filter=(
                    tuple(args.dump_filter) if args.dump_filter is not None else None
                ),
                debug_tensor_dump_filter_fn=args.dump_filter_fn,
            ),
        ),
    )
    # The engine picks the executor: one replica runs inline; more spawn one
    # managed worker per replica on the first visible GPUs (select physical
    # GPUs with CUDA_VISIBLE_DEVICES on this launcher).
    deployment = DeploymentConfig(
        replica_count=args.replica_count,
        process_config=WorkerSupervisorConfig(startup_timeout_s=args.startup_timeout),
    )

    engine = Engine(engine_args, deployment=deployment)
    try:
        # Managed workers receive requests over a process pipe, so inputs are
        # built on the CPU; the inline engine takes them on the GPU directly.
        managed = engine.mode != "inline"
        request_device = torch.device("cpu" if managed else "cuda")
        processor = None
        if args.raw:
            request = make_raw_request(
                batch_size=args.batch_size,
                num_images=args.num_images,
                plugin_cfg=plugin_cfg,
                device=request_device,
                dtype=dtype,
            )
        else:
            # Lazy import so --raw runs without phyai_utils_tools / tokenizer load.
            from phyai_utils_tools.models.pi05 import PI05Processor

            processor = PI05Processor(
                image_size=plugin_cfg.vision.image_size,
                num_channels=plugin_cfg.vision.num_channels,
                num_images=args.num_images,
                tokenizer_max_length=plugin_cfg.tokenizer_max_length,
                action_dim=plugin_cfg.max_action_dim,
                device=request_device,
                params_dtype=dtype,
            )
            request = make_processed_request(
                processor,
                batch_size=args.batch_size,
                num_images=args.num_images,
                plugin_cfg=plugin_cfg,
                device=request_device,
                dtype=dtype,
            )

        # The dispatcher round-robins idle replicas, so warm each one at least
        # once; otherwise a cold replica's first step lands in the timed window.
        actions, stats = benchmark(
            engine,
            request,
            n_warmup=max(args.n_warmup, args.replica_count),
            n_timed=args.n_timed,
            synchronize_cuda=not managed,
        )

        # Managed results are CUDA-IPC views of worker memory; take a durable
        # CPU copy before the CPU-side postprocess.
        if managed:
            actions = actions.cpu()

        # Postprocess the raw action chunk through the processor (no-op slice
        # here since action_dim == max_action_dim for the default config, but
        # this is where unnormalization / trimming would happen with stats).
        if processor is not None:
            actions = processor.postprocess(actions)

        print(f"action chunk shape : {tuple(actions.shape)}")
        print(f"action chunk dtype : {actions.dtype}")
        print(f"action chunk device: {actions.device}")
        print(
            f"executor           : {engine.mode} "
            f"({args.replica_count} replica{'s' if args.replica_count != 1 else ''})"
        )
        print(
            f"step latency       : mean={stats['mean']:.2f} ms  "
            f"median={stats['median']:.2f} ms  std={stats['stdev']:.2f} ms  "
            f"min={stats['min']:.2f} ms  max={stats['max']:.2f} ms  "
            f"(n_warmup={args.n_warmup}, n_timed={args.n_timed})"
        )
        print(f"first action row   : {actions[0, 0].float().tolist()}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
