"""Run GR00T-N1.7 inference end-to-end through the phyai engine plugin path.

The ``phyai_utils_tools.models.gr00t.GR00TProcessor`` turns raw video, state,
and language inputs into ``GR00TN17Request`` tensors; the engine runs the
Qwen3-VL backbone plus the flow-matching action head; the processor decodes
the normalized action chunk back to physical units. ``phyai`` owns model
execution, while preprocessing and decoding stay in ``phyai-utils-tools``.

Run::

    PYTHONPATH=phyai/src:phyai-utils-tools/src \\
    python examples/gr00t/run_gr00t.py \\
        --checkpoint <gr00t-checkpoint-dir> \\
        --embodiment-tag LIBERO_PANDA

Inputs are synthetic, so action values are not meaningful. This script checks
the processor -> engine -> scheduler -> decoder wiring and reports engine-step
latency. Common options include ``--batch-size``, ``--image-size``,
``--task``, ``--params-dtype``, ``--no-cuda-graph``, ``--online``, and
``--backbone-model-name-or-path``. Pass ``--dump-dir`` to capture debug tensor
dumps; dumping disables CUDA graphs because forward hooks do not run inside
captured graphs.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np
import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.gr00t_n17.configuration_gr00t_n17 import GR00TN17Config
from phyai.models.gr00t_n17.main_gr00t_n17 import GR00TN17Args
from phyai.models.gr00t_n17.scheduler_ws1_gr00t_n17 import GR00TN17Request
from phyai.utils import load_config


def make_synthetic_observation(
    processor,
    *,
    batch_size: int,
    image_size: int,
    task: str,
):
    """Build a random ``GR00TObservation`` matching the checkpoint modality config.

    Shapes/dtypes follow what the processor validates (uint8 ``(B,T,H,W,3)`` video,
    float32 ``(B,T,D)`` state, ``[[task]]`` language).
    """
    from phyai_utils_tools.models.gr00t import GR00TObservation

    cfg = processor.modality_config
    tag = processor.embodiment_tag
    video = {}
    for key in cfg["video"].modality_keys:
        t = len(cfg["video"].delta_indices)
        video[key] = np.random.randint(
            0, 256, size=(batch_size, t, image_size, image_size, 3), dtype=np.uint8
        )
    state = {}
    for key in cfg["state"].modality_keys:
        t = len(cfg["state"].delta_indices)
        try:
            dim = int(processor.norm_params[tag]["state"][key]["dim"])
        except (KeyError, TypeError):
            dim = 7
        state[key] = np.random.rand(batch_size, t, dim).astype(np.float32) * 2 - 1
    language_key = cfg["language"].modality_keys[0]
    language = {language_key: [[task]] * batch_size}
    return GR00TObservation(video=video, state=state, language=language)


def benchmark(
    engine: Engine,
    request: GR00TN17Request,
    *,
    n_warmup: int,
    n_timed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Warm up, then time ``n_timed`` ``engine.step`` calls."""
    action: torch.Tensor | None = None
    for _ in range(n_warmup):
        action = engine.step(request)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(n_timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        action = engine.step(request)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    assert action is not None
    return action, {
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
        help="GR00T checkpoint folder (config.json + safetensors + processor_config/statistics).",
    )
    parser.add_argument(
        "--embodiment-tag",
        default="LIBERO_PANDA",
        help="Embodiment tag for the processor (e.g. LIBERO_PANDA).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Synthetic camera H/W (the eval transform handles arbitrary sizes).",
    )
    parser.add_argument(
        "--task",
        default="pick up the object",
        help="Synthetic language instruction passed through the GR00T processor.",
    )
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-timed", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--params-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Engine parameter/activation dtype.",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Run without CUDA graph capture.",
    )
    parser.add_argument(
        "--backbone-model-name-or-path",
        type=str,
        default=None,
        help=(
            "Optional Qwen3-VL backbone tokenizer/config path or HuggingFace repo id. "
            "Defaults to the backbone model name stored in the GR00T config."
        ),
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow HuggingFace downloads (default: local_files_only).",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help=(
            "Enable debug tensor dumping to this directory: every leaf operator's "
            "output is written to <dir>/rank{R}_pid{P}/pass{N}.pt, one file per "
            "engine.step(). Forces use_cuda_graph=False because forward hooks do "
            "not run under a captured graph. Load a pass with "
            "phyai.runtime.tensor_dump.load_pass."
        ),
    )
    parser.add_argument(
        "--dump-filter",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Dump only operators whose dotted name matches any of these regexes "
            "(e.g. --dump-filter 'action_head\\.model\\.transformer_blocks\\.' "
            "'\\.visual\\.'). Omit to dump every operator. No effect without "
            "--dump-dir."
        ),
    )
    parser.add_argument(
        "--dump-filter-fn",
        type=str,
        default=None,
        help=(
            "Path to a (name, module) -> bool predicate as 'pkg.module:func' or "
            "'/path/to/file.py:func'. No effect without --dump-dir; mutually "
            "exclusive with --dump-filter."
        ),
    )
    args = parser.parse_args()

    if args.n_warmup < 0 or args.n_timed <= 0:
        parser.error("--n-warmup must be >= 0 and --n-timed must be > 0.")
    if args.dump_filter is not None and args.dump_filter_fn is not None:
        parser.error("--dump-filter and --dump-filter-fn are mutually exclusive.")

    if not args.checkpoint.is_dir():
        raise NotADirectoryError(
            f"--checkpoint must be a directory, got: {args.checkpoint}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.params_dtype]
    loading_kwargs = {"trust_remote_code": True, "local_files_only": not args.online}
    use_cuda_graph = args.dump_dir is None and not args.no_cuda_graph

    engine = Engine(
        EngineArgs(
            plugin="gr00t_n17",
            plugin_args=GR00TN17Args(
                checkpoint_dir=args.checkpoint,
                max_batch_size=args.batch_size,
                backbone_model_name_or_path=args.backbone_model_name_or_path,
                backbone_transformers_loading_kwargs=loading_kwargs,
            ),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=dtype),
                runtime=RuntimeConfig(
                    use_cuda_graph=use_cuda_graph,
                    debug_tensor_dump_dir=(
                        str(args.dump_dir) if args.dump_dir is not None else None
                    ),
                    debug_tensor_dump_filter=(
                        tuple(args.dump_filter)
                        if args.dump_filter is not None
                        else None
                    ),
                    debug_tensor_dump_filter_fn=args.dump_filter_fn,
                ),
            ),
        )
    )
    try:
        # Lazy import keeps --help fast and the engine import free of the processor.
        from phyai_utils_tools.models.gr00t import GR00TProcessor

        cfg = load_config(args.checkpoint, GR00TN17Config)
        processor = GR00TProcessor.from_pretrained(
            args.checkpoint,
            embodiment_tag=args.embodiment_tag,
            model_name=args.backbone_model_name_or_path or cfg.backbone.model_name,
            transformers_loading_kwargs=loading_kwargs,
        )
        observation = make_synthetic_observation(
            processor,
            batch_size=args.batch_size,
            image_size=args.image_size,
            task=args.task,
        )
        prepared = processor.process_observation(observation)
        request = GR00TN17Request(tensors=prepared.tensors)

        normalized_action, stats = benchmark(
            engine,
            request,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
        )

        action = processor.decode_action(
            normalized_action, raw_state=prepared.raw_state
        )

        print(f"normalized action shape: {tuple(normalized_action.shape)}")
        print(f"normalized action dtype: {normalized_action.dtype}")
        print(f"normalized action device: {normalized_action.device}")
        print(f"decoded action keys    : {sorted(action.keys())}")
        print(
            f"step latency           : mean={stats['mean']:.2f} ms  "
            f"median={stats['median']:.2f} ms  std={stats['stdev']:.2f} ms  "
            f"min={stats['min']:.2f} ms  max={stats['max']:.2f} ms  "
            f"(n_warmup={args.n_warmup}, n_timed={args.n_timed})"
        )
        first_key = sorted(action.keys())[0]
        print(f"{first_key}[0, 0]: {np.asarray(action[first_key])[0, 0].tolist()}")
        if args.dump_dir is not None:
            print(
                f"tensor dump written to : {args.dump_dir} "
                "(rank*/pass*.pt; "
                "load with phyai.runtime.tensor_dump.load_pass)"
            )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
