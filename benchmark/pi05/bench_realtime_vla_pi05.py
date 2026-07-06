#!/usr/bin/env python3
"""realtime-vla PI0.5 latency benchmark using the PhyAI bench runner.

This script is a thin adapter around realtime-vla's ``Pi05Inference.forward``.
It mirrors ``benchmark/bench_n_batch_ws1_pi05.py`` by delegating warmup, timed
iterations, JSONL output, and optional profiling to ``benchmark/bench_n_batch.py``.

Run::

    python benchmark/pi05/bench_realtime_vla_pi05.py \
        --realtime-vla-root <REALTIME_VLA_REPO> \
        --flashrt-root <FLASHRT_REPO> \
        --checkpoint <PI05_CHECKPOINT_OR_CONVERTED_FILE> \
        --batch-sizes 1 --n-warmup 100 --n-timed 100 \
        --result-file results/realtime_vla_pi05.jsonl

Only batch size 1 is supported because realtime-vla's PI0.5 inference object
used here is allocated for one synthetic request.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import torch

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BENCHMARK_DIR))
import bench_n_batch as bnb  # noqa: E402
from phyai.utils.profile import (  # noqa: E402
    add_profile_cli_args,
    install_profiler,
    profile_config_from_args,
)


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def load_checkpoint(path: Path, flashrt_root: Path | None, trust_pickle: bool):
    if path.is_dir():
        path = path / "model.safetensors"
    if path.suffix == ".safetensors":
        if flashrt_root is None:
            raise ValueError(
                "--flashrt-root or FLASHRT_ROOT is required when loading a safetensors checkpoint"
            )
        # Reuse FlashRT's tested PI0.5 key conversion instead of duplicating it here.
        sys.path.insert(0, str(flashrt_root))
        from flash_rt.frontends.torch.pi05_rtx import convert_pi05_safetensors

        return convert_pi05_safetensors(path)
    if path.suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=True)
    if path.suffix in {".pkl", ".pickle"}:
        if not trust_pickle:
            raise ValueError(
                "Refusing to load pickle checkpoint without --trust-pickle-checkpoint. "
                "Only use that flag for checkpoints from a trusted source."
            )
        with path.open("rb") as f:
            return pickle.load(f)  # nosec B301: guarded by --trust-pickle-checkpoint.
    raise ValueError(
        f"Unsupported checkpoint suffix {path.suffix!r}; expected .safetensors, .pt, .pth, .pkl, or .pickle"
    )


def make_setup_fn(args: argparse.Namespace):
    # Use the checked-out realtime-vla repository directly; installation is optional.
    sys.path.insert(0, str(args.realtime_vla_root))
    from pi05_infer import Pi05Inference

    def setup_fn(batch_size: int) -> bnb.BenchSpec:
        if batch_size != 1:
            raise ValueError("realtime-vla PI0.5 wrapper supports only batch_size=1")

        checkpoint = load_checkpoint(
            args.checkpoint, args.flashrt_root, args.trust_pickle_checkpoint
        )
        if "language_embeds" not in checkpoint:
            # Latency-only fallback for checkpoints that do not include prompt embeds.
            checkpoint["language_embeds"] = torch.randn(
                args.prompt_len, 2048, dtype=torch.bfloat16
            )

        infer = Pi05Inference(
            checkpoint=checkpoint,
            num_views=args.num_views,
            chunk_size=args.chunk_size,
            tokenizer_path=str(args.tokenizer) if args.tokenizer else None,
            discrete_state_input=args.discrete_state_input,
        )
        torch.manual_seed(args.seed)
        input_image = torch.randn(
            args.num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda"
        )
        input_noise = torch.randn(
            args.chunk_size, 32, dtype=torch.bfloat16, device="cuda"
        )

        state_tokens = None
        if args.discrete_state_input:
            import numpy as np

            state_tokens = np.zeros(args.state_dim, dtype=np.int64)

        def step() -> None:
            infer.forward(
                input_image,
                input_noise,
                task_prompt=args.prompt,
                state_tokens=state_tokens,
            )

        return bnb.BenchSpec(
            name="realtime_vla_pi05",
            step_callable=step,
            teardown_callable=lambda: None,
        )

    return setup_fn


def make_extras_fn(args: argparse.Namespace):
    def extras_fn(batch_size: int, spec: bnb.BenchSpec) -> dict[str, Any]:
        return {
            "runtime": "realtime-vla",
            "checkpoint": str(args.checkpoint),
            "realtime_vla_root": str(args.realtime_vla_root),
            "flashrt_root": str(args.flashrt_root) if args.flashrt_root else None,
            "precision": "bf16",
            "batch_size_contract": 1,
            "num_views": args.num_views,
            "chunk_size": args.chunk_size,
            "prompt": args.prompt,
            "prompt_len": args.prompt_len,
            "seed": args.seed,
            "discrete_state_input": args.discrete_state_input,
            "timing_scope": "Pi05Inference.forward hot path; common runner CUDA event wraps one forward call",
        }

    return extras_fn


def main() -> None:
    realtime_default = env_path("REALTIME_VLA_ROOT")
    flashrt_default = env_path("FLASHRT_ROOT")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--realtime-vla-root",
        type=Path,
        default=realtime_default,
        required=realtime_default is None,
        help="Path to the realtime-vla repository clone. Can also be set with REALTIME_VLA_ROOT.",
    )
    parser.add_argument(
        "--flashrt-root",
        type=Path,
        default=flashrt_default,
        help="Path to FlashRT. Required only when --checkpoint is a PI0.5 safetensors checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="realtime-vla .pkl/.pt checkpoint, PI0.5 model.safetensors, or a directory containing model.safetensors.",
    )
    parser.add_argument(
        "--trust-pickle-checkpoint",
        action="store_true",
        help="Allow loading .pkl/.pickle checkpoints. Only use with trusted checkpoint files.",
    )
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--prompt", default="do something")
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--discrete-state-input",
        action="store_true",
        help="Use realtime-vla tokenizer/state-token prompt path instead of precomputed language_embeds.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Tokenizer path for --discrete-state-input.",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=8,
        help="Synthetic state token count for --discrete-state-input.",
    )

    bnb.add_bench_cli_args(parser)
    parser.set_defaults(bench_name="realtime_vla_pi05", batch_sizes=[1])
    add_profile_cli_args(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for realtime-vla PI0.5 benchmarking")

    install_profiler(profile_config_from_args(args))
    runner = bnb.NBatchBenchRunner(
        setup_fn=make_setup_fn(args),
        extras_fn=make_extras_fn(args),
        device=torch.device("cuda", torch.cuda.current_device()),
        **bnb.bench_runner_kwargs_from_args(args),
    )
    runner.run()


if __name__ == "__main__":
    main()
