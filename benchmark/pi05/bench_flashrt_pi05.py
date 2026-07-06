#!/usr/bin/env python3
"""FlashRT PI0.5 latency benchmark using the PhyAI bench runner.

This is a thin adapter around FlashRT's public ``flash_rt.load_model(...).predict``
path. It mirrors ``benchmark/bench_n_batch_ws1_pi05.py``: framework-specific
setup lives here, while warmup, timed iterations, JSONL output, and optional
profiling are handled by ``benchmark/bench_n_batch.py``.

Run::

    python benchmark/pi05/bench_flashrt_pi05.py \
        --flashrt-root <FLASHRT_REPO> \
        --checkpoint <PI05_CHECKPOINT> \
        --batch-sizes 1 --n-warmup 100 --n-timed 100 \
        --result-file results/flashrt_pi05.jsonl

Only batch size 1 is supported because the FlashRT PI0.5 public predict path
used here takes one robot request at a time.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
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


class LazySummary(dict):
    def __init__(self, values_fn_or_items):
        if callable(values_fn_or_items):
            super().__init__()
            self._values_fn = values_fn_or_items
        else:
            super().__init__(values_fn_or_items)
            self._values_fn = None

    def items(self):
        if self._values_fn is None:
            return super().items()
        summary = summarize(self._values_fn())
        return (summary or {}).items()


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    return {
        "count": len(xs),
        "mean_ms": float(statistics.fmean(xs)),
        "median_ms": float(statistics.median(xs)),
        "p50_ms": float(np.percentile(xs, 50)),
        "p90_ms": float(np.percentile(xs, 90)),
        "p99_ms": float(np.percentile(xs, 99)),
        "min_ms": xs[0],
        "max_ms": xs[-1],
    }


def make_images(num_views: int, seed: int) -> list[np.ndarray]:
    """Deterministic synthetic HWC uint8 images for latency-only runs."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        for _ in range(num_views)
    ]


def import_flashrt(repo: Path):
    # Use the checked-out FlashRT repository directly; installation is optional.
    sys.path.insert(0, str(repo))
    import flash_rt

    return flash_rt


def make_setup_fn(args: argparse.Namespace):
    flash_rt = import_flashrt(args.flashrt_root)

    def setup_fn(batch_size: int) -> bnb.BenchSpec:
        if batch_size != 1:
            raise ValueError("FlashRT PI0.5 wrapper supports only batch_size=1")

        if args.precision == "bf16":
            # FlashRT uses this environment switch to force the BF16 PI0.5 RTX path.
            os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1"
        else:
            os.environ.pop("FVK_PI05_RTX_FORCE_BF16", None)

        model = flash_rt.load_model(
            args.checkpoint,
            framework="torch",
            config="pi05",
            hardware=args.hardware,
            num_views=args.num_views,
            num_steps=args.chunk_size,
            cache_frames=1,
            use_fp8=(args.precision == "fp8_bf16"),
            use_fp16=(args.precision == "bf16"),
        )
        images = make_images(args.num_views, args.seed)

        # First call sets the prompt and lazily builds/calibrates the runtime graph.
        model.predict(images, prompt=args.prompt)
        torch.cuda.synchronize()

        call_count = 0

        def step() -> None:
            nonlocal call_count
            if call_count == args.n_warmup and hasattr(model._pipe, "latency_records"):
                model._pipe.latency_records.clear()
            model.predict(images)
            # FlashRT PI0.5 may use an internal CUDA stream, so synchronize inside the
            # step to make the common runner's timing boundary conservative.
            torch.cuda.synchronize()
            call_count += 1

        spec = bnb.BenchSpec(
            name="flashrt_pi05",
            step_callable=step,
            teardown_callable=lambda: None,
        )
        spec.flashrt_internal_latency_ms = LazySummary(
            lambda: (
                [float(x) for x in model._pipe.latency_records]
                if hasattr(model, "_pipe") and hasattr(model._pipe, "latency_records")
                else []
            )
        )  # type: ignore[attr-defined]
        return spec

    return setup_fn


def make_extras_fn(args: argparse.Namespace):
    def extras_fn(batch_size: int, spec: bnb.BenchSpec) -> dict[str, Any]:
        return {
            "runtime": "FlashRT",
            "checkpoint": str(args.checkpoint),
            "flashrt_root": str(args.flashrt_root),
            "precision": args.precision,
            "hardware": args.hardware,
            "batch_size_contract": 1,
            "num_views": args.num_views,
            "chunk_size": args.chunk_size,
            "prompt": args.prompt,
            "seed": args.seed,
            "internal_latency_ms": getattr(spec, "flashrt_internal_latency_ms", {}),
            "timing_scope": "FlashRT predict hot path after first graph-building call; common runner uses perf-counter wall time because FlashRT may run work on a non-default CUDA stream",
        }

    return extras_fn


def main() -> None:
    flashrt_default = env_path("FLASHRT_ROOT")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--flashrt-root",
        type=Path,
        default=flashrt_default,
        required=flashrt_default is None,
        help="Path to the FlashRT repository clone. Can also be set with FLASHRT_ROOT.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="PI0.5 checkpoint directory readable by FlashRT.",
    )
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--prompt", default="do something")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("bf16", "fp8_bf16"), default="bf16")
    parser.add_argument("--hardware", default="auto")

    bnb.add_bench_cli_args(parser)
    parser.set_defaults(bench_name="flashrt_pi05", batch_sizes=[1])
    add_profile_cli_args(parser)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FlashRT PI0.5 benchmarking")

    install_profiler(profile_config_from_args(args))
    runner = bnb.NBatchBenchRunner(
        setup_fn=make_setup_fn(args),
        extras_fn=make_extras_fn(args),
        # FlashRT may execute on an internal CUDA stream. Force the common
        # runner's perf-counter path; step() synchronizes CUDA before returning.
        device=torch.device("cpu"),
        **bnb.bench_runner_kwargs_from_args(args),
    )
    runner.run()


if __name__ == "__main__":
    main()
