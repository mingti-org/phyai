#!/usr/bin/env python3
"""vla.cpp PI0.5 ZMQ client benchmark using the PhyAI bench runner.

Start ``vla-server`` separately, then run this client. The common PhyAI runner
handles warmup, timed iterations, JSONL output, and optional profiling. This
script forces CPU timing in the runner because the measured operation is a ZMQ
request to an external server process; CUDA events in the client process would
not cover server-side GPU work.

Run::

    python benchmark/pi05/bench_vlacpp_pi05_client.py \
        --vlacpp-root <VLACPP_REPO> \
        --addr tcp://127.0.0.1:5555 \
        --tokenizer <TOKENIZER_DIR_OR_ID> \
        --stats-json <LIBERO_STATS_JSON> \
        --batch-sizes 1 --n-warmup 100 --n-timed 100 \
        --result-file results/vlacpp_pi05.jsonl

Only batch size 1 is supported because the vla.cpp Python client sends one
request at a time.
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
import bench_n_batch as bnb
from phyai.utils.profile import (
    add_profile_cli_args,
    install_profiler,
    profile_config_from_args,
)


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


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


def make_obs(seed: int, num_views: int, prompt: str) -> dict[str, Any]:
    """Create deterministic synthetic observations for latency-only requests.

    vla.cpp's official PI0.5 client expects CHW float32 images in [0, 1] and an
    unnormalized robot state vector; it converts these into server protobufs.
    """
    rng = np.random.default_rng(seed)
    obs: dict[str, Any] = {
        "observation.images.image": rng.random((3, 224, 224), dtype=np.float32),
        "observation.state": rng.standard_normal(8).astype(np.float32),
        "task": prompt,
        "prompt": prompt,
    }
    if num_views >= 2:
        obs["observation.images.image2"] = rng.random((3, 224, 224), dtype=np.float32)
    if num_views >= 3:
        obs["observation.images.image3"] = rng.random((3, 224, 224), dtype=np.float32)
    return obs


def make_setup_fn(args: argparse.Namespace):
    eval_root = args.vlacpp_root / "eval"
    sys.path.insert(0, str(eval_root))
    sys.path.insert(0, str(eval_root / "client"))
    from client.vla_cpp_client import VlaCppClient

    def setup_fn(batch_size: int) -> bnb.BenchSpec:
        if batch_size != 1:
            raise ValueError("vla.cpp PI0.5 wrapper supports only batch_size=1")

        image_keys = ["observation.images.image"]
        if args.num_views >= 2:
            image_keys.append("observation.images.image2")
        if args.num_views >= 3:
            image_keys.append("observation.images.image3")

        client = VlaCppClient(
            vla_addr=args.addr,
            arch=args.arch,
            tokenizer_name=args.tokenizer,
            image_keys=image_keys,
            max_length=args.max_length,
            real_action_dim=args.real_action_dim,
            n_action_steps=1,
            stats_json=args.stats_json,
        )
        obs = make_obs(args.seed, args.num_views, args.prompt)
        phase_samples: dict[str, list[float]] = {
            "server_total_latency_ms": [],
            "server_vision_latency_ms": [],
            "server_inference_latency_ms": [],
            "server_prefill_latency_ms": [],
            "server_denoise_latency_ms": [],
        }
        phase_summary: dict[str, Any] = {}
        call_count = 0

        def update_phase_summary() -> None:
            phase_summary.clear()
            for key, values in phase_samples.items():
                phase_summary[key] = summarize(values)

        def step() -> None:
            nonlocal call_count
            client.get_action(obs)
            call_count += 1
            if call_count <= args.n_warmup:
                return
            r = getattr(client, "_last_response", None)
            if r is None:
                return
            phase_samples["server_total_latency_ms"].append(float(r.latency_ms_total))
            phase_samples["server_vision_latency_ms"].append(float(r.latency_ms_vision))
            phase_samples["server_inference_latency_ms"].append(
                float(r.latency_ms_inference)
            )
            phase_samples["server_prefill_latency_ms"].append(
                float(r.latency_ms_prefill)
            )
            phase_samples["server_denoise_latency_ms"].append(
                float(r.latency_ms_denoise)
            )
            update_phase_summary()

        def teardown() -> None:
            sock = getattr(client, "sock", None)
            if sock is not None:
                sock.close(linger=0)

        spec = bnb.BenchSpec(
            name="vlacpp_pi05_zmq_client",
            step_callable=step,
            teardown_callable=teardown,
        )
        # Attach dynamic summary for extras_fn. The runner copies the dict after
        # timed steps finish, so it records the final server phase statistics.
        spec.vlacpp_phase_summary = phase_summary  # type: ignore[attr-defined]
        return spec

    return setup_fn


def make_extras_fn(args: argparse.Namespace):
    def extras_fn(batch_size: int, spec: bnb.BenchSpec) -> dict[str, Any]:
        return {
            "runtime": "vla.cpp",
            "vlacpp_root": str(args.vlacpp_root),
            "addr": args.addr,
            "arch": args.arch,
            "tokenizer": args.tokenizer,
            "stats_json": str(args.stats_json) if args.stats_json else None,
            "batch_size_contract": 1,
            "num_views": args.num_views,
            "chunk_size_metadata": args.chunk_size,
            "prompt": args.prompt,
            "seed": args.seed,
            "server_phase_latency_ms": getattr(spec, "vlacpp_phase_summary", {}),
            "timing_scope": "client ZMQ request wall time; server phase timings are copied from PredictResponse extras",
            "notes": "vla.cpp server enforces the GGUF chunk size; prefix/expert may be combined in server phase timing.",
        }

    return extras_fn


def main() -> None:
    vlacpp_default = env_path("VLACPP_ROOT")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--vlacpp-root",
        type=Path,
        default=vlacpp_default,
        required=vlacpp_default is None,
        help="Path to the vla.cpp repository clone. Can also be set with VLACPP_ROOT.",
    )
    parser.add_argument("--addr", default="tcp://127.0.0.1:5555")
    parser.add_argument("--arch", default="pi05")
    parser.add_argument(
        "--tokenizer",
        default="google/paligemma-3b-pt-224",
        help="Tokenizer name or local tokenizer path. Prefer a local path if the HF repo is gated.",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Local LIBERO meta/stats.json for arch=pi05; avoids network fetch.",
    )
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--prompt", default="do something")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--real-action-dim", type=int, default=7)

    bnb.add_bench_cli_args(parser)
    parser.set_defaults(bench_name="vlacpp_pi05", batch_sizes=[1])
    add_profile_cli_args(parser)
    args = parser.parse_args()

    install_profiler(profile_config_from_args(args))
    runner = bnb.NBatchBenchRunner(
        setup_fn=make_setup_fn(args),
        extras_fn=make_extras_fn(args),
        # Force perf-counter timing: the GPU work happens in the vla-server process.
        device=torch.device("cpu"),
        **bnb.bench_runner_kwargs_from_args(args),
    )
    runner.run()


if __name__ == "__main__":
    main()
