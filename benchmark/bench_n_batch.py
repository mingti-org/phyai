"""Generic n-batch sweep + profile harness for phyai engines.

Most phyai benchmarks share the same outline: build an engine for some
batch size, do a few warmup steps, time a fixed number of inference
steps with CUDA events, then tear the engine down and move on to the
next batch size. The interesting variation is everything around it —
how the engine is constructed, which profiler is wrapped around the
timed window, and how results are aggregated.

This module factors out the invariant scaffolding so a per-model
script (see ``bench_n_batch_ws1_pi05.py``) reduces to:

1. A ``setup_fn(batch_size) -> BenchSpec`` closure that builds the
   engine and returns the zero-arg ``step_callable`` to time and a
   ``teardown_callable`` to release the engine after the sweep step.
2. CLI plumbing via :func:`add_bench_cli_args` plus the profile flags
   from :mod:`phyai.utils.profile`.

The runner records per-step latency with :class:`torch.cuda.Event`
when CUDA is available (perf-counter fallback otherwise), produces a
full percentile spread (mean / median / p50 / p90 / p99 / stdev / min
/ max) per batch size, and appends one JSON line per result to
``--result-file``. The profile window is gated by
``--profile-start-step`` / ``--profile-num-steps`` so a long timed
loop can keep a tight, file-size-bounded trace window.

Run via the per-model script::

    LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat/lib uv run python \\
        benchmark/bench_n_batch_ws1_pi05.py \\
        --checkpoint /path/to/pi05_base \\
        --batch-sizes 1 2 4 --n-warmup 5 --n-timed 30 \\
        --profile-backend torch --profile-start-step 5 --profile-num-steps 3

For nsys::

    nsys profile --capture-range=cudaProfilerApi \\
        --capture-range-end=stop -o trace.nsys-rep \\
        uv run python benchmark/bench_n_batch_ws1_pi05.py \\
        --checkpoint /path/to/pi05_base --batch-sizes 4 \\
        --profile-backend nsys --profile-start-step 5 --profile-num-steps 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from phyai.utils.profile import (
    Profiler,
    event_scope,
    get_profiler,
)


# ============================================================================ #
# Specs and results                                                            #
# ============================================================================ #


@dataclass
class BenchSpec:
    """What :func:`setup_fn` returns for one batch-size invocation.

    ``step_callable`` is a zero-arg closure: typically
    ``lambda: engine.step(request)``. ``teardown_callable`` releases
    whatever the setup allocated — usually ``engine.close``.
    """

    name: str
    step_callable: Callable[[], Any]
    teardown_callable: Callable[[], None]


@dataclass
class BenchResult:
    """One row of bench output, also one JSONL line.

    ``extras`` is a freeform dict so scripts can stash run-specific
    fields (model variant, dtype, kv pool size, ...) without
    extending the schema.
    """

    run_name: str
    bench_name: str
    batch_size: int
    n_warmup: int
    n_timed: int
    latency_ms_mean: float
    latency_ms_median: float
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p99: float
    latency_ms_stdev: float
    latency_ms_min: float
    latency_ms_max: float
    throughput_samples_per_s: float
    extras: dict[str, Any] = field(default_factory=dict)


# ============================================================================ #
# Runner                                                                       #
# ============================================================================ #


class NBatchBenchRunner:
    """Sweep a sequence of batch sizes, time each, optionally profile.

    Lifecycle per batch size:

    1. ``setup_fn(batch_size)`` builds the engine and returns a
       :class:`BenchSpec`.
    2. ``n_warmup`` warm steps run with the profiler off.
    3. ``n_timed`` timed steps run; per-step latency is recorded with
       :class:`torch.cuda.Event` (perf-counter when CUDA is absent).
    4. If a profiler is installed, it starts at
       ``profile_start_step`` (0-indexed into the timed loop) and
       stops after ``profile_num_steps`` (or at the end of the timed
       loop if ``profile_num_steps`` is ``None``).
    5. The spec's ``teardown_callable`` runs, even on exceptions.

    The runner doesn't touch :func:`phyai.utils.profile.install_profiler`
    — the bench script is expected to install whichever profiler it
    wants before calling :meth:`run`. The runner just calls ``start``
    and ``stop`` on whatever :func:`get_profiler` returns.
    """

    def __init__(
        self,
        *,
        setup_fn: Callable[[int], BenchSpec],
        batch_sizes: Sequence[int],
        n_warmup: int = 5,
        n_timed: int = 30,
        profile_start_step: int | None = None,
        profile_num_steps: int | None = None,
        run_name: str = "default",
        bench_name: str = "n_batch",
        result_file: Path | None = None,
        device: torch.device | None = None,
        extras_fn: Callable[[int, BenchSpec], dict[str, Any]] | None = None,
    ) -> None:
        if not batch_sizes:
            raise ValueError("batch_sizes must be a non-empty sequence")
        if n_warmup < 0:
            raise ValueError(f"n_warmup must be >= 0, got {n_warmup}")
        if n_timed <= 0:
            raise ValueError(f"n_timed must be > 0, got {n_timed}")
        if profile_start_step is not None and profile_start_step < 0:
            raise ValueError(
                f"profile_start_step must be >= 0 if set, got {profile_start_step}"
            )
        if profile_num_steps is not None and profile_num_steps <= 0:
            raise ValueError(
                f"profile_num_steps must be > 0 if set, got {profile_num_steps}"
            )
        if profile_start_step is not None and profile_start_step >= n_timed:
            raise ValueError(
                f"profile_start_step={profile_start_step} is past the timed loop "
                f"(n_timed={n_timed}); no profile would ever be collected."
            )

        self.setup_fn = setup_fn
        self.batch_sizes = list(batch_sizes)
        self.n_warmup = int(n_warmup)
        self.n_timed = int(n_timed)
        self.profile_start_step = profile_start_step
        self.profile_num_steps = profile_num_steps
        self.run_name = run_name
        self.bench_name = bench_name
        self.result_file = Path(result_file) if result_file else None
        self.device = device if device is not None else _infer_device()
        self.extras_fn = extras_fn

    # ------------------------------------------------------------------ #
    # Driver                                                             #
    # ------------------------------------------------------------------ #

    def run(self) -> list[BenchResult]:
        """Run the sweep and return per-batch-size results.

        Also appends each result as one JSON line to ``result_file``
        when set. Per-batch-size teardown runs in a ``finally`` so a
        crash in one configuration doesn't leak the engine.
        """
        results: list[BenchResult] = []

        for bs in self.batch_sizes:
            print(
                f"\n[bench] batch_size={bs}  warmup={self.n_warmup}  "
                f"timed={self.n_timed}  device={self.device}"
            )
            spec = self.setup_fn(bs)
            try:
                extras = self.extras_fn(bs, spec) if self.extras_fn else {}
                result = self._run_one(spec, batch_size=bs, extras=extras)
                results.append(result)
                _print_result(result)
            finally:
                try:
                    spec.teardown_callable()
                except Exception as e:  # pragma: no cover - teardown best-effort
                    print(f"[bench] teardown error for bs={bs}: {e!r}")

        if self.result_file is not None and results:
            _append_jsonl(self.result_file, results)
            print(f"\n[bench] appended {len(results)} result(s) to {self.result_file}")

        _print_summary_table(results)
        return results

    # ------------------------------------------------------------------ #
    # Inner loop                                                         #
    # ------------------------------------------------------------------ #

    def _run_one(
        self,
        spec: BenchSpec,
        *,
        batch_size: int,
        extras: dict[str, Any],
    ) -> BenchResult:
        step_callable = spec.step_callable
        profiler: Profiler = get_profiler()
        use_cuda = self.device.type == "cuda" and torch.cuda.is_available()

        # ---- Warmup ----
        for _ in range(self.n_warmup):
            step_callable()
        if use_cuda:
            torch.cuda.synchronize()

        # ---- Profile window resolution ----
        prof_start = (
            self.profile_start_step if self.profile_start_step is not None else 0
        )
        prof_end = (
            prof_start + self.profile_num_steps
            if self.profile_num_steps is not None
            else self.n_timed
        )
        prof_end = min(prof_end, self.n_timed)
        profile_enabled = profiler is not None and profiler.name != "none"

        # ---- Timed loop ----
        latencies_ms: list[float] = []
        try:
            for i in range(self.n_timed):
                if profile_enabled and i == prof_start:
                    profiler.start()

                latency_ms = _time_one_step(step_callable, use_cuda=use_cuda)
                latencies_ms.append(latency_ms)

                if profile_enabled and profiler.is_active and i + 1 == prof_end:
                    profiler.stop()
        finally:
            # Safety stop if the loop bailed out before reaching prof_end.
            if profile_enabled and profiler.is_active:
                profiler.stop()

        return _build_result(
            run_name=self.run_name,
            bench_name=self.bench_name,
            spec_name=spec.name,
            batch_size=batch_size,
            n_warmup=self.n_warmup,
            n_timed=self.n_timed,
            latencies_ms=latencies_ms,
            extras=extras,
        )


# ============================================================================ #
# Helpers                                                                      #
# ============================================================================ #


def _infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _time_one_step(step_callable: Callable[[], Any], *, use_cuda: bool) -> float:
    """One step, measured. Returns latency in milliseconds.

    Wraps the step in ``event_scope("bench.step")`` so an active
    profiler sees a clean per-step boundary regardless of which named
    scopes the model emits inside.
    """
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with event_scope("bench.step"):
            step_callable()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)
    t0 = time.perf_counter()
    with event_scope("bench.step"):
        step_callable()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def _build_result(
    *,
    run_name: str,
    bench_name: str,
    spec_name: str,
    batch_size: int,
    n_warmup: int,
    n_timed: int,
    latencies_ms: list[float],
    extras: dict[str, Any],
) -> BenchResult:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean = float(arr.mean())
    median = float(np.median(arr))
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))
    stdev = float(statistics.stdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0
    mn = float(arr.min())
    mx = float(arr.max())
    # Throughput in samples/s = batch_size / mean_latency_seconds.
    throughput = batch_size / (mean / 1000.0) if mean > 0 else float("inf")
    merged_extras = {**extras, "spec_name": spec_name}
    return BenchResult(
        run_name=run_name,
        bench_name=bench_name,
        batch_size=batch_size,
        n_warmup=n_warmup,
        n_timed=n_timed,
        latency_ms_mean=mean,
        latency_ms_median=median,
        latency_ms_p50=p50,
        latency_ms_p90=p90,
        latency_ms_p99=p99,
        latency_ms_stdev=stdev,
        latency_ms_min=mn,
        latency_ms_max=mx,
        throughput_samples_per_s=throughput,
        extras=merged_extras,
    )


def _print_result(r: BenchResult) -> None:
    print(
        f"  latency: mean={r.latency_ms_mean:.3f}ms  "
        f"median={r.latency_ms_median:.3f}ms  "
        f"p90={r.latency_ms_p90:.3f}ms  p99={r.latency_ms_p99:.3f}ms  "
        f"stdev={r.latency_ms_stdev:.3f}ms  "
        f"min={r.latency_ms_min:.3f}ms  max={r.latency_ms_max:.3f}ms"
    )
    print(f"  throughput: {r.throughput_samples_per_s:.2f} samples/s")


def _print_summary_table(results: list[BenchResult]) -> None:
    if not results:
        return
    print("\n[bench] summary")
    header = (
        f"{'B':>4} | {'mean(ms)':>10} {'p50(ms)':>10} "
        f"{'p90(ms)':>10} {'p99(ms)':>10} | "
        f"{'throughput (sample/s)':>22}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.batch_size:>4d} | "
            f"{r.latency_ms_mean:>10.3f} {r.latency_ms_p50:>10.3f} "
            f"{r.latency_ms_p90:>10.3f} {r.latency_ms_p99:>10.3f} | "
            f"{r.throughput_samples_per_s:>22.2f}"
        )


def _append_jsonl(path: Path, results: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), default=str) + "\n")


# ============================================================================ #
# CLI                                                                          #
# ============================================================================ #


def add_bench_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add the common bench-runner flag group.

    Pair with :func:`bench_runner_kwargs_from_args` to extract the
    constructor kwargs for :class:`NBatchBenchRunner`. The profile
    flags from :func:`phyai.utils.profile.add_profile_cli_args` are
    independent — call both on the same parser.
    """
    group = parser.add_argument_group("bench")
    group.add_argument(
        "--run-name",
        type=str,
        default="default",
        help="Label for this run; recorded in every BenchResult and trace name.",
    )
    group.add_argument(
        "--bench-name",
        type=str,
        default="n_batch",
        help="Label for the benchmark family; recorded in every BenchResult.",
    )
    group.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Sequence of batch sizes to sweep (e.g. --batch-sizes 1 2 4 8).",
    )
    group.add_argument(
        "--n-warmup",
        type=int,
        default=5,
        help="Warmup steps before timing (and before any profile window).",
    )
    group.add_argument(
        "--n-timed",
        type=int,
        default=30,
        help="Timed steps to record per batch size.",
    )
    group.add_argument(
        "--result-file",
        type=Path,
        default=None,
        help="Optional path to append BenchResults as JSONL (one line per batch size).",
    )
    group.add_argument(
        "--profile-start-step",
        type=int,
        default=None,
        help="0-indexed step in the timed loop at which to start the profiler. "
        "Defaults to 0 (profile the whole timed window).",
    )
    group.add_argument(
        "--profile-num-steps",
        type=int,
        default=None,
        help="Number of timed steps to keep the profiler active for. "
        "Defaults to until the end of the timed loop.",
    )


def bench_runner_kwargs_from_args(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Lift the bench CLI namespace into :class:`NBatchBenchRunner` kwargs.

    Doesn't include ``setup_fn`` — the per-model script provides it.
    """
    return dict(
        batch_sizes=list(args.batch_sizes),
        n_warmup=args.n_warmup,
        n_timed=args.n_timed,
        profile_start_step=args.profile_start_step,
        profile_num_steps=args.profile_num_steps,
        run_name=args.run_name,
        bench_name=args.bench_name,
        result_file=args.result_file,
    )


__all__ = [
    "BenchResult",
    "BenchSpec",
    "NBatchBenchRunner",
    "add_bench_cli_args",
    "bench_runner_kwargs_from_args",
]
