#!/usr/bin/env python3
"""Measure HTTP inference throughput and latency for the gateway or SGLang."""

import argparse
import asyncio
import math
import time
from collections import Counter
from dataclasses import dataclass

import httpx
import msgpack
import numpy as np

from phyai_gateway.adapters.rlinf import pack_numpy, unpack_numpy


@dataclass
class Result:
    elapsed_s: float
    error: str = ""


def build_gateway_body(args: argparse.Namespace) -> bytes:
    payload = {
        "observation": {
            "main_images": np.zeros(
                (args.batch_size, args.height, args.width, 3), dtype=np.uint8
            ),
            "wrist_images": np.zeros(
                (args.batch_size, args.height, args.width, 3), dtype=np.uint8
            ),
            "states": np.zeros((args.batch_size, args.state_dim), dtype=np.float32),
            "task_descriptions": ["pick up the object"] * args.batch_size,
        },
        "model_name": args.model,
        "metadata": {
            "mode": "eval",
            "batch_size": args.batch_size,
            "stage_id": 0,
            "reset": False,
        },
        "requested_action_horizon": args.horizon,
    }
    return msgpack.packb(payload, default=pack_numpy, use_bin_type=True)


def build_sglang_body(args: argparse.Namespace) -> bytes:
    image = np.zeros((args.height, args.width, 3), dtype=np.uint8).tolist()
    payload = {
        "model": args.model,
        "input": {
            "task": "pick up the object",
            "observation": {
                "images": {"image": image, "image2": image},
                "state": [0.0] * args.state_dim,
            },
        },
        "parameters": {
            "action_horizon": args.horizon,
            "action_dim": 32,
            "num_inference_steps": 10,
        },
        "runtime": {
            "response_format": "envelope",
            "output_format": "numpy",
            "return_timing": True,
            "cuda_graph": False,
        },
    }
    return msgpack.packb(payload, use_bin_type=True)


def validate_response(
    response: httpx.Response, target: str, batch_size: int, horizon: int
) -> None:
    response.raise_for_status()
    decoded = msgpack.unpackb(response.content, raw=False, object_hook=unpack_numpy)
    if target == "gateway":
        actions = decoded.get("actions") if isinstance(decoded, dict) else None
        if not isinstance(actions, np.ndarray):
            raise ValueError("gateway response has no NumPy actions")
        if (
            actions.ndim != 3
            or actions.shape[:2] != (batch_size, horizon)
            or actions.shape[2] == 0
        ):
            raise ValueError(f"unexpected gateway action shape: {actions.shape}")
    else:
        try:
            action = decoded["data"][0]["action"]
            actions = np.asarray(action["values"])
            declared_shape = tuple(action["shape"])
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("SGLang response has no action envelope") from error
        if (
            actions.shape != declared_shape
            or actions.ndim != 2
            or actions.shape[0] < horizon
            or actions.shape[1] == 0
        ):
            raise ValueError(f"unexpected SGLang action shape: {actions.shape}")
    if actions.dtype.kind != "f" or not np.isfinite(actions).all():
        raise ValueError("actions must contain finite floating-point values")


async def send_one(
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    target: str,
    batch_size: int,
    horizon: int,
) -> Result:
    started = time.perf_counter()
    error_message = ""
    try:
        response = await client.post(
            url,
            content=body,
            headers={
                "content-type": "application/msgpack",
                "accept": "application/msgpack",
            },
        )
        validate_response(response, target, batch_size, horizon)
    except (httpx.HTTPError, msgpack.UnpackException, ValueError, TypeError) as error:
        error_message = f"{type(error).__name__}: {error}"
    return Result(time.perf_counter() - started, error_message)


async def run_phase(
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    args: argparse.Namespace,
    *,
    duration: float,
    requests: int | None = None,
    collect: bool = True,
) -> tuple[list[Result], float]:
    started = time.perf_counter()
    deadline = started + duration
    remaining = requests
    results = []

    async def worker():
        nonlocal remaining
        while (remaining is None and time.perf_counter() < deadline) or (
            remaining is not None and remaining > 0
        ):
            if remaining is not None:
                remaining -= 1
            result = await send_one(
                client, url, body, args.target, args.batch_size, args.horizon
            )
            if collect:
                results.append(result)

    await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    return results, time.perf_counter() - started


def print_summary(results: list[Result], elapsed_s: float, batch_size: int) -> None:
    successes = [result for result in results if not result.error]
    errors = Counter(result.error for result in results if result.error)
    latencies = np.asarray([result.elapsed_s * 1000 for result in successes])
    throughput = len(successes) / elapsed_s if elapsed_s else 0.0
    print(
        f"Requests: {len(results)}, succeeded: {len(successes)}, failed: {sum(errors.values())}"
    )
    print(f"Elapsed: {elapsed_s:.3f} s")
    print(
        f"Throughput: {throughput:.3f} requests/s, {throughput * batch_size:.3f} samples/s"
    )
    if latencies.size:
        p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
        print(
            f"Latency (ms): mean={latencies.mean():.3f}, "
            f"p50={p50:.3f}, p95={p95:.3f}, p99={p99:.3f}, "
            f"min={latencies.min():.3f}, max={latencies.max():.3f}"
        )
    for error, count in errors.most_common(10):
        print(f"{count}x {error}")


async def main(args: argparse.Namespace) -> int:
    default_url = (
        "http://127.0.0.1:30000"
        if args.target == "gateway"
        else "http://127.0.0.1:30001"
    )
    url = (args.url or default_url).rstrip("/") + "/v1/actions/generations"
    body = (
        build_gateway_body(args)
        if args.target == "gateway"
        else build_sglang_body(args)
    )
    print(f"Target: {url}, payload: {len(body)} bytes, concurrency: {args.concurrency}")
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        if args.warmup:
            await run_phase(
                client, url, body, args, duration=args.warmup, collect=False
            )
        results, elapsed = await run_phase(
            client, url, body, args, duration=args.duration, requests=args.requests
        )
    print_summary(results, elapsed, args.batch_size)
    return int(not results or any(result.error for result in results))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("gateway", "sglang"), default="gateway")
    parser.add_argument("--url", help="Service base URL")
    parser.add_argument("--model", default="pi05")
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--duration", type=float, default=60.0, help="Measured seconds")
    limit.add_argument(
        "--requests", type=int, help="Number of requests instead of a duration"
    )
    parser.add_argument("--warmup", type=float, default=10.0, help="Warmup seconds")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Per-request seconds"
    )
    args = parser.parse_args()
    for name in (
        "duration",
        "timeout",
        "concurrency",
        "batch_size",
        "height",
        "width",
        "state_dim",
        "horizon",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.warmup) or args.warmup < 0:
        parser.error("--warmup must be non-negative and finite")
    if args.requests is not None and args.requests <= 0:
        parser.error("--requests must be positive")
    if args.target == "sglang" and args.batch_size != 1:
        parser.error("--batch-size must be 1 for SGLang")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
