#!/usr/bin/env python3
"""Load-test the Gateway RLinf HTTP endpoint."""
import argparse
import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any
import httpx
import msgpack
import numpy as np

TEST_MODE=("time","N-Request")

@dataclass
class Result:
    ok: bool
    elapsed_s: float
    error: str = ""
def pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(order="C"),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(type(value).__name__)
def unpack_numpy(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    marker = value.get(
        "__ndarray__",
        value.get(b"__ndarray__"),
    )
    if not marker:
        return value
    data = value.get(
        "data",
        value.get(b"data"),
    )
    dtype = np.dtype(
        value.get(
            "dtype",
            value.get(b"dtype"),
        )
    )
    shape = tuple(
        value.get(
            "shape",
            value.get(b"shape"),
        )
    )
    return np.frombuffer(
        data,
        dtype=dtype,
    ).reshape(shape)
def build_gateway_body(
    batch_size: int,
    height: int,
    width: int,
    state_dim: int,
    horizon: int,
) -> bytes:
    payload = {
        "observation": {
            "main_images": np.zeros(
                (batch_size, height, width, 3),
                dtype=np.uint8,
            ),
            "wrist_images": np.zeros(
                (batch_size, height, width, 3),
                dtype=np.uint8,
            ),
            "states": np.zeros(
                (batch_size, state_dim),
                dtype=np.float32,
            ),
            "task_descriptions": [
                "pick up the object"
            ]
            * batch_size,
        },
        "model_name": "pi05",
        "model": "pi05",
        "metadata": {
            "mode": "eval",
            "batch_size": batch_size,
            "stage_id": 0,
            "reset": False,
        },
        "requested_action_horizon": horizon,
    }
    return msgpack.packb(
        payload,
        default=pack_numpy,
        use_bin_type=True,
    )


def build_sglang_body(
    height: int,
    width: int,
    state_dim: int,
    horizon: int,
) -> bytes:
    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )
    payload = {
        "model": "pi05",
        "input": {
            "task": "pick up the object",
            "observation": {
                "images": {
                    "image": image.tolist(),
                    "image2": image.tolist(),
                },
                "state": np.zeros(
                    state_dim,
                    dtype=np.float32,
                ).tolist(),
            },
        },
        "parameters": {
            "action_horizon": horizon,
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
    return msgpack.packb(
        payload,
        use_bin_type=True,
    )


def validate_response(
    response: httpx.Response,
    target: str,
    batch_size: int,
    horizon: int,
) -> None:
    response.raise_for_status()
    decoded = msgpack.unpackb(
        response.content,
        raw=False,
        object_hook=unpack_numpy,
    )
    if target == "gateway":
        actions = (
            decoded.get("actions")
            if isinstance(decoded, dict)
            else None
        )
        if not isinstance(actions, np.ndarray):
            raise ValueError(
                "Gateway response does not contain NumPy actions"
            )
        if (
            actions.ndim != 3
            or actions.shape[0] != batch_size
            or actions.shape[1] != horizon
        ):
            raise ValueError(
                f"Gateway returned action shape {actions.shape}; "
                f"expected [{batch_size}, {horizon}, action_dim]"
            )
        return
    try:
        action = decoded["data"][0]["action"]
        values = action["values"]
        declared_shape = tuple(action["shape"])
    except (
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        raise ValueError(
            "SGLang response does not contain envelope action"
        ) from exc
    actual_shape = (
        tuple(values.shape)
        if isinstance(values, np.ndarray)
        else tuple(np.asarray(values).shape)
    )
    if actual_shape != declared_shape:
        raise ValueError(
            f"SGLang values shape {actual_shape} "
            f"does not match declared shape {declared_shape}"
        )
    if len(declared_shape) != 2:
        raise ValueError(
            f"SGLang action must be rank 2, got {declared_shape}"
        )
    if declared_shape[0] < horizon:
        raise ValueError(
            f"SGLang returned horizon {declared_shape[0]}, "
            f"requested at least {horizon}"
        )

async def send_one(
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    target: str,
    batch_size: int,
    horizon: int,
) -> Result:
    started = time.perf_counter()
    try:
        response = await client.post(
            url,
            content=body,
            headers={
                "content-type": "application/msgpack",
                "accept": "application/msgpack",
            },
        )
        validate_response(
            response,
            target,
            batch_size,
            horizon,
        )
        return Result(
            ok=True,
            elapsed_s=time.perf_counter() - started,
        )
    except Exception as exc:
        return Result(
            ok=False,
            elapsed_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
async def run_phase(
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    target: str,
    batch_size: int,
    horizon: int,
    concurrency: int,
    duration: float,
    request_num: int,
    collect: bool,
    mode: str,
) -> tuple[list[Result], float]:
    if mode not in TEST_MODE:
        raise ValueError(f"invalid mode:{mode}")
    started = time.perf_counter()
    
    pending: set[asyncio.Task[Result]] = set()
    results: list[Result] = []

    if mode=="time":
        deadline = started + duration
        while time.perf_counter() < deadline or pending:
            while (
                time.perf_counter() < deadline
                and len(pending) < concurrency
            ):
                pending.add(
                    asyncio.create_task(
                        send_one(
                            client,
                            url,
                            body,
                            target,
                            batch_size,
                            horizon,
                        )

                    )
                )
            if not pending:
                break
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            completed = await asyncio.gather(*done)
            if collect:
                results.extend(completed)
    elif mode=="N-Request":
        send_num: int=0
        while send_num < request_num or pending:
            while (
                send_num < request_num
                and len(pending) < concurrency
            ):
                pending.add(
                    asyncio.create_task(
                        send_one(
                            client,
                            url,
                            body,
                            target,
                            batch_size,
                            horizon,
                        )
                    )
                )
                send_num+=1
            if not pending:
                break
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            completed = await asyncio.gather(*done)
            if collect:
                results.extend(completed)
    return results, time.perf_counter() - started
def print_summary(
    results: list[Result],
    elapsed_s: float,
    batch_size: int,
) -> None:
    successes = [
        result for result in results if result.ok
    ]
    failures = [
        result for result in results if not result.ok
    ]
    latencies_ms = np.asarray(
        [
            result.elapsed_s * 1000.0
            for result in successes
        ],
        dtype=np.float64,
    )
    request_throughput = (
        len(successes) / elapsed_s
        if elapsed_s > 0
        else 0.0
    )
    sample_throughput = (
        len(successes) * batch_size / elapsed_s
        if elapsed_s > 0
        else 0.0
    )
    print()
    print("Measured results")
    print(f"  total time ms:        {elapsed_s * 1000:.3f}ms")
    print(f"  total time:           {elapsed_s:.3f}S")
    print(f"  total requests:       {len(results)}")
    print(f"  successful requests:  {len(successes)}")
    print(f"  failed requests:      {len(failures)}")
    print(
        "  request throughput:   "
        f"{request_throughput:.3f} requests/s"
    )
    print(
        "  sample throughput:    "
        f"{sample_throughput:.3f} samples/s"
    )
    if latencies_ms.size:
        print(
            "  latency avg:          "
            f"{np.mean(latencies_ms):.3f} ms"
        )
        print(
            "  latency p50:          "
            f"{np.percentile(latencies_ms, 50):.3f} ms"
        )
        print(
            "  latency p95:          "
            f"{np.percentile(latencies_ms, 95):.3f} ms"
        )
        print(
            "  latency p99:          "
            f"{np.percentile(latencies_ms, 99):.3f} ms"
        )
        print(
            "  latency min:          "
            f"{np.min(latencies_ms):.3f} ms"
        )
        print(
            "  latency max:          "
            f"{np.max(latencies_ms):.3f} ms"
        )
    if failures:
        print("  errors:")
        error_counts = Counter(
            result.error for result in failures
        )
        for error, count in error_counts.most_common(10):
            print(f"    {count}x {error}")
async def main(args: argparse.Namespace) -> None:
    if args.target == "sglang" and args.batch_size != 1:
        raise SystemExit(
            "--batch-size must be 1 when --target sglang"
        )
    if args.url is not None:
        base_url = args.url
    elif args.target == "gateway":
        base_url = args.gateway
    else:
        base_url = args.sglang
    url = (
        base_url.rstrip("/")
        + "/v1/actions/generations"
    )
    if args.target == "gateway":
        body = build_gateway_body(
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            state_dim=args.state_dim,
            horizon=args.horizon,
        )
    else:
        body = build_sglang_body(
            height=args.height,
            width=args.width,
            state_dim=args.state_dim,
            horizon=args.horizon,
        )
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    print(f"target: {args.target}")
    print(f"endpoint: {url}")
    print(f"payload bytes: {len(body)}")
    print(f"batch size: {args.batch_size}")
    print(f"concurrency: {args.concurrency}")
    print(f"warmup: {args.warmup}s")
    print(f"duration: {args.duration}s")
    async with httpx.AsyncClient(
        timeout=args.timeout,
        limits=limits,
    ) as client:
        if args.warmup > 0:
            print("warming up...")
            await run_phase(
                client=client,
                url=url,
                body=body,
                target=args.target,
                batch_size=args.batch_size,
                horizon=args.horizon,
                concurrency=args.concurrency,
                duration=args.warmup,
                request_num=args.request_num,
                collect=False,
                mode="time",
            )
        print("measuring...")
        results, elapsed_s = await run_phase(
            client=client,
            url=url,
            body=body,
            target=args.target,
            batch_size=args.batch_size,
            horizon=args.horizon,
            concurrency=args.concurrency,
            duration=args.duration,
            request_num=args.request_num,
            collect=True,
            mode=args.mode,
        )
    print_summary(
        results,
        elapsed_s,
        args.batch_size,
    )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--target",
        choices=("gateway", "sglang"),
        default="gateway",
        help="Target service to load-test",
    )
    parser.add_argument(
            "--mode",
            choices=TEST_MODE,
            default="time",
            help="Test Mode",
        )
    parser.add_argument(

        "--url",
        default=None,
        help="Override target base URL",
    )
    parser.add_argument(
        "--gateway",
        default="http://127.0.0.1:30000",
        help="Gateway base URL",
    )
    parser.add_argument(
        "--sglang",
        default="http://127.0.0.1:30001",
        help="SGLang base URL",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Measured duration in seconds",
    )
    parser.add_argument(
            "--request_num",
            type=int,
            default=1024,
            help="Measured request_num samples",
        )
    parser.add_argument(
        "--warmup",
        type=float,
        default=10.0,
        help="Warmup duration in seconds",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Maximum in-flight requests",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Gateway RLinf batch size",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="Image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=360,
        help="Image width",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=8,
        help="State dimension",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="Requested action horizon",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds",
    )
    args = parser.parse_args()
    if (
        args.duration <= 0
        or args.warmup < 0
        or args.concurrency <= 0
        or args.batch_size <= 0
        or args.height <= 0
        or args.width <= 0
        or args.state_dim <= 0
        or args.horizon <= 0
        or args.timeout <= 0
        or args.request_num <=0
    ):
        parser.error(
            "durations, concurrency, sizes, dimensions, "
            "horizon, and timeout must be positive; "
            "warmup must be non-negative"
        )
    return args

if __name__ == "__main__":
    asyncio.run(main(parse_args()))

