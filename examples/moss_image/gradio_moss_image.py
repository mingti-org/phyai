"""Gradio client and server adapter for MOSS-Image."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import grpc
import torch
from PIL import Image

from phyai.models.moss_image import MOSSImageRequest
from phyai.models.moss_image.model_runner_moss_image import (
    normalize_moss_image_graph_buckets,
)
from phyai.server.backends.tensor import TensorBackend, decode_tensor, encode_tensor
from phyai.server.lifecycle import EngineUnavailableError
from phyai.utils import get_logger
from phyai_gateway.bindings import model_inference_pb2 as pb
from phyai_gateway.bindings import model_inference_pb2_grpc as rpc


MAX_MESSAGE_BYTES = 100 * 1024 * 1024
CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
]
logger = get_logger(__name__)
PARAMETERS = {
    "prompt",
    "negative_prompt",
    "height",
    "width",
    "num_inference_steps",
    "guidance_scale",
    "num_images_per_prompt",
    "seed",
    "max_sequence_length",
    "uncond_skip_layers",
}


def make_request(parameters: dict) -> MOSSImageRequest:
    if not isinstance(parameters, dict):
        raise ValueError("Generation parameters must be a JSON object")
    if unknown := parameters.keys() - PARAMETERS:
        raise ValueError(f"Unknown generation parameters: {', '.join(sorted(unknown))}")
    if (
        not isinstance(parameters.get("prompt"), str)
        or not parameters["prompt"].strip()
    ):
        raise ValueError("Enter a prompt")
    if not isinstance(parameters.get("negative_prompt", ""), str):
        raise ValueError("Negative prompt must be text")
    seed = parameters.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 4294967295")
    layers = parameters.get("uncond_skip_layers", [22])
    if not isinstance(layers, (list, tuple)) or any(
        type(layer) is not int or not 0 <= layer < 42 for layer in layers
    ):
        raise ValueError("Unconditional skip layers must be integers between 0 and 41")
    return MOSSImageRequest(**parameters, output_type="pt")


class MOSSImageBackend(TensorBackend):
    """Send generated images as uint8 tensors with shape [batch, height, width, 3]."""

    def infer(self, request, context):
        try:
            if request.inputs:
                raise ValueError("This text-to-image endpoint accepts parameters only")
            extensions = json.loads(request.extensions_json or "{}")
            if not isinstance(extensions, dict):
                raise ValueError("extensions_json must be a JSON object")
            payload = make_request(extensions.get("parameters", {}))
        except (ValueError, TypeError, OverflowError) as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

        start = time.perf_counter_ns()
        try:
            output = self.engine.step(payload)
            pixels = output.images.clamp(0, 1).mul(255).round().to(torch.uint8)
            images = encode_tensor(pixels.permute(0, 2, 3, 1))
        except RuntimeError as error:
            unavailable = isinstance(error, EngineUnavailableError)
            if unavailable:
                self._healthy = False
            if (
                isinstance(error, torch.cuda.OutOfMemoryError)
                or "CUDA out of memory" in str(error)
                or "CUBLAS_STATUS_ALLOC_FAILED" in str(error)
            ):
                logger.exception("MOSS-Image GPU memory allocation failed")
                context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    "GPU memory allocation failed. Restart the server on a GPU with more free memory using --devices. "
                    "For large requests, reduce the image size or image count.",
                )
            if unavailable:
                context.abort(
                    grpc.StatusCode.UNAVAILABLE, "Model engine is unavailable"
                )
            raise
        return pb.InferenceResponse(
            request_id=request.request_id,
            outputs={"images": images},
            inference_time_us=(time.perf_counter_ns() - start) // 1000,
        )


def check_health(address: str, model_name: str, timeout: float = 5) -> bool:
    with grpc.insecure_channel(address, options=CHANNEL_OPTIONS) as channel:
        response = rpc.ModelInferenceStub(channel).CheckHealth(
            pb.HealthRequest(model_name=model_name), timeout=timeout
        )
        return response.healthy


def generate_images(
    address: str, model_name: str, parameters: dict, timeout: float
) -> tuple[list[Image.Image], float]:
    payload = make_request(parameters)
    request = pb.InferenceRequest(
        request_id=uuid.uuid4().hex,
        model_name=model_name,
        extensions_json=json.dumps({"parameters": parameters}, allow_nan=False),
    )
    with grpc.insecure_channel(address, options=CHANNEL_OPTIONS) as channel:
        response = rpc.ModelInferenceStub(channel).Infer(request, timeout=timeout)
    if response.request_id != request.request_id or "images" not in response.outputs:
        raise ValueError("Server returned an invalid image response")
    pixels = decode_tensor(response.outputs["images"])
    expected = (payload.num_images_per_prompt, payload.height, payload.width, 3)
    if pixels.dtype != torch.uint8 or tuple(pixels.shape) != expected:
        raise ValueError("Server must return uint8 RGB images with the requested size")
    return [Image.fromarray(image.numpy()) for image in pixels], (
        response.inference_time_us / 1_000_000
    )


def rpc_error_message(error: grpc.RpcError) -> str:
    if error.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
        return error.details() or "The model server ran out of resources."
    if error.code() == grpc.StatusCode.UNAVAILABLE:
        return "Cannot reach the model server. Check its address and wait for it to be ready."
    if error.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        return "Generation timed out. The server may still be running it; increase --timeout before retrying."
    if error.code() == grpc.StatusCode.NOT_FOUND:
        return "The server does not serve this model name. Check --model-name."
    return f"{error.code().name}: {error.details()}"


def build_demo(address: str, model_name: str, timeout: float, concurrency: int = 1):
    import gradio as gr

    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("Concurrency must be a positive integer")

    def health_status():
        try:
            return (
                "Ready" if check_health(address, model_name) else "Model is not ready"
            )
        except grpc.RpcError as error:
            return rpc_error_message(error)

    def generate(
        prompt, negative, width, height, steps, guidance, seed, count, length, skip
    ):
        try:
            seed = secrets.randbelow(2**32) if seed == -1 else int(seed)
            parameters = dict(
                prompt=prompt,
                negative_prompt=negative,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=guidance,
                seed=seed,
                num_images_per_prompt=int(count),
                max_sequence_length=int(length),
                uncond_skip_layers=[
                    int(value) for value in skip.replace(",", " ").split()
                ],
            )
            images, elapsed = generate_images(address, model_name, parameters, timeout)
        except grpc.RpcError as error:
            raise gr.Error(rpc_error_message(error)) from error
        except (ValueError, TypeError, OverflowError) as error:
            raise gr.Error(str(error)) from error
        return (
            images,
            f"Seed: {seed} · {len(images)} image(s) · Server time: {elapsed:.1f}s",
        )

    with gr.Blocks(
        title="MOSS-Image", analytics_enabled=False, delete_cache=(3600, 86400)
    ) as demo:
        gr.Markdown("# MOSS-Image\nGenerate images with your local PhyAI server.")
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=5,
                    placeholder="Describe the image you want to create…",
                )
                negative = gr.Textbox(label="Negative prompt", lines=2)
                with gr.Row():
                    width = gr.Slider(256, 2048, value=1024, step=16, label="Width")
                    height = gr.Slider(256, 2048, value=1024, step=16, label="Height")
                with gr.Row():
                    steps = gr.Slider(1, 100, value=50, step=1, label="Steps")
                    guidance = gr.Slider(
                        0, 10, value=3, step=0.1, label="Guidance (CFG)"
                    )
                with gr.Row():
                    seed = gr.Number(
                        value=42,
                        precision=0,
                        minimum=-1,
                        maximum=2**32 - 1,
                        label="Seed (-1 for random)",
                    )
                    count = gr.Slider(1, 4, value=1, step=1, label="Images")
                with gr.Accordion("Advanced", open=False):
                    length = gr.Slider(
                        32, 4096, value=4096, step=32, label="Maximum text tokens"
                    )
                    skip = gr.Textbox(
                        value="22",
                        label="Unconditional skip layers",
                        info="Comma-separated layer indices. Leave empty to disable.",
                    )
                submit = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    label="Images",
                    columns=2,
                    height=640,
                    format="png",
                    buttons=["download", "download_all", "fullscreen"],
                    interactive=False,
                )
                status = gr.Textbox(label="Generation details", interactive=False)
        with gr.Accordion("Server connection", open=False):
            gr.Textbox(
                value=f"{address} / {model_name}",
                label="Server / model",
                interactive=False,
            )
            health = gr.Textbox(label="Status", interactive=False)
            refresh = gr.Button("Check connection")
        submit.click(
            generate,
            inputs=[
                prompt,
                negative,
                width,
                height,
                steps,
                guidance,
                seed,
                count,
                length,
                skip,
            ],
            outputs=[gallery, status],
            concurrency_limit=concurrency,
            api_name="generate",
        )
        refresh.click(health_status, outputs=health, queue=False, api_name="health")
        demo.load(health_status, outputs=health, queue=False)
    return demo.queue(
        max_size=max(8, concurrency * 2), default_concurrency_limit=concurrency
    )


def serve_model(args: argparse.Namespace) -> None:
    from phyai import DeploymentConfig, EngineArgs
    from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
    from phyai.models.moss_image.main_moss_image import MOSSImageArgs
    from phyai.server import WorkerSupervisorConfig
    from phyai.server.config import ServerConfig, ServerOptions
    from phyai.server.serving import serve

    serve(
        ServerConfig(
            server=ServerOptions(
                model_name=args.model_name,
                host=args.host,
                port=args.port,
                workers=args.replicas,
                max_message_bytes=MAX_MESSAGE_BYTES,
            ),
            engine_args=EngineArgs(
                plugin="moss_image",
                plugin_args=MOSSImageArgs(
                    checkpoint_dir=args.checkpoint,
                    cpu_offload=args.cpu_offload,
                    torch_compile=args.torch_compile,
                    vae_slicing=args.vae_slicing,
                    graph_cache_size=args.graph_cache_size,
                    graph_warmup_buckets=args.graph_bucket,
                ),
                config=EngineConfig(
                    device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                    runtime=RuntimeConfig(use_cuda_graph=not args.no_cuda_graph),
                ),
            ),
            deployment=DeploymentConfig(
                devices=args.devices,
                replica_count=args.replicas,
                process_config=WorkerSupervisorConfig(startup_timeout_s=1800),
            ),
            adapter_factory=f"{__name__}:MOSSImageBackend",
            adapter_args={},
        )
    )


def add_model_arguments(server: argparse.ArgumentParser) -> None:
    server.add_argument("--checkpoint", type=Path, required=True)
    server.add_argument("--model-name", default="moss_image")
    server.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload text encoder and VAE; CUDA graphs keep the transformer on GPU and need more VRAM",
    )
    server.add_argument("--torch-compile", action="store_true")
    server.add_argument("--vae-slicing", action="store_true")
    server.add_argument(
        "--no-cuda-graph", action="store_true", help="Disable CUDA graph capture"
    )
    server.add_argument("--graph-cache-size", type=int, default=4)
    server.add_argument(
        "--graph-bucket",
        nargs=3,
        type=int,
        action="append",
        default=[],
        metavar=("HEIGHT", "WIDTH", "TEXT_TOKENS"),
        help="Prewarm an image size; text tokens round up to 64, 128, ..., 4096. "
        "Graphs are reused per sample at any batch size (two cache entries per unique bucket)",
    )
    server.add_argument(
        "--devices",
        nargs="+",
        type=int,
        help="Logical GPU indices within CUDA_VISIBLE_DEVICES, one per replica",
    )
    server.add_argument(
        "--replicas",
        type=int,
        help="Concurrent model replicas (default: number of --devices, or 1)",
    )


def add_ui_arguments(ui: argparse.ArgumentParser, concurrency: int | None = 1) -> None:
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument(
        "--concurrency",
        type=int,
        default=concurrency,
        help="Concurrent generation requests (launch defaults to the replica count; ui defaults to 1)",
    )
    ui.add_argument(
        "--timeout", type=float, default=1800, help="Generation timeout in seconds"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    combined = commands.add_parser(
        "launch", help="Start the model server and UI together"
    )
    add_model_arguments(combined)
    add_ui_arguments(combined, concurrency=None)
    combined.add_argument("--server-port", type=int, default=50063)
    combined.add_argument(
        "--startup-timeout",
        type=float,
        default=1800,
        help="Seconds to wait for model loading and graph warmup before opening the UI",
    )
    server = commands.add_parser("serve", help="Start the local PhyAI model server")
    add_model_arguments(server)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=50063)
    ui = commands.add_parser(
        "ui", help="Open a Gradio client for an existing model server"
    )
    ui.add_argument(
        "--server-address", default="127.0.0.1:50063", help="gRPC host:port"
    )
    ui.add_argument("--model-name", default="moss_image")
    add_ui_arguments(ui)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.command in ("serve", "launch"):
        args.devices = tuple(args.devices or ())
        if len(set(args.devices)) != len(args.devices) or any(
            device < 0 for device in args.devices
        ):
            parser.error("--devices must contain unique non-negative GPU indices")
        if args.replicas is None:
            args.replicas = len(args.devices) or 1
        if args.replicas < 1:
            parser.error("--replicas must be positive")
        if args.devices and len(args.devices) != args.replicas:
            parser.error("--devices must specify exactly one GPU per replica")
        if args.graph_cache_size < 1:
            parser.error("--graph-cache-size must be positive")
        if args.no_cuda_graph and args.graph_bucket:
            parser.error("--graph-bucket requires CUDA graphs")
        for height, width, tokens in args.graph_bucket:
            if (
                height <= 0
                or width <= 0
                or height % 16
                or width % 16
                or not 1 <= tokens <= 4096
            ):
                parser.error(
                    "--graph-bucket requires positive image sizes divisible by 16 and 1..4096 text tokens"
                )
        args.graph_bucket = normalize_moss_image_graph_buckets(args.graph_bucket)
        if args.graph_cache_size < 2 * len(args.graph_bucket):
            parser.error(
                "--graph-cache-size must hold two entries per unique --graph-bucket"
            )
    if args.command in ("ui", "launch"):
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("--timeout must be finite and positive")
        if args.concurrency is None:
            args.concurrency = args.replicas
        if args.concurrency < 1:
            parser.error("--concurrency must be positive")
    if args.command == "launch":
        if not 1 <= args.server_port <= 65535:
            parser.error("--server-port must be between 1 and 65535")
        if args.port == args.server_port:
            parser.error("--port and --server-port must differ")
        if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
            parser.error("--startup-timeout must be finite and positive")
    return args


def server_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "serve",
        "--checkpoint",
        str(args.checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.server_port),
        "--model-name",
        args.model_name,
        "--replicas",
        str(args.replicas),
        "--graph-cache-size",
        str(args.graph_cache_size),
    ]
    for name in ("cpu_offload", "torch_compile", "vae_slicing", "no_cuda_graph"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    if args.devices:
        command.extend(["--devices", *(str(device) for device in args.devices)])
    for bucket in args.graph_bucket:
        command.extend(["--graph-bucket", *(str(value) for value in bucket)])
    return command


def ensure_port_available(host: str, port: int) -> None:
    try:
        for family, kind, protocol, _, address in socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        ):
            with socket.socket(family, kind, protocol) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(address)
    except OSError as error:
        raise RuntimeError(f"Cannot use {host}:{port}: {error}") from error


def wait_for_model(
    process: subprocess.Popen, address: str, model_name: str, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"Model server exited with code {process.returncode}; see its logs above"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Model server did not become ready within {timeout:g} seconds"
            )
        try:
            if check_health(address, model_name, timeout=min(5, remaining)):
                if process.poll() is None:
                    return
                continue
        except grpc.RpcError:
            pass
        time.sleep(min(0.25, max(0, deadline - time.monotonic())))


def stop_server(process: subprocess.Popen, timeout: float = 15) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    # The server owns a session, so forced cleanup also reaches its GPU workers.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


def launch_app(args: argparse.Namespace) -> None:
    ensure_port_available("127.0.0.1", args.server_port)
    ensure_port_available(args.host, args.port)
    address = f"127.0.0.1:{args.server_port}"
    process = None
    demo = None
    previous_sigterm = signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        print(
            f"Starting MOSS-Image server at {address}; waiting for model readiness…",
            flush=True,
        )
        process = subprocess.Popen(
            server_command(args), start_new_session=os.name == "posix"
        )
        wait_for_model(process, address, args.model_name, args.startup_timeout)
        print(
            "Model ready. Starting the UI; press Ctrl+C to stop both services.",
            flush=True,
        )
        demo = build_demo(address, args.model_name, args.timeout, args.concurrency)
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
            prevent_thread_lock=True,
        )
        while demo.is_running:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Model server exited with code {process.returncode}; see its logs above"
                )
            time.sleep(0.25)
    finally:
        try:
            if demo is not None:
                demo.close()
        finally:
            try:
                if process is not None:
                    stop_server(process)
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> None:
    args = parse_args()
    if args.command == "serve":
        serve_model(args)
    elif args.command == "launch":
        try:
            launch_app(args)
        except KeyboardInterrupt:
            print("MOSS-Image server and UI stopped.", flush=True)
    else:
        build_demo(
            args.server_address, args.model_name, args.timeout, args.concurrency
        ).launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
