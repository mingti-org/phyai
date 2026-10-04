"""Serve a local LeRobot PI0.5 checkpoint through the gateway inference API."""

import argparse
import json
import math
import signal
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import torch

from phyai_gateway.bindings import model_inference_pb2, model_inference_pb2_grpc
from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_pi05 import PI05Request
from phyai.utils import get_logger, load_config
from phyai.utils.logging import configure_logging
from phyai_utils_tools.models.pi05 import PI05Processor
from phyai_utils_tools.tokenizer import get_tokenizer


logger = get_logger(__name__)


def remap_lerobot_weight(key: str) -> str:
    return key.removeprefix("model.")


class ModelRegistryReporter:
    def __init__(
        self,
        registry_address,
        endpoint,
        model_name,
        registration_retry_seconds,
    ):
        self.endpoint = endpoint
        self.model_name = model_name
        self.registration_retry_seconds = registration_retry_seconds
        self.channel = grpc.insecure_channel(registry_address)
        self.stub = model_inference_pb2_grpc.ModelRegistryStub(self.channel)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="model-registry-reporter",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=4)
        self.channel.close()

    def _register(self):
        response = self.stub.Register(
            model_inference_pb2.RegisterRequest(
                endpoint=self.endpoint,
                model_name=self.model_name,
            ),
            timeout=3,
        )
        if not response.accepted:
            logger.warning(
                "Model server registration rejected: %s",
                response.message,
            )
            return None
        if not response.server_id:
            logger.warning("Gateway accepted registration without server_id")
            return None
        heartbeat_interval = max(
            1,
            response.heartbeat_interval_seconds,
        )
        logger.info(
            "Registered with gateway: server_id=%s, heartbeat_interval=%ss",
            response.server_id,
            heartbeat_interval,
        )
        return response.server_id, heartbeat_interval

    def _run(self):
        registration = None
        while not self.stop_event.is_set():
            if registration is None:
                try:
                    registration = self._register()
                except grpc.RpcError as error:
                    logger.warning(
                        "Model server registration failed: %s: %s",
                        error.code(),
                        error.details(),
                    )
                if registration is None:
                    self.stop_event.wait(self.registration_retry_seconds)
                    continue
            server_id, heartbeat_interval = registration
            if self.stop_event.wait(heartbeat_interval):
                return
            try:
                response = self.stub.Heartbeat(
                    model_inference_pb2.HeartbeatRequest(
                        server_id=server_id,
                    ),
                    timeout=3,
                )
            except grpc.RpcError as error:
                logger.warning(
                    "Model server heartbeat failed: %s: %s",
                    error.code(),
                    error.details(),
                )
                continue
            if response.registered:
                continue
            logger.warning(
                "Gateway no longer recognizes server_id=%s: %s; registering again",
                server_id,
                response.message,
            )
            registration = None


class PI05Runtime:
    def __init__(
        self,
        checkpoint_dir,
        tokenizer_dir,
        image_names,
        image_shape,
        state_dim,
        action_dim,
        max_batch_size,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for PI0.5 inference")

        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.image_names = image_names
        self.image_shape = image_shape
        self.state_shape = (state_dim,)
        self.action_dim = action_dim
        self.max_batch_size = max_batch_size
        self.config = load_config(checkpoint_dir, PI05Config)
        if self.image_shape[2] != self.config.vision.num_channels:
            raise ValueError(
                "image channels must match PI0.5 vision config: "
                f"{self.config.vision.num_channels}"
            )
        if self.action_dim > self.config.max_action_dim:
            raise ValueError(
                f"action dimension exceeds PI0.5 maximum: {self.config.max_action_dim}"
            )
        self.engine = None

        logger.info("Loading PI0.5 checkpoint")
        tokenizer = get_tokenizer(
            str(tokenizer_dir),
            local_files_only=True,
        )
        self.processor = PI05Processor.from_pretrained(
            checkpoint_dir,
            tokenizer=tokenizer,
            tokenizer_name=str(tokenizer_dir),
            image_size=self.config.vision.image_size,
            num_channels=self.config.vision.num_channels,
            num_images=len(self.image_names),
            action_dim=self.action_dim,
            normalize_pixels=True,
            device=self.device,
            params_dtype=self.dtype,
            local_files_only=True,
        )
        self.engine = Engine(
            EngineArgs(
                plugin="pi05",
                plugin_args=PI05Args(
                    checkpoint_dir=checkpoint_dir,
                    max_batch_size=self.max_batch_size,
                    weight_remap=remap_lerobot_weight,
                    inputs_image_shape=[
                        list(self.image_shape) for _ in self.image_names
                    ],
                ),
                config=EngineConfig(
                    device=DeviceConfig(target="cuda", params_dtype=self.dtype),
                    runtime=RuntimeConfig(use_cuda_graph=True),
                ),
            )
        )
        try:
            self._warm_up()
        except BaseException:
            self.close()
            raise
        logger.info("PI0.5 runtime is ready")

    def _make_request(self, images, state, instructions):
        processed = self.processor.preprocess(
            {
                "images": images,
                "task": instructions,
                "state": state,
            }
        )
        return PI05Request(
            pixel_values=processed.pixel_values.to(
                device=self.device, dtype=self.dtype
            ),
            input_ids=processed.input_ids.to(self.device),
            lang_lens=processed.lang_lens.to(self.device),
        )

    def _warm_up(self):
        logger.info("Warming up PI0.5 inference path")
        images = [
            torch.zeros(
                1,
                self.image_shape[2],
                self.image_shape[0],
                self.image_shape[1],
                dtype=torch.float32,
            )
            for _ in self.image_names
        ]
        state = torch.zeros(1, self.state_shape[0], dtype=torch.float32)
        request = self._make_request(images, state, ["warm up"])
        actions = self.engine.step(request)
        self.processor.postprocess(actions[..., : self.action_dim])
        torch.cuda.synchronize()

    def infer(self, images, state, instructions, horizon):
        request = self._make_request(images, state, instructions)
        torch.cuda.synchronize()
        start_ns = time.perf_counter_ns()
        actions = self.engine.step(request)
        torch.cuda.synchronize()
        inference_time_us = (time.perf_counter_ns() - start_ns) // 1000

        actions = self.processor.postprocess(actions[..., : self.action_dim])
        batch_size = state.shape[0]
        if (
            actions.ndim != 3
            or actions.shape[0] != batch_size
            or actions.shape[2] != self.action_dim
        ):
            raise RuntimeError(f"unexpected PI0.5 action shape: {tuple(actions.shape)}")
        if actions.shape[1] < horizon:
            raise RuntimeError(
                f"PI0.5 returned horizon {actions.shape[1]}, requested {horizon}"
            )

        actions = actions[:, :horizon].to(dtype=torch.float32).contiguous().cpu()
        if not torch.isfinite(actions).all():
            raise RuntimeError("PI0.5 returned non-finite actions")
        return actions, inference_time_us

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None


class ModelInferenceServicer(model_inference_pb2_grpc.ModelInferenceServicer):
    def __init__(self, runtime):
        self.runtime = runtime
        self.image_names = runtime.image_names
        self.image_shape = runtime.image_shape
        self.state_shape = runtime.state_shape
        self.max_batch_size = runtime.max_batch_size

    def Infer(self, request, context):
        images, state, instructions, is_batch = self._validate_and_decode(
            request,
            context,
            self.runtime.config.chunk_size,
        )

        if not context.is_active():
            context.abort(grpc.StatusCode.CANCELLED, "request was cancelled")

        try:
            actions, inference_time_us = self.runtime.infer(
                images,
                state,
                instructions,
                request.requested_action_horizon or self.runtime.config.chunk_size,
            )
        except torch.cuda.OutOfMemoryError:
            logger.exception("CUDA out of memory for request %s", request.request_id)
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "PI0.5 inference ran out of GPU memory",
            )
        except Exception:
            logger.exception(
                "PI0.5 inference failed for request %s", request.request_id
            )
            context.abort(grpc.StatusCode.INTERNAL, "PI0.5 inference failed")

        if not is_batch:
            actions = actions[0]
        action_array = actions.numpy().astype("<f4", copy=False)
        return model_inference_pb2.InferenceResponse(
            request_id=request.request_id,
            actions=model_inference_pb2.Tensor(
                data=action_array.tobytes(order="C"),
                shape=list(action_array.shape),
                dtype=model_inference_pb2.DATA_TYPE_FLOAT32,
            ),
            inference_time_us=inference_time_us,
        )

    def _validate_and_decode(self, request, context, max_action_horizon):
        if not request.request_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id is required")
        if len(request.images) != len(self.image_names):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"exactly {len(self.image_names)} images are required",
            )
        if request.requested_action_horizon > max_action_horizon:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"requested_action_horizon must be between 0 and {max_action_horizon}",
            )

        images_by_name = {}
        batch_size = None
        is_batch = None
        for image in request.images:
            if image.name not in self.image_names:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"unknown image name: {image.name!r}",
                )
            if image.name in images_by_name:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"duplicate image name: {image.name!r}",
                )
            if image.dtype != model_inference_pb2.DATA_TYPE_UINT8:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"image {image.name!r} dtype must be UINT8",
                )
            if image.encoding != model_inference_pb2.IMAGE_ENCODING_RAW:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"image {image.name!r} encoding must be RAW",
                )
            if image.layout != model_inference_pb2.IMAGE_LAYOUT_HWC:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"image {image.name!r} layout must be HWC",
                )

            image_shape = tuple(image.shape)
            if image_shape == self.image_shape:
                image_is_batch = False
                image_batch_size = 1
            elif len(image_shape) == 4 and image_shape[1:] == self.image_shape:
                image_is_batch = True
                image_batch_size = image_shape[0]
                if not 1 <= image_batch_size <= self.max_batch_size:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"image batch size must be between 1 and {self.max_batch_size}",
                    )
            else:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"image {image.name!r} shape must be {list(self.image_shape)} "
                    f"or [B, {self.image_shape[0]}, {self.image_shape[1]}, "
                    f"{self.image_shape[2]}]",
                )

            if is_batch is None:
                is_batch = image_is_batch
                batch_size = image_batch_size
            elif is_batch != image_is_batch or batch_size != image_batch_size:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "all images must use the same batch shape",
                )
            if len(image.data) != math.prod(image_shape):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"image {image.name!r} data length does not match its shape",
                )
            images_by_name[image.name] = image

        if is_batch:
            try:
                extensions = json.loads(request.extensions_json)
            except (TypeError, ValueError):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "extensions_json must be valid JSON for batch inference",
                )
            if not isinstance(extensions, dict):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "extensions_json must be a JSON object for batch inference",
                )
            instructions = extensions.get("instructions")
            if (
                not isinstance(instructions, list)
                or len(instructions) != batch_size
                or any(
                    not isinstance(instruction, str) or not instruction.strip()
                    for instruction in instructions
                )
            ):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"extensions_json.instructions must contain exactly {batch_size} "
                    "non-empty strings",
                )
        else:
            if not request.instruction.strip():
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "instruction is required",
                )
            instructions = [request.instruction]

        robot_state = request.robot_state
        if robot_state.dtype != model_inference_pb2.DATA_TYPE_FLOAT32:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "robot_state dtype must be FLOAT32",
            )
        expected_state_shape = (
            (batch_size, *self.state_shape) if is_batch else self.state_shape
        )
        if tuple(robot_state.shape) != expected_state_shape:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"robot_state shape must be {list(expected_state_shape)}",
            )
        if len(robot_state.data) != math.prod(expected_state_shape) * 4:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "robot_state data length does not match its shape",
            )

        image_tensors = []
        for image_name in self.image_names:
            image = images_by_name[image_name]
            image_array = (
                np.frombuffer(
                    image.data,
                    dtype=np.uint8,
                )
                .reshape(tuple(image.shape))
                .copy()
            )
            if not is_batch:
                image_array = np.expand_dims(image_array, axis=0)
            image_tensors.append(
                torch.from_numpy(image_array)
                .permute(0, 3, 1, 2)
                .to(dtype=torch.float32)
                .div_(255.0)
            )

        state_array = np.frombuffer(robot_state.data, dtype="<f4").copy()
        if not np.isfinite(state_array).all():
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "robot_state must contain finite values",
            )
        state_tensor = torch.from_numpy(state_array).reshape(
            batch_size, *self.state_shape
        )
        return image_tensors, state_tensor, instructions, is_batch


def serve(
    checkpoint_dir,
    tokenizer_dir,
    model_name,
    gateway_registry,
    registration_retry_seconds,
    image_names,
    image_shape,
    state_dim,
    action_dim,
    max_batch_size,
    port=None,
    listen=None,
    advertised_endpoint=None,
):
    if port is None and listen is None:
        port = 50063
    if port is not None:
        listen = f"[::]:{port}"
        advertised_endpoint = advertised_endpoint or f"127.0.0.1:{port}"
    elif advertised_endpoint is None:
        raise ValueError("--advertised-endpoint is required when --listen is used")

    configure_logging()
    runtime = PI05Runtime(
        checkpoint_dir=checkpoint_dir,
        tokenizer_dir=tokenizer_dir,
        image_names=image_names,
        image_shape=image_shape,
        state_dim=state_dim,
        action_dim=action_dim,
        max_batch_size=max_batch_size,
    )
    try:
        # One worker owns the processor and GPU from decode through postprocess.
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            server = grpc.server(
                executor,
                options=[
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                    ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ],
            )
            registry_reporter = ModelRegistryReporter(
                registry_address=gateway_registry,
                endpoint=advertised_endpoint,
                model_name=model_name,
                registration_retry_seconds=registration_retry_seconds,
            )
            try:
                model_inference_pb2_grpc.add_ModelInferenceServicer_to_server(
                    ModelInferenceServicer(runtime), server
                )
                if server.add_insecure_port(listen) == 0:
                    raise RuntimeError(f"failed to bind gRPC server to {listen}")
                server.start()
                registry_reporter.start()
                logger.info("PI0.5 inference server listening on %s", listen)
                server.wait_for_termination()
            except KeyboardInterrupt:
                logger.info("Stopping PI0.5 inference server")
            finally:
                registry_reporter.stop()
                server.stop(grace=2).wait()
        # Executor shutdown waits for any RPC that outlasted the gRPC grace period.
    finally:
        runtime.close()


def main():
    parser = argparse.ArgumentParser(description="PI0.5 gRPC Model Server")
    listen_group = parser.add_mutually_exclusive_group()
    listen_group.add_argument(
        "--port",
        type=int,
        default=None,
        help="Listen port; defaults to 50063",
    )
    listen_group.add_argument(
        "--listen",
        default=None,
        help="Full gRPC listen address, for example [::]:50063",
    )
    parser.add_argument(
        "--advertised-endpoint",
        default=None,
        help="Endpoint registered in Gateway, for example 127.0.0.1:50063",
    )
    parser.add_argument(
        "--gateway-registry",
        default="127.0.0.1:50111",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Local LeRobot checkpoint with model and processor files",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="Local PaliGemma tokenizer directory",
    )
    parser.add_argument(
        "--model-name",
        default="pi05",
    )
    parser.add_argument(
        "--registration-retry-seconds",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--image-names",
        nargs="+",
        default=("agentview", "robot0_eye_in_hand"),
    )
    parser.add_argument(
        "--image-shape",
        type=int,
        nargs=3,
        metavar=("HEIGHT", "WIDTH", "CHANNELS"),
        default=(360, 360, 3),
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    if not Path(args.checkpoint_dir).is_dir():
        parser.error(f"--checkpoint-dir must be a directory: {args.checkpoint_dir}")
    if not Path(args.tokenizer_dir).is_dir():
        parser.error(f"--tokenizer-dir must be a directory: {args.tokenizer_dir}")
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not math.isfinite(args.registration_retry_seconds) or (
        args.registration_retry_seconds <= 0
    ):
        parser.error("--registration-retry-seconds must be positive")
    if any(not name.strip() for name in args.image_names):
        parser.error("--image-names must contain non-empty names")
    if len(set(args.image_names)) != len(args.image_names):
        parser.error("--image-names must not contain duplicates")
    if any(dimension <= 0 for dimension in args.image_shape):
        parser.error("--image-shape dimensions must be positive")
    if args.state_dim <= 0:
        parser.error("--state-dim must be positive")
    if args.action_dim <= 0:
        parser.error("--action-dim must be positive")
    if args.max_batch_size <= 0:
        parser.error("--max-batch-size must be positive")
    if not args.gateway_registry.strip():
        parser.error("--gateway-registry must be non-empty")
    if not args.model_name.strip():
        parser.error("--model-name must be non-empty")
    if args.listen is not None and args.advertised_endpoint is None:
        parser.error("--advertised-endpoint is required when --listen is used")
    if args.advertised_endpoint is not None and not args.advertised_endpoint.strip():
        parser.error("--advertised-endpoint must be non-empty")

    signal.signal(signal.SIGTERM, signal.default_int_handler)
    serve(
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_dir=args.tokenizer_dir,
        model_name=args.model_name,
        gateway_registry=args.gateway_registry,
        registration_retry_seconds=args.registration_retry_seconds,
        image_names=tuple(args.image_names),
        image_shape=tuple(args.image_shape),
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        max_batch_size=args.max_batch_size,
        port=args.port,
        listen=args.listen,
        advertised_endpoint=args.advertised_endpoint,
    )


if __name__ == "__main__":
    main()
