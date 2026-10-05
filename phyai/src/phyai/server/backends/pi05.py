"""PI0.5 preprocessing and inference for the gateway policy fields."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import replace

import grpc
import numpy as np
import torch

from phyai.engine import Engine, EngineArgs, EngineUnavailableError
from phyai.engine_config import EngineConfig
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_pi05 import PI05Request
from phyai.server.deployment import DeploymentConfig
from phyai.utils import get_logger, load_config
from phyai_gateway.bindings import model_inference_pb2
from phyai_utils_tools.models.pi05 import PI05Processor
from phyai_utils_tools.tokenizer import get_tokenizer

logger = get_logger(__name__)


def remap_lerobot_weight(key: str) -> str:
    return key.removeprefix("model.")


class PI05Backend:
    def __init__(
        self,
        engine_args: EngineArgs,
        deployment: DeploymentConfig | None,
        *,
        tokenizer_dir: str,
        image_names: list[str],
        state_dim: int,
        action_dim: int,
    ):
        args = engine_args.plugin_args
        if engine_args.plugin != "pi05" or not isinstance(args, PI05Args):
            raise ValueError("PI05Backend requires plugin: pi05")
        if (
            not args.checkpoint_dir
            or not isinstance(tokenizer_dir, str)
            or not tokenizer_dir.strip()
        ):
            raise ValueError("PI05Backend requires checkpoint_dir and tokenizer_dir")
        for name, value in (
            ("state_dim", state_dim),
            ("action_dim", action_dim),
            ("max_batch_size", args.max_batch_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(image_names, (list, tuple))
            or not image_names
            or any(
                not isinstance(name, str) or not name.strip() for name in image_names
            )
            or len(set(image_names)) != len(image_names)
        ):
            raise ValueError("image_names must contain unique, nonempty camera names")
        config = args.config or load_config(args.checkpoint_dir, PI05Config)
        image_shapes = args.inputs_image_shape
        if image_shapes is None:
            image_shapes = [
                [
                    config.vision.image_size,
                    config.vision.image_size,
                    config.vision.num_channels,
                ]
            ] * 3
        if len(image_shapes) != len(image_names) or any(
            len(shape) != 3 or any(type(size) is not int or size < 1 for size in shape)
            for shape in image_shapes
        ):
            raise ValueError(
                "inputs_image_shape must provide one positive [H, W, C] shape per camera"
            )
        if any(shape != image_shapes[0] for shape in image_shapes):
            raise ValueError(
                "PI05Backend currently requires the same input shape for each camera"
            )
        args = replace(
            args,
            config=config,
            weight_remap=remap_lerobot_weight
            if args.weight_remap is None
            else args.weight_remap,
        )
        self._lock = threading.Lock()
        self.runtime = PI05Runtime(
            engine_args=replace(engine_args, plugin_args=args),
            deployment=deployment,
            tokenizer_dir=tokenizer_dir,
            image_names=image_names,
            image_shape=image_shapes[0],
            state_dim=state_dim,
            action_dim=action_dim,
        )
        self.adapter = PI05RequestAdapter(self.runtime)

    @property
    def healthy(self):
        return self.runtime.healthy

    def infer(self, request, context):
        # The processor stores per-request state used by postprocess.
        with self._lock:
            return self.adapter.infer(request, context)

    def close(self):
        self.runtime.close()


class PI05Runtime:
    def __init__(
        self,
        engine_args: EngineArgs,
        deployment: DeploymentConfig | None,
        *,
        tokenizer_dir,
        image_names,
        image_shape,
        state_dim,
        action_dim,
    ):
        self._ready = False
        self.engine = None
        checkpoint_dir = engine_args.plugin_args.checkpoint_dir
        resolved_config = EngineConfig.from_env(base=engine_args.config)
        self.device = torch.device(resolved_config.device.target)
        if self.device.type != "cuda":
            raise ValueError("PI0.5 preprocessing requires a CUDA device")
        self.dtype = resolved_config.device.params_dtype
        self.image_names = tuple(image_names)
        self.image_shape = tuple(image_shape)
        self.state_shape = (state_dim,)
        self.action_dim = action_dim
        self.max_batch_size = engine_args.plugin_args.max_batch_size
        self.config = engine_args.plugin_args.config or load_config(
            checkpoint_dir, PI05Config
        )
        if self.image_shape[2] != self.config.vision.num_channels:
            raise ValueError(
                "image channels must match PI0.5 vision config: "
                f"{self.config.vision.num_channels}"
            )
        if self.action_dim > self.config.max_action_dim:
            raise ValueError(
                f"action dimension exceeds PI0.5 maximum: {self.config.max_action_dim}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for PI0.5 inference")
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
        self.engine = Engine(engine_args, deployment=deployment)
        try:
            self.engine.setup()
            self._warm_up()
        except BaseException:
            self.close()
            raise
        self._ready = True
        logger.info("PI0.5 runtime is ready")

    @property
    def healthy(self):
        return self._ready and self.engine is not None

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
        torch.cuda.synchronize(self.device)

    def infer(self, images, state, instructions, horizon):
        request = self._make_request(images, state, instructions)
        torch.cuda.synchronize(self.device)
        start_ns = time.perf_counter_ns()
        try:
            actions = self.engine.step(request)
        except EngineUnavailableError:
            self._ready = False
            raise
        torch.cuda.synchronize(self.device)
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
        self._ready = False
        if self.engine is not None:
            self.engine.close()
            self.engine = None


class PI05RequestAdapter:
    def __init__(self, runtime):
        self.runtime = runtime
        self.image_names = runtime.image_names
        self.image_shape = runtime.image_shape
        self.state_shape = runtime.state_shape
        self.max_batch_size = runtime.max_batch_size

    def infer(self, request, context):
        if not self.runtime.healthy:
            context.abort(grpc.StatusCode.UNAVAILABLE, "PI0.5 runtime is unavailable")
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
        except EngineUnavailableError:
            context.abort(grpc.StatusCode.UNAVAILABLE, "PI0.5 runtime is unavailable")
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
