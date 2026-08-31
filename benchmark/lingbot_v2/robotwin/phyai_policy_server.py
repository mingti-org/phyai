"""Serve PHYAI LingBot V2 through the official RoboTwin WebSocket contract."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torchvision.transforms.v2 import Resize

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.lingbot_v2 import (
    LingBotV2Args,
    LingBotV2Request,
    LingBotVLA2Config,
)
from phyai.utils import load_config
from phyai_utils_tools.models.lingbot_v2 import (
    RoboTwinLingBotV2Adapter,
    canonical_robotwin_stats,
    load_robotwin_stats,
)
from phyai_utils_tools.models.lingbot_v2.processor_lingbotv2 import (
    LingBotV2Processor,
)

from benchmark.lingbot_v2.robotwin.msgpack_numpy import Packer, unpackb

LOGGER = logging.getLogger("phyai.lingbot_v2.robotwin")

DIAGNOSTIC_NOISE_KEY = "_lingbot_diagnostic_noise"

# LingBot V2 was trained with fixed feature slots from lingbotvla_cli.yaml:
# arm.position=14, end.position=14, effector.position=2. RoboTwin provides
# 12 arm joints and two effectors, so its active model indices are 0:12 and
# 28:30 rather than one contiguous 14-D prefix.
ROBOTWIN_CANONICAL_ARM_DIM = 12
ROBOTWIN_CANONICAL_EFFECTOR_DIM = 2
ROBOTWIN_MODEL_EFFECTOR_OFFSET = 28
ROBOTWIN_MODEL_EFFECTOR_END = 30
ROBOTWIN_MODEL_FEATURE_DIM = ROBOTWIN_MODEL_EFFECTOR_END


def str_to_bool(value: str | bool) -> bool:
    """Parse the boolean spellings accepted by the deployment CLI."""

    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"yes", "true", "t", "1"}:
        return True
    if normalized in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}.")


def required_path(value: str | None, *, name: str) -> Path:
    """Resolve a required path argument and fail with a useful message."""

    if not value:
        raise ValueError(f"{name} is required (argument or environment variable).")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch for repeatable policy requests."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_lingbot_v2_config(checkpoint: Path) -> LingBotVLA2Config:
    """Load the same merged model/train YAML used by Official deployment."""

    training_config_path = checkpoint.parent.parent.parent / "lingbotvla_cli.yaml"
    if not training_config_path.is_file():
        LOGGER.warning(
            "training config not found at %s; falling back to checkpoint config.json",
            training_config_path,
        )
        return load_config(checkpoint, LingBotVLA2Config)

    with training_config_path.open(encoding="utf-8") as stream:
        training_config = yaml.safe_load(stream)
    if not isinstance(training_config, Mapping):
        raise TypeError(f"training config must be a mapping: {training_config_path}")
    model_config = training_config.get("model")
    train_config = training_config.get("train")
    if not isinstance(model_config, Mapping) or not isinstance(train_config, Mapping):
        raise TypeError(
            "training config must contain mapping-valued 'model' and 'train' "
            f"sections: {training_config_path}"
        )
    merged_config = dict(model_config)
    merged_config.update(train_config)
    config = LingBotVLA2Config.from_dict(merged_config)
    LOGGER.info("loaded LingBot V2 architecture from %s", training_config_path)
    return config


def scatter_robotwin_state_to_model_slots(
    state: torch.Tensor,
    *,
    max_state_dim: int,
) -> torch.Tensor:
    """Scatter normalized canonical RoboTwin state into training-time slots."""

    canonical_dim = ROBOTWIN_CANONICAL_ARM_DIM + ROBOTWIN_CANONICAL_EFFECTOR_DIM
    if state.ndim != 2 or state.shape[-1] != canonical_dim:
        raise ValueError(
            "normalized RoboTwin state must have shape "
            f"(B,{canonical_dim}), got {tuple(state.shape)}."
        )
    if max_state_dim < ROBOTWIN_MODEL_FEATURE_DIM:
        raise ValueError(
            f"max_state_dim={max_state_dim} is smaller than the RoboTwin "
            f"training layout width {ROBOTWIN_MODEL_FEATURE_DIM}."
        )
    model_state = state.new_zeros((state.shape[0], max_state_dim))
    model_state[..., :ROBOTWIN_CANONICAL_ARM_DIM] = state[
        ..., :ROBOTWIN_CANONICAL_ARM_DIM
    ]
    model_state[
        ...,
        ROBOTWIN_MODEL_EFFECTOR_OFFSET:ROBOTWIN_MODEL_EFFECTOR_END,
    ] = state[..., ROBOTWIN_CANONICAL_ARM_DIM:]
    return model_state


def gather_robotwin_action_from_model_slots(action: torch.Tensor) -> torch.Tensor:
    """Gather active RoboTwin action slots before inverse normalization."""

    if action.ndim != 3 or action.shape[-1] < ROBOTWIN_MODEL_FEATURE_DIM:
        raise ValueError(
            "model action must have shape (B,T,D) with "
            f"D>={ROBOTWIN_MODEL_FEATURE_DIM}, got {tuple(action.shape)}."
        )
    return torch.cat(
        (
            action[..., :ROBOTWIN_CANONICAL_ARM_DIM],
            action[
                ...,
                ROBOTWIN_MODEL_EFFECTOR_OFFSET:ROBOTWIN_MODEL_EFFECTOR_END,
            ],
        ),
        dim=-1,
    )


class LingBotV2RoboTwinPolicy:
    """In-process PHYAI policy with the same input/output boundary as official."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        processor_path: str,
        stats_path: Path,
        device: str,
        use_length: int,
        image_size: int,
        max_patches_per_image: int,
        patch_embed_backend: str,
        linear_kernel: str,
        use_cuda_graph: bool,
        seed: int,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for LingBot V2 RoboTwin inference.")
        if use_length <= 0:
            raise ValueError("use_length must be positive.")
        if image_size <= 0 or max_patches_per_image <= 0:
            raise ValueError("image and patch capacities must be positive.")

        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        set_seed(seed)

        self.config = load_lingbot_v2_config(checkpoint)
        if use_length > self.config.chunk_size:
            raise ValueError(
                f"use_length={use_length} exceeds chunk_size={self.config.chunk_size}."
            )
        stats = canonical_robotwin_stats(load_robotwin_stats(stats_path))
        self.adapter = RoboTwinLingBotV2Adapter(use_length=use_length)
        self.resize = Resize((image_size, image_size))
        self.processor = LingBotV2Processor(
            processor_name=processor_path,
            num_images=3,
            num_channels=self.config.vision.in_channels,
            patch_vector_dim=self.config.vision.patch_vector_dim,
            max_patches_per_image=max_patches_per_image,
            tokenizer_max_length=self.config.tokenizer_max_length,
            max_state_dim=self.config.max_state_dim,
            action_dim=14,
            dataset_stats=stats,
            normalization_eps=1e-6,
            device=self.device,
            params_dtype=torch.bfloat16,
        )
        max_vision_tokens = (
            max_patches_per_image // self.config.vision.spatial_merge_unit
        )
        self.engine = Engine(
            EngineArgs(
                plugin="lingbot_v2",
                plugin_args=LingBotV2Args(
                    checkpoint_dir=checkpoint,
                    config=self.config,
                    max_batch_size=1,
                    num_images=3,
                    max_vision_tokens_per_image=max_vision_tokens,
                    weight_strict=True,
                    vision_patch_embed_backend=patch_embed_backend,
                ),
                config=EngineConfig(
                    device=DeviceConfig(
                        target=str(self.device),
                        params_dtype=torch.bfloat16,
                    ),
                    runtime=RuntimeConfig(
                        use_cuda_graph=use_cuda_graph,
                        force_linear_kernel=linear_kernel,
                    ),
                ),
            )
        )
        self.metadata = {
            "model": "phyai-lingbot-v2",
            "robot": "robotwin",
            "chunk_size": use_length,
            "action_dim": 14,
            "normalization": "official bounds_99_woclip (eps=1e-6)",
        }

    def close(self) -> None:
        """Release the in-process PHYAI engine."""

        self.engine.close()

    def reset(self, robo_name: str) -> None:
        """Reset the engine after validating the requested RoboTwin mapping."""

        if robo_name not in {"robotwin", "robotwin_clean_and_aug"}:
            raise ValueError(
                "this policy server only supports the RoboTwin mapping, got "
                f"{robo_name!r}."
            )

    def _resize_images(self, images: list[np.ndarray]) -> list[torch.Tensor]:
        resized: list[torch.Tensor] = []
        for image in images:
            tensor = torch.as_tensor(image).permute(2, 0, 1).contiguous()
            resized.append(self.resize(tensor.to(dtype=torch.float32)))
        return resized

    @torch.inference_mode()
    def _infer_one(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        direct_observation = dict(observation)
        diagnostic_noise = direct_observation.pop(DIAGNOSTIC_NOISE_KEY, None)
        prepared = self.adapter.prepare_observation(direct_observation)
        prepared["images"] = self._resize_images(prepared["images"])
        prepared["state"] = torch.as_tensor(
            prepared["state"],
            dtype=torch.float32,
            device=self.device,
        )
        if diagnostic_noise is not None:
            noise = torch.as_tensor(diagnostic_noise)
            if noise.ndim == 2:
                noise = noise.unsqueeze(0)
            expected_shape = (
                1,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            if tuple(noise.shape) != expected_shape:
                raise ValueError(
                    f"{DIAGNOSTIC_NOISE_KEY} must have shape "
                    f"{expected_shape} or {expected_shape[1:]}, got "
                    f"{tuple(noise.shape)}."
                )
            prepared["noise"] = noise
        processed = self.processor.preprocess(prepared)
        canonical_dim = ROBOTWIN_CANONICAL_ARM_DIM + ROBOTWIN_CANONICAL_EFFECTOR_DIM
        processed.state = scatter_robotwin_state_to_model_slots(
            processed.state[..., :canonical_dim],
            max_state_dim=self.config.max_state_dim,
        )
        raw_action = self.engine.step(LingBotV2Request(**vars(processed)))
        canonical_action = self.processor.postprocess(
            gather_robotwin_action_from_model_slots(raw_action)
        )
        return self.adapter.format_action_chunk(canonical_action)

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Handle one legacy request and return an unnormalized action chunk."""

        if observation.get("reset"):
            self.reset(str(observation.get("robo_name", "robotwin")))
            return {"action": None}
        if "batch" in observation:
            raise ValueError(
                "the RoboTwin launcher uses one environment per server; batched "
                "WebSocket observations are not supported by this adapter."
            )
        return self._infer_one(observation)


class RoboTwinWebSocketServer:
    """Serve the PHYAI policy over the legacy MessagePack WebSocket protocol."""

    def __init__(self, policy: LingBotV2RoboTwinPolicy, *, host: str, port: int):
        self.policy = policy
        self.host = host
        self.port = port

    async def _handler(self, websocket: Any) -> None:
        from websockets.exceptions import ConnectionClosed

        LOGGER.info("connection opened: %s", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self.policy.metadata))
        previous_total: float | None = None
        while True:
            try:
                start = time.monotonic()
                message = await websocket.recv()
                observation = unpackb(message)
                infer_start = time.monotonic()
                response = self.policy.infer(observation)
                infer_ms = (time.monotonic() - infer_start) * 1000.0
                response["server_timing"] = {"infer_ms": infer_ms}
                if previous_total is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total * 1000.0
                await websocket.send(packer.pack(response))
                previous_total = time.monotonic() - start
            except ConnectionClosed:
                LOGGER.info("connection closed: %s", websocket.remote_address)
                break
            except Exception:
                LOGGER.exception("WebSocket policy request failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="policy inference failed")
                raise

    async def run(self) -> None:
        """Bind the configured WebSocket endpoint and serve until cancelled."""

        from websockets.asyncio.server import serve

        async with serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ) as server:
            LOGGER.info("serving ws://%s:%d", self.host, self.port)
            await server.serve_forever()

    def serve_forever(self) -> None:
        """Run the asynchronous server and close policy resources on exit."""

        try:
            asyncio.run(self.run())
        finally:
            self.policy.close()


def parse_args() -> argparse.Namespace:
    """Parse model, processor, normalization, and server CLI options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        "--checkpoint",
        dest="checkpoint",
        default=os.getenv("LINGBOT_CHECKPOINT"),
    )
    parser.add_argument(
        "--processor",
        default=os.getenv("LINGBOT_PROCESSOR"),
    )
    parser.add_argument(
        "--stats_json",
        "--stats-json",
        dest="stats_json",
        default=os.getenv("LINGBOT_ROBOTWIN_STATS"),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_length", type=int, default=50)
    parser.add_argument("--chunk_ret", type=str_to_bool, default=True)
    parser.add_argument("--use_compile", type=str_to_bool, default=False)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-patches-per-image", type=int, default=256)
    parser.add_argument(
        "--patch-embed-backend",
        choices=("conv3d", "gemm"),
        default=os.getenv("LINGBOT_PATCH_EMBED_BACKEND", "gemm"),
    )
    parser.add_argument(
        "--linear-kernel",
        choices=("torch", "flashinfer"),
        default=os.getenv("LINGBOT_LINEAR_KERNEL", "torch"),
    )
    parser.add_argument("--use-cuda-graph", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Construct the policy and serve RoboTwin requests until shutdown."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if not args.chunk_ret:
        raise ValueError("RoboTwin evaluation requires --chunk_ret true.")
    if args.use_compile:
        raise ValueError(
            "PHYAI uses its CUDA Graph runtime; --use_compile is not supported."
        )
    checkpoint = required_path(args.checkpoint, name="checkpoint")
    stats_path = required_path(args.stats_json, name="RoboTwin stats")
    processor_path = required_path(args.processor, name="Qwen processor")
    policy = LingBotV2RoboTwinPolicy(
        checkpoint=checkpoint,
        processor_path=str(processor_path),
        stats_path=stats_path,
        device=args.device,
        use_length=args.use_length,
        image_size=args.image_size,
        max_patches_per_image=args.max_patches_per_image,
        patch_embed_backend=args.patch_embed_backend,
        linear_kernel=args.linear_kernel,
        use_cuda_graph=args.use_cuda_graph,
        seed=args.seed,
    )
    RoboTwinWebSocketServer(policy, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
