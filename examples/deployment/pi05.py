"""pi0.5 serving with checkpoint-owned pre/postprocessing pipelines.

The server receives named RGB arrays and raw state, and returns absolute action
chunks. Server YAML defines the interface; the checkpoint pipelines define the
math. Robot bindings and timing belong to the client.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
from threading import Lock, Event
from dataclasses import field, replace, dataclass
from collections.abc import Mapping

import numpy as np
from omegaconf import MISSING
from phyai_robot import FeatureSpec
from numpy.typing import NDArray
from phyai_robot.remote import PolicySpec
from phyai_robot.adapters import STATE_KEY, ACTIONS_KEY, IMAGE_PREFIX

from .configuration import repository_path


@dataclass
class Pi05PolicyConfig:
    """Configuration dataclass for Pi0.5 model weights and the PhyAI inference engine.

    Attributes:
        checkpoint: Local path or repo path to the exported Pi0.5 checkpoint directory.
            Must contain config.json, weights, tokenizer files and serialized processing pipelines.
        kernel_policy: Path to kernel selection YAML policy enforcing exact GELU implementations.
        seed: Random seed for the CPU diffusion initial Gaussian noise generator.
        num_inference_steps: Number of Euler integration steps during flow-matching denoising.
            Set to null to use the checkpoint's native default (e.g. 5 steps for Tianji, 10 for AgileX).
        use_cuda_graph: Enable CUDA graph capture and replay for minimized execution jitter.
        num_threads: Number of CPU worker threads allocated to the PhyAI engine runtime.
    """

    checkpoint: str = MISSING
    kernel_policy: str = "examples/deployment/configs/pi05_bf16.yaml"
    seed: int = 42
    num_inference_steps: int | None = None
    use_cuda_graph: bool = True
    num_threads: int = 16

    def validate(self) -> None:
        """Validate configuration paths and parameters without allocating GPU memory."""
        if self.num_inference_steps is not None and self.num_inference_steps < 1:
            raise ValueError("policy.num_inference_steps must be positive or null")
        if self.num_threads < 1:
            raise ValueError("policy.num_threads must be positive")
        if not 0 <= self.seed < 2**64:
            raise ValueError("policy.seed must be in [0, 2**64)")
        if not self.checkpoint.strip() or not repository_path(self.checkpoint).is_dir():
            raise ValueError("policy.checkpoint must name an existing directory")
        if not repository_path(self.kernel_policy).is_file():
            raise ValueError("policy.kernel_policy must name an existing YAML file")


@dataclass
class Pi05ProcessorConfig:
    """Serving interface and optional external factory; math lives in pipelines."""

    factory: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    camera_names: list[str] | None = None
    action_dim: int | None = None
    action_dtype: str = "float32"
    image_resize_backend: str = "pil"

    def validate(self) -> None:
        """Verify that processor configuration options are well-formed."""
        if self.factory is not None and (
            not self.factory.strip() or ":" not in self.factory
        ):
            raise ValueError(
                "processor.factory must be module:factory or file.py:factory"
            )
        if self.factory is None and self.options:
            raise ValueError("processor.options requires processor.factory")
        if {"checkpoint", "task", "config"}.intersection(self.options):
            raise ValueError(
                "processor.options cannot override checkpoint, task or config"
            )
        if self.action_dtype not in ("float32", "float64"):
            raise ValueError("processor.action_dtype must be float32 or float64")
        names = self.camera_names
        if names is not None and (
            not names or len(set(names)) != len(names) or not all(names)
        ):
            raise ValueError("processor.camera_names must be unique nonempty names")
        if self.action_dim is not None and self.action_dim < 1:
            raise ValueError("processor.action_dim must be positive or null")
        if self.image_resize_backend not in ("pil", "torch"):
            raise ValueError("processor.image_resize_backend must be pil or torch")


class Pi05Processing:
    """Adapt wire arrays to native PI05Processor pipelines and back."""

    def __init__(
        self, checkpoint: Path, *, task: str, config: Pi05ProcessorConfig
    ) -> None:
        import torch
        from phyai_utils_tools.models.pi05 import PI05Processor

        config.validate()
        if not task.strip():
            raise ValueError("task must be non-empty")
        if config.camera_names is None or config.action_dim is None:
            raise ValueError("Set processor.camera_names and processor.action_dim")
        self.task = task.strip()
        self._torch = torch
        self.camera_names = list(config.camera_names)
        self.action_dim = config.action_dim
        self.action_dtype = config.action_dtype
        self._processor = PI05Processor.from_pretrained(
            checkpoint,
            tokenizer_name=str(checkpoint),
            num_images=len(self.camera_names),
            image_resize_backend=config.image_resize_backend,
            normalize_pixels=True,
            action_dim=self.action_dim,
            params_dtype=torch.float32,
        )
        self.image_size = int(self._processor.image_size)

    def input_schema(self) -> dict[str, FeatureSpec]:
        image = FeatureSpec((self.image_size, self.image_size, 3), "uint8")
        return {
            **{IMAGE_PREFIX + name: image for name in self.camera_names},
            STATE_KEY: FeatureSpec((self.action_dim,), "float32"),
        }

    def preprocess(self, inputs: Mapping[str, NDArray[Any]]) -> Any:
        # Copy read-only wire buffers before handing them to torch.
        images = [
            self._torch.from_numpy(np.array(inputs[IMAGE_PREFIX + name]))
            .permute(2, 0, 1)
            .unsqueeze(0)
            for name in self.camera_names
        ]
        state = self._torch.from_numpy(np.array(inputs[STATE_KEY])).unsqueeze(0)
        return self._processor.preprocess(
            {"images": images, "task": self.task, "state": state}
        )

    def postprocess(self, normalized: Any, state: NDArray[np.float32]) -> NDArray[Any]:
        # The pipeline owns slicing, inverse normalization and action decoding.
        actions = self._processor.postprocess(
            {"action": normalized, "state": np.array(state)}
        )
        values = actions.detach().cpu().numpy().astype(self.action_dtype)[0]
        if not np.isfinite(values).all():
            raise ValueError("model produced non-finite actions")
        return values


class Pi05Policy:
    """Manages the resident PhyAI Pi0.5 inference engine and reproducible noise stream."""

    def __init__(self, config: Pi05PolicyConfig, processing: Pi05Processing) -> None:
        """Load Pi0.5 weights and initialize the PhyAI execution engine."""
        import torch
        from phyai.utils import load_config
        from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
        from phyai.kernel.config import KernelConfig
        from phyai.models.pi05.main_pi05 import PI05Args
        from phyai.models.pi05.scheduler_pi05 import PI05Request
        from phyai.models.pi05.configuration_pi05 import PI05Config
        from phyai import Engine, EngineArgs

        checkpoint = repository_path(config.checkpoint)
        model_config = load_config(checkpoint, PI05Config)
        if config.num_inference_steps is not None:
            # Override flow-matching Euler steps before engine initialization.
            model_config = replace(
                model_config, num_inference_steps=config.num_inference_steps
            )

        self.processing = processing
        self.num_inference_steps = int(model_config.num_inference_steps)
        self.horizon = int(model_config.chunk_size)

        self._torch = torch
        self._request_type = PI05Request
        self._max_action_dim = int(model_config.max_action_dim)
        self._seed = config.seed
        self._generator = torch.Generator().manual_seed(config.seed)

        # Initialize the high-performance PhyAI CUDA inference engine.
        self._engine = Engine(
            EngineArgs(
                plugin="pi05",
                plugin_args=PI05Args(
                    checkpoint_dir=checkpoint, config=model_config, max_batch_size=1
                ),
                config=EngineConfig(
                    device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                    runtime=RuntimeConfig(
                        use_cuda_graph=config.use_cuda_graph,
                        num_threads=config.num_threads,
                    ),
                    kernel=KernelConfig(
                        config_path=str(repository_path(config.kernel_policy))
                    ),
                ),
            )
        )
        self._lock = Lock()
        self._close_requested = Event()
        self._closed = False

    def spec(self, metadata: Mapping[str, Any]) -> PolicySpec:
        """Assemble the complete PolicySpec describing model inputs, outputs, and metadata."""
        processing = self.processing
        return PolicySpec(
            inputs=processing.input_schema(),
            outputs={
                ACTIONS_KEY: FeatureSpec(
                    (self.horizon, processing.action_dim), processing.action_dtype
                )
            },
            metadata=dict(
                metadata,
                horizon=self.horizon,
                image_size=processing.image_size,
                camera_names=processing.camera_names,
                action_dim=processing.action_dim,
                num_inference_steps=self.num_inference_steps,
            ),
        )

    def predict(self, inputs: Mapping[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
        """Synchronously execute preprocessing, Euler denoising, and postprocessing.

        Must be called while holding self._lock. Generates deterministic Gaussian noise
        for the full chunk horizon using the CPU generator stream.
        """
        torch = self._torch
        with self._lock:
            if self._close_requested.is_set():
                raise RuntimeError("Pi05Policy is closed")
            try:
                processed = self.processing.preprocess(inputs)
                # Sample initial Gaussian noise (shape: (1, horizon, max_action_dim)).
                noise = torch.randn(
                    1,
                    self.horizon,
                    self._max_action_dim,
                    generator=self._generator,
                    dtype=torch.float32,
                )
                with torch.inference_mode():
                    normalized = (
                        self._engine.step(
                            self._request_type(
                                pixel_values=processed.pixel_values,
                                input_ids=processed.input_ids,
                                lang_lens=processed.lang_lens,
                                noise=noise,
                            )
                        )
                        .detach()
                        .float()
                        .cpu()
                    )
                actions = self.processing.postprocess(normalized, inputs[STATE_KEY])
                return {ACTIONS_KEY: actions}
            finally:
                if self._close_requested.is_set():
                    self._close_engine()

    def reset_run(self) -> None:
        """Reset the CPU initial noise generator seed at the beginning of a client run."""
        with self._lock:
            if self._close_requested.is_set():
                raise RuntimeError("Pi05Policy is closed")
            self._generator.manual_seed(self._seed)

    def _close_engine(self) -> None:
        """Release underlying PhyAI engine resources once."""
        if not self._closed:
            self._engine.close()
            self._closed = True

    def close(self) -> None:
        """Request engine shutdown, safely releasing GPU resources."""
        self._close_requested.set()
        if self._lock.acquire(blocking=False):
            try:
                self._close_engine()
            finally:
                self._lock.release()
