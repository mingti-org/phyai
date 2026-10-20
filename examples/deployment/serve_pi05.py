"""Resident PhyAI Pi0.5 model inference server: ZMQ RPC -> Engine -> ActionChunk.

Architecture & Decoupling:
    `serve_pi05.py` hosts a resident model policy server. It is completely embodiment-agnostic
    and does not import any robot drivers, physical schemas, or ROS 2 dependencies.
    - Input: Receives preprocessed model arrays ('image.<camera>', 'state') over ZMQ RPC.
    - Execution: Performs flow-matching Euler denoising using the GPU-accelerated PhyAI Engine
      with deterministic CPU noise sampling and optional CUDA graph capture.
    - Output: Restores relative joint deltas to absolute coordinates and returns action
      matrices of shape (action_horizon, action_dim).
    - Lifecycle: Keeps the model weights resident across multiple sequential robot episodes.
      Automatically frees idle run reservations via lease timeout recovery.

Usage Examples:
    # Tianji (Marvin Pro):
    python -m examples.deployment.serve_pi05 \\
        --config examples/deployment/configs/pi05_tianji.yaml \\
        policy.checkpoint=/path/to/tianji_checkpoint

    # AgileX (Dual Piper):
    python -m examples.deployment.serve_pi05 \\
        --config examples/deployment/configs/pi05_agilex.yaml \\
        policy.checkpoint=/path/to/agilex_checkpoint
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import field, dataclass

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.deployment"

from phyai_robot.remote import PolicyServer

from .pi05 import Pi05Policy, Pi05PolicyConfig, Pi05ProcessorConfig
from .configuration import parse_config, repository_path
from .processor_loader import build_processing

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "pi05.yaml"


@dataclass
class Pi05ServerConfig:
    """Model server deployment configuration.

    Attributes:
        task: Natural language instruction describing the robot's manipulation task.
        endpoint: ZMQ socket binding URL (default: loopback 'tcp://127.0.0.1:5555').
        lease_timeout_s: Inactive run reservation expiration period in seconds.
            If a connected client terminates without sending 'end', the server automatically
            releases the run after this timeout, preventing permanent lockup.
        policy: Checkpoint path, engine parameters, noise seed, and CUDA graph toggles.
        processor: Tokenizer, normalization stats, camera ordering, and image resize settings.
    """

    task: str = ""
    endpoint: str = "tcp://127.0.0.1:5555"
    lease_timeout_s: float = 60.0
    policy: Pi05PolicyConfig = field(default_factory=Pi05PolicyConfig)
    processor: Pi05ProcessorConfig = field(default_factory=Pi05ProcessorConfig)

    def validate(self) -> None:
        """Validate server configuration before loading weights or binding network sockets."""
        if not self.task.strip() or not self.endpoint.strip():
            raise ValueError("task and endpoint must not be empty")
        if self.lease_timeout_s <= 0:
            raise ValueError("lease_timeout_s must be positive")
        self.policy.validate()
        self.processor.validate()


def main(argv=None) -> None:
    """Parse configuration, load Pi0.5 weights, and start the resident ZMQ policy server."""
    config = parse_config(
        Pi05ServerConfig,
        default_config=DEFAULT_CONFIG,
        argv=argv,
        validate=Pi05ServerConfig.validate,
    )
    checkpoint = repository_path(config.policy.checkpoint)

    # Initialize tokenizer, normalization statistics, and pre/post-processing pipelines (built-in or external factory).
    processing = build_processing(checkpoint, task=config.task, config=config.processor)

    # Initialize the PhyAI CUDA execution engine.
    policy = Pi05Policy(config.policy, processing)

    try:
        spec = policy.spec({"task": processing.task, "seed": config.policy.seed})
        server = PolicyServer(
            policy=policy,
            spec=spec,
            reset_run=policy.reset_run,
            endpoint=config.endpoint,
            lease_timeout_s=config.lease_timeout_s,
        )
        print(
            f"pi0.5 server on {config.endpoint}: task={processing.task!r}, "
            f"cameras={processing.camera_names}, action_dim={processing.action_dim}, "
            f"horizon={policy.horizon}, denoise_steps={policy.num_inference_steps}",
            flush=True,
        )
        server.serve()
    finally:
        policy.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
