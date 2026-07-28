"""Serve PhyAI GR00T-N1.7 to the official LIBERO rollout client.

The official ``rollout_policy.py`` process stays in its Python 3.10 LIBERO
environment and talks to this process through the same ZeroMQ/msgpack protocol
as NVIDIA's policy server. This process stays in the PhyAI environment and
owns the processor, CUDA engine, and checkpoint.

The transport packages are intentionally not added to PhyAI's locked project
dependencies. Run the adapter with an ephemeral uv overlay instead::

    uv run --with pyzmq --with msgpack --with msgpack-numpy \
        python benchmark/gr00t/libero_rollout_server.py \
        --checkpoint <gr00t-checkpoint-dir> \
        --host 127.0.0.1 --port 5556 --max-batch-size 5 \
        --capture-suite libero_10

Then point the unmodified official client at the same address::

    <libero-python> gr00t/eval/rollout_policy.py \
        --policy-client-host 127.0.0.1 --policy-client-port 5556 \
        --n-envs 5 <other-rollout-options>

With CUDA Graphs enabled, ``--max-batch-size`` must equal the client's actual
batch size (``--n-envs`` for the official client). With ``--no-cuda-graph``, it
is only an upper bound.
Wait for the server's ``Server is ready`` log before starting the client; model
loading and the setup profile intentionally finish before the socket is bound.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import logging
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np


logger = logging.getLogger("phyai.libero_rollout_server")

LIBERO_10_CAPTURE_TASKS = (
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on "
    "the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right "
    "of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
)


def require_transport_modules():
    """Import the optional official-wire-protocol dependencies."""
    try:
        import msgpack
        import msgpack_numpy
        import zmq
    except ImportError as error:
        raise RuntimeError(
            "The LIBERO policy transport needs pyzmq, msgpack, and "
            "msgpack-numpy. Keep the PhyAI lockfile unchanged by launching "
            "with: uv run --with pyzmq --with msgpack --with msgpack-numpy "
            "python benchmark/gr00t/libero_rollout_server.py ..."
        ) from error
    return zmq, msgpack, msgpack_numpy


class OfficialMsgSerializer:
    """Serializer compatible with Isaac-GR00T's ``PolicyClient``."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        _, msgpack, msgpack_numpy = require_transport_modules()

        def encode(value: Any) -> Any:
            if isinstance(value, np.ndarray) and value.dtype.kind == "O":
                raise TypeError(
                    "Refusing to encode an object-dtype ndarray because "
                    "msgpack-numpy would serialize it with pickle."
                )
            return msgpack_numpy.encode(value)

        return msgpack.packb(data, default=encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        _, msgpack, msgpack_numpy = require_transport_modules()

        def decode(value: Any) -> Any:
            if isinstance(value, dict):
                marker_key = "".join(("n", "d"))
                ndarray_marker = value.get(marker_key.encode(), value.get(marker_key))
                kind = value.get(b"kind", value.get("kind"))
                if ndarray_marker and kind in (b"O", "O"):
                    raise ValueError(
                        "Refusing to decode a pickle-bearing object-dtype ndarray."
                    )
            return msgpack_numpy.decode(value)

        return msgpack.unpackb(data, object_hook=decode, raw=False)


class OfficialPolicyServer:
    """Small REP server implementing the endpoints used by ``PolicyClient``."""

    def __init__(
        self,
        policy: PhyAILiberoPolicy,
        *,
        host: str,
        port: int,
        api_token: str | None = None,
    ) -> None:
        zmq, _, _ = require_transport_modules()
        self.policy = policy
        self.api_token = api_token
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://{host}:{port}")
        self.endpoint = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self.handlers: dict[str, tuple[Callable[..., Any], bool]] = {
            "ping": (self.handle_ping, False),
            "kill": (self.handle_kill, False),
            "get_action": (self.policy.get_action, True),
            "reset": (self.policy.reset, True),
            "get_modality_config": (self.policy.get_modality_config, False),
        }

    @staticmethod
    def handle_ping() -> dict[str, str]:
        return {
            "status": "ok",
            "message": "Server is running",
            "backend": "phyai",
            "policy": "gr00t_n17",
        }

    def handle_kill(self) -> None:
        self.running = False

    def run(self) -> None:
        logger.info("Server is ready and listening on %s", self.endpoint)
        while self.running:
            try:
                request = OfficialMsgSerializer.from_bytes(self.socket.recv())
                response = self.handle_request(request)
            except KeyboardInterrupt:
                raise
            except Exception as error:
                logger.exception("Policy request failed")
                response = {"error": str(error)}
            self.socket.send(OfficialMsgSerializer.to_bytes(response))

    def handle_request(self, request: Any) -> Any:
        if not isinstance(request, dict):
            raise TypeError(
                f"Policy request must be a dict, got {type(request).__name__}."
            )
        if self.api_token is not None and request.get("api_token") != self.api_token:
            return {"error": "Unauthorized: Invalid API token"}
        endpoint = request.get("endpoint", "get_action")
        if endpoint not in self.handlers:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        handler, requires_input = self.handlers[endpoint]
        if not requires_input:
            return handler()
        data = request.get("data", {})
        if not isinstance(data, dict):
            raise TypeError(f"Endpoint data must be a dict, got {type(data).__name__}.")
        return handler(**data)

    def close(self) -> None:
        self.running = False
        self.socket.close(linger=0)
        self.context.term()


class PhyAILiberoPolicy:
    """Translate official flat simulation observations into PhyAI requests."""

    def __init__(
        self,
        *,
        processor: Any,
        engine: Any,
        request_type: type,
        device: Any,
        max_batch_size: int,
        seed: int,
    ) -> None:
        self.processor = processor
        self.engine = engine
        self.request_type = request_type
        self.device = device
        self.max_batch_size = max_batch_size
        self.seed = seed

    @staticmethod
    def get_flat_value(observation: dict[str, Any], modality: str, key: str) -> Any:
        nested = observation.get(modality)
        if isinstance(nested, dict) and key in nested:
            return nested[key]
        flat_key = f"{modality}.{key}"
        if flat_key not in observation:
            raise KeyError(f"Observation is missing required key {flat_key!r}.")
        return observation[flat_key]

    @staticmethod
    def get_language_value(observation: dict[str, Any], key: str) -> Any:
        nested = observation.get("language")
        if isinstance(nested, dict) and key in nested:
            return nested[key]
        for candidate in (
            key,
            "task",
            "annotation.human.action.task_description",
            "annotation.human.coarse_action",
        ):
            if candidate in observation:
                return observation[candidate]
        raise KeyError(
            f"Observation is missing language key {key!r}; accepted fallback "
            "keys are 'task', 'annotation.human.action.task_description', "
            "and 'annotation.human.coarse_action'."
        )

    @staticmethod
    def to_language_batch(value: Any, horizon: int) -> list[list[str]]:
        if isinstance(value, str):
            rows: list[Any] = [value]
        elif isinstance(value, np.ndarray):
            rows = value.tolist()
        elif isinstance(value, (tuple, list)):
            rows = list(value)
        else:
            raise TypeError(
                "Language observation must be a string, list, tuple, or ndarray, "
                f"got {type(value).__name__}."
            )

        result: list[list[str]] = []
        for row in rows:
            if isinstance(row, np.ndarray):
                row = row.tolist()
            if isinstance(row, (tuple, list)):
                values = [str(item) for item in row]
                if len(values) != horizon:
                    raise ValueError(
                        f"Language horizon must be {horizon}, got {len(values)}."
                    )
                result.append(values)
            else:
                result.append([str(row)] * horizon)
        return result

    def convert_observation(self, observation: dict[str, Any]):
        from phyai_utils_tools.models.gr00t import GR00TObservation

        if not isinstance(observation, dict):
            raise TypeError(
                f"Observation must be a dict, got {type(observation).__name__}."
            )
        config = self.processor.modality_config
        video: dict[str, np.ndarray] = {}
        for key in config["video"].modality_keys:
            value = np.asarray(self.get_flat_value(observation, "video", key))
            if value.dtype != np.uint8:
                raise TypeError(
                    f"Video key 'video.{key}' must have dtype uint8, got {value.dtype}."
                )
            video[key] = np.ascontiguousarray(value)

        state = {
            key: np.ascontiguousarray(
                np.asarray(
                    self.get_flat_value(observation, "state", key), dtype=np.float32
                )
            )
            for key in config["state"].modality_keys
        }
        language = {}
        for key in config["language"].modality_keys:
            language[key] = self.to_language_batch(
                self.get_language_value(observation, key),
                len(config["language"].delta_indices),
            )
        return GR00TObservation(video=video, state=state, language=language)

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        import torch

        del options
        start = time.perf_counter()
        prepared = self.processor.process_observation(
            self.convert_observation(observation)
        )
        batch_size = int(prepared.tensors["state"].shape[0])
        if batch_size > self.max_batch_size:
            raise ValueError(
                "Rollout request batch exceeds --max-batch-size: "
                f"{batch_size} > {self.max_batch_size}. In CUDA Graph mode, "
                "restart with --max-batch-size equal to the client's actual "
                "request batch (--n-envs for rollout_policy.py). With "
                "--no-cuda-graph, set it to at least the request batch."
            )
        tensors = {
            key: value.to(device=self.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in prepared.tensors.items()
        }
        normalized_action = self.engine.step(self.request_type(tensors=tensors))
        action = self.processor.decode_action(
            normalized_action, raw_state=prepared.raw_state
        )
        return action, {
            "backend": "phyai",
            "batch_size": batch_size,
            "server_step_seconds": time.perf_counter() - start,
        }

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        import torch

        seed = self.seed if options is None else int(options.get("seed", self.seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return {}

    def get_modality_config(self) -> dict[str, dict[str, Any]]:
        """Return official ``ModalityConfig`` envelopes for client introspection."""
        result = {}
        for modality, config in self.processor.modality_config.items():
            action_configs = config.action_configs
            result[modality] = {
                "__ModalityConfig__": True,
                "as_json": {
                    "delta_indices": list(config.delta_indices),
                    "modality_keys": list(config.modality_keys),
                    "sin_cos_embedding_keys": None,
                    "mean_std_embedding_keys": (
                        list(config.mean_std_embedding_keys)
                        if config.mean_std_embedding_keys is not None
                        else None
                    ),
                    "action_configs": (
                        [
                            asdict(item) if is_dataclass(item) else item
                            for item in action_configs
                        ]
                        if action_configs is not None
                        else None
                    ),
                },
            }
        return result

    def close(self) -> None:
        self.engine.close()


def make_capture_observation(
    processor: Any,
    *,
    batch_size: int,
    image_size: int,
    task: str,
):
    """Build a representative setup-time CUDA Graph profile."""
    from phyai_utils_tools.models.gr00t import GR00TObservation

    config = processor.modality_config
    video = {
        key: np.zeros(
            (
                batch_size,
                len(config["video"].delta_indices),
                image_size,
                image_size,
                3,
            ),
            dtype=np.uint8,
        )
        for key in config["video"].modality_keys
    }
    state = {
        key: np.zeros(
            (
                batch_size,
                len(config["state"].delta_indices),
                int(
                    processor.norm_params[processor.embodiment_tag]["state"][key]["dim"]
                ),
            ),
            dtype=np.float32,
        )
        for key in config["state"].modality_keys
    }
    language = {
        key: [
            [task for _ in config["language"].delta_indices] for _ in range(batch_size)
        ]
        for key in config["language"].modality_keys
    }
    return GR00TObservation(video=video, state=state, language=language)


def build_policy(args: argparse.Namespace) -> PhyAILiberoPolicy:
    """Load the PhyAI processor and engine without importing Isaac-GR00T."""
    import torch

    from phyai.engine import Engine, EngineArgs
    from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
    from phyai.models.gr00t_n17.configuration_gr00t_n17 import GR00TN17Config
    from phyai.models.gr00t_n17.main_gr00t_n17 import GR00TN17Args
    from phyai.models.gr00t_n17.scheduler_ws1_gr00t_n17 import GR00TN17Request
    from phyai.utils import load_config
    from phyai_utils_tools.models.gr00t import GR00TProcessor

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise NotADirectoryError(f"--checkpoint must be a directory, got: {checkpoint}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.params_dtype]
    device = torch.device(args.device)
    loading_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": not args.online,
    }
    model_config = load_config(checkpoint, GR00TN17Config)
    processor = GR00TProcessor.from_pretrained(
        checkpoint,
        embodiment_tag=args.embodiment_tag,
        model_name=(
            args.processor_model_name_or_path or model_config.backbone.model_name
        ),
        transformers_loading_kwargs=loading_kwargs,
    )
    capture_tasks = list(args.capture_task)
    if args.capture_suite == "libero_10":
        capture_tasks.extend(LIBERO_10_CAPTURE_TASKS)
    capture_tasks = list(dict.fromkeys(capture_tasks))
    capture_profiles = tuple(
        GR00TN17Request(
            tensors=processor.process_observation(
                make_capture_observation(
                    processor,
                    batch_size=args.max_batch_size,
                    image_size=args.capture_image_size,
                    task=task,
                )
            ).tensors
        )
        for task in capture_tasks
    )

    engine = Engine(
        EngineArgs(
            plugin="gr00t_n17",
            plugin_args=GR00TN17Args(
                checkpoint_dir=checkpoint,
                max_batch_size=args.max_batch_size,
                capture_profiles=capture_profiles,
            ),
            config=EngineConfig(
                device=DeviceConfig(target=args.device, params_dtype=dtype),
                runtime=RuntimeConfig(use_cuda_graph=not args.no_cuda_graph),
            ),
        )
    )
    return PhyAILiberoPolicy(
        processor=processor,
        engine=engine,
        request_type=GR00TN17Request,
        device=device,
        max_batch_size=args.max_batch_size,
        seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help=(
            "Batch size captured during CUDA Graph setup; it must equal the "
            "client's actual request batch (--n-envs for the official client). "
            "With --no-cuda-graph, this is only an upper bound (default: 8)."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--params-dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument(
        "--processor-model-name-or-path",
        default=None,
        help=(
            "Optional tokenizer/preprocessor path or Hugging Face repo id used "
            "only by GR00TProcessor. Defaults to the model name stored in the "
            "GR00T config."
        ),
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow Hugging Face downloads (default: local files only).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allow Transformers to execute custom code from the processor model "
            "repository. Only use this with a repository you trust."
        ),
    )
    parser.add_argument(
        "--capture-task",
        action="append",
        default=[],
        help=(
            "Task text scanned before the server accepts clients. Repeat this "
            "option to cover every task structure; equivalent Graph structures "
            "are captured only once."
        ),
    )
    parser.add_argument(
        "--capture-suite",
        choices=("libero_10",),
        default=None,
        help=(
            "Scan every task in the selected fixed LIBERO suite and capture "
            "each unique Graph structure once."
        ),
    )
    parser.add_argument("--capture-image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--api-token", default=None)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if args.max_batch_size <= 0:
        parser.error("--max-batch-size must be positive.")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535].")
    if args.capture_image_size <= 0:
        parser.error("--capture-image-size must be positive.")
    if not args.no_cuda_graph and args.capture_suite is None and not args.capture_task:
        parser.error(
            "CUDA Graph mode requires --capture-suite or at least one --capture-task."
        )
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.trust_remote_code:
        logger.warning(
            "--trust-remote-code allows execution of code from the processor "
            "model repository."
        )
    require_transport_modules()
    policy = build_policy(args)
    server: OfficialPolicyServer | None = None
    try:
        server = OfficialPolicyServer(
            policy,
            host=args.host,
            port=args.port,
            api_token=args.api_token,
        )
        server.run()
    except KeyboardInterrupt:
        logger.info("Stopping policy server")
    finally:
        if server is not None:
            server.close()
        policy.close()


if __name__ == "__main__":
    main()
