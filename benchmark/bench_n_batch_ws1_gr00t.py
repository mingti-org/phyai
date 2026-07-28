"""Prepared-request GR00T-N1.7 ws1 latency benchmark, swept over batch sizes.

Builds the GR00T-N1.7 engine via the standard plugin path once per batch
size, prepares one synthetic request during setup, then times repeated
``engine.step(request)`` calls with :class:`bench_n_batch.NBatchBenchRunner`.

The timed latency includes the GR00T scheduler path, Qwen3-VL backbone,
and flow-matching action-head denoising. Raw observation preprocessing and
tokenization run once during setup, so every timed step reuses the same
prepared request. Action decoding is not performed.

Inputs are synthetic, so action values are not meaningful. Use this script for
stable cross-machine latency numbers, not accuracy or parity validation.
With CUDA graphs enabled, the prepared request is captured while the engine is
constructed; benchmark warmups only stabilize the latency measurement.

Run::

    uv run python \\
        benchmark/bench_n_batch_ws1_gr00t.py \\
        --checkpoint <gr00t-checkpoint-dir> \\
        --embodiment-tag LIBERO_PANDA \\
        --batch-sizes 1 2 4 --n-warmup 5 --n-timed 30 \\
        --result-file gr00t_ws1_results.jsonl

Profile under Nsight Systems::

    mkdir -p ./prof
    nsys profile --capture-range=cudaProfilerApi \\
        --capture-range-end=stop -o ./prof/gr00t_ws1 \\
        uv run python benchmark/bench_n_batch_ws1_gr00t.py \\
        --checkpoint <gr00t-checkpoint-dir> --embodiment-tag LIBERO_PANDA \\
        --batch-sizes 1 --profile-backend nsys \\
        --profile-start-step 5 --profile-num-steps 3

Selecting ``--profile-backend nsys`` without an enclosing ``nsys profile``
only emits NVTX/cudaProfiler calls; it does not create a report by itself.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

import bench_n_batch as bnb
from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.gr00t_n17.configuration_gr00t_n17 import GR00TN17Config
from phyai.models.gr00t_n17.main_gr00t_n17 import GR00TN17Args
from phyai.models.gr00t_n17.scheduler_ws1_gr00t_n17 import GR00TN17Request
from phyai.utils import load_config
from phyai.utils.profile import (
    add_profile_cli_args,
    get_profiler,
    install_profiler,
    profile_config_from_args,
)


_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

_EMBODIMENT_ALIASES = {
    "LIBERO_PANDA": "libero_sim",
}


def load_num_views(checkpoint: Path, embodiment_tag: str) -> int | None:
    processor_config = checkpoint / "processor_config.json"
    if not processor_config.is_file():
        return None
    data = json.loads(processor_config.read_text(encoding="utf-8"))
    modality_configs = data.get("processor_kwargs", {}).get("modality_configs", {})
    key = _EMBODIMENT_ALIASES.get(embodiment_tag, embodiment_tag)
    config = modality_configs.get(key)
    if config is None:
        return None
    return len(config.get("video", {}).get("modality_keys", []))


def make_synthetic_observation(
    processor,
    *,
    batch_size: int,
    image_size: int,
    task: str,
):
    """Build a raw GR00T observation matching the checkpoint modality config."""
    from phyai_utils_tools.models.gr00t import GR00TObservation

    cfg = processor.modality_config
    tag = processor.embodiment_tag

    video = {}
    for key in cfg["video"].modality_keys:
        num_frames = len(cfg["video"].delta_indices)
        video[key] = np.random.randint(
            0,
            256,
            size=(batch_size, num_frames, image_size, image_size, 3),
            dtype=np.uint8,
        )

    state = {}
    for key in cfg["state"].modality_keys:
        num_steps = len(cfg["state"].delta_indices)
        dim = int(processor.norm_params[tag]["state"][key]["dim"])
        state[key] = (
            np.random.rand(batch_size, num_steps, dim).astype(np.float32) * 2 - 1
        )

    language_key = cfg["language"].modality_keys[0]
    language = {language_key: [[task]] * batch_size}
    return GR00TObservation(video=video, state=state, language=language)


def move_tensors_to_device(
    tensors: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move prepared request tensors once during setup."""
    return {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in tensors.items()
    }


def make_request(
    *,
    checkpoint: Path,
    embodiment_tag: str,
    processor_model_name_or_path: str,
    transformers_loading_kwargs: dict[str, Any],
    batch_size: int,
    image_size: int,
    task: str,
) -> GR00TN17Request:
    """Prepare a CPU capture profile matching the timed request."""
    from phyai_utils_tools.models.gr00t import GR00TProcessor

    processor = GR00TProcessor.from_pretrained(
        checkpoint,
        embodiment_tag=embodiment_tag,
        model_name=processor_model_name_or_path,
        transformers_loading_kwargs=transformers_loading_kwargs,
    )
    observation = make_synthetic_observation(
        processor,
        batch_size=batch_size,
        image_size=image_size,
        task=task,
    )
    prepared = processor.process_observation(observation)
    return GR00TN17Request(tensors=prepared.tensors)


def move_request_to_device(
    request: GR00TN17Request,
    *,
    plugin_cfg: GR00TN17Config,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> GR00TN17Request:
    """Build the fixed GPU request reused by every timed step."""
    tensors = move_tensors_to_device(request.tensors, device=device)
    noise = torch.randn(
        batch_size,
        plugin_cfg.action_head.action_horizon,
        plugin_cfg.action_head.max_action_dim,
        dtype=dtype,
        device=device,
    )
    return GR00TN17Request(tensors=tensors, noise=noise)


def make_setup_fn(
    *,
    checkpoint: Path,
    embodiment_tag: str,
    dtype: torch.dtype,
    device_target: str,
    use_cuda_graph: bool,
    image_size: int,
    task: str,
    online: bool,
    trust_remote_code: bool,
    processor_model_name_or_path: str | None,
    emit_module_nvtx: bool,
):
    """Build the per-batch-size ``setup_fn`` closure."""
    plugin_cfg = load_config(checkpoint, GR00TN17Config)
    device = torch.device(device_target)
    model_name = processor_model_name_or_path or plugin_cfg.backbone.model_name
    loading_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": not online,
    }

    def setup_fn(batch_size: int) -> bnb.BenchSpec:
        capture_profile = make_request(
            checkpoint=checkpoint,
            embodiment_tag=embodiment_tag,
            processor_model_name_or_path=model_name,
            transformers_loading_kwargs=loading_kwargs,
            batch_size=batch_size,
            image_size=image_size,
            task=task,
        )
        engine = Engine(
            EngineArgs(
                plugin="gr00t_n17",
                plugin_args=GR00TN17Args(
                    checkpoint_dir=checkpoint,
                    max_batch_size=batch_size,
                    capture_profiles=(capture_profile,),
                ),
                config=EngineConfig(
                    device=DeviceConfig(target=device_target, params_dtype=dtype),
                    runtime=RuntimeConfig(use_cuda_graph=use_cuda_graph),
                ),
            )
        )
        if emit_module_nvtx:
            profiler = get_profiler()
            for root_name, module in engine.entry.dump_targets().items():
                profiler.attach_module_hooks(module, prefix=root_name)
        request = move_request_to_device(
            capture_profile,
            plugin_cfg=plugin_cfg,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        return bnb.BenchSpec(
            name="ws1_gr00t_n17",
            step_callable=lambda: engine.step(request),
            teardown_callable=engine.close,
        )

    return setup_fn


def make_extras_fn(
    *,
    dtype_name: str,
    device_target: str,
    use_cuda_graph: bool,
    num_views: int | None,
    action_horizon: int,
    action_dim: int,
    denoising_steps: int,
    image_size: int,
    task: str,
    embodiment_tag: str,
    processor_model_name_or_path: str | None,
):
    def extras_fn(batch_size: int, spec: bnb.BenchSpec) -> dict[str, Any]:
        return {
            "model": "gr00t_n17",
            "scheduler": "ws1",
            "dtype": dtype_name,
            "vit_dtype": dtype_name,
            "device": device_target,
            "use_cuda_graph": use_cuda_graph,
            "max_batch_size": batch_size,
            "embodiment_tag": embodiment_tag,
            "num_views": num_views,
            "image_size": image_size,
            "task": task,
            "processor_model_name_or_path": processor_model_name_or_path,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "denoising_steps": denoising_steps,
            "timed_surface": "prepared_request_inference",
        }

    return extras_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "GR00T checkpoint folder containing config.json, safetensors, "
            "processor_config.json, and statistics.json."
        ),
    )
    parser.add_argument(
        "--embodiment-tag",
        default="LIBERO_PANDA",
        help="Embodiment tag used by the GR00T processor.",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="bf16",
        help="Engine params_dtype and fixed denoising-noise dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Engine device target (default "cuda"; use "cpu" only for debug).',
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA graph capture and replay.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Synthetic camera H/W before the GR00T eval transform.",
    )
    parser.add_argument(
        "--task",
        default="pick up the object",
        help="Synthetic language instruction passed through the GR00T processor.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow HuggingFace downloads (default: local_files_only).",
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
        "--processor-model-name-or-path",
        type=str,
        default=None,
        help=(
            "Optional processor-only tokenizer/preprocessor path or Hugging Face "
            "repo id. This does not override the engine Backbone config or weights. "
            "Defaults to the model name stored in the GR00T config."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

    bnb.add_bench_cli_args(parser)
    add_profile_cli_args(parser)

    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        raise NotADirectoryError(
            f"--checkpoint must be a directory, got: {args.checkpoint}"
        )
    if args.trust_remote_code:
        warnings.warn(
            "--trust-remote-code allows execution of code from the processor "
            "model repository.",
            stacklevel=1,
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype = _DTYPES[args.dtype]
    use_cuda_graph = not args.no_cuda_graph and torch.device(args.device).type == "cuda"
    plugin_cfg = load_config(args.checkpoint, GR00TN17Config)

    profile_cfg = profile_config_from_args(args)
    install_profiler(profile_cfg)

    setup_fn = make_setup_fn(
        checkpoint=args.checkpoint,
        embodiment_tag=args.embodiment_tag,
        dtype=dtype,
        device_target=args.device,
        use_cuda_graph=use_cuda_graph,
        image_size=args.image_size,
        task=args.task,
        online=args.online,
        trust_remote_code=args.trust_remote_code,
        processor_model_name_or_path=args.processor_model_name_or_path,
        emit_module_nvtx=profile_cfg.emit_module_nvtx,
    )
    extras_fn = make_extras_fn(
        dtype_name=args.dtype,
        device_target=args.device,
        use_cuda_graph=use_cuda_graph,
        num_views=load_num_views(args.checkpoint, args.embodiment_tag),
        action_horizon=plugin_cfg.action_head.action_horizon,
        action_dim=plugin_cfg.action_head.max_action_dim,
        denoising_steps=plugin_cfg.action_head.num_inference_timesteps,
        image_size=args.image_size,
        task=args.task,
        embodiment_tag=args.embodiment_tag,
        processor_model_name_or_path=args.processor_model_name_or_path,
    )

    runner = bnb.NBatchBenchRunner(
        setup_fn=setup_fn,
        extras_fn=extras_fn,
        **bnb.bench_runner_kwargs_from_args(args),
    )
    runner.run()


if __name__ == "__main__":
    main()
