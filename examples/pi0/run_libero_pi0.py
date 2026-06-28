"""Run a LIBERO rollout with the phyai pi0 engine.

This script reuses LeRobot's LIBERO environment wrapper, but policy inference
goes through ``phyai.engine.Engine(plugin="pi0")`` and phyai's PI0Processor.
It is intended as a local smoke test for converted pi0 checkpoints.

Example:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl \
    CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=0 \
    uv run python examples/run_libero_pi0.py \
      --checkpoint /path/to/pi0_libero_checkpoint \
      --tokenizer-name /path/to/paligemma-3b-pt-224 \
      --assets-root /path/to/libero-assets \
      --task libero_object \
      --task-ids "[0]" \
      --batch-size 1 \
      --n-episodes 1 \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi0.configuration_pi0 import PI0Config
from phyai.models.pi0.main_pi0 import PI0Args
from phyai.models.pi0.scheduler_ws1_pi0 import PI0Request
from phyai.utils import load_config
from phyai_utils_tools.models.pi0 import PI0Processor


ACTION_DIM = 7
DEFAULT_TOKENIZER = "google/paligemma-3b-pt-224"
LEROBOT_SITE_PACKAGES = (
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
)


def parse_task_ids(value: str | None) -> list[int] | None:
    """Accept ``"[0, 1]"`` or ``"0,1"`` CLI forms."""

    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("--task-ids JSON form must be a list.")
        return [int(x) for x in parsed]
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def add_lerobot_paths(lerobot_root: Path) -> None:
    """Make a sibling LeRobot checkout and its venv importable."""

    candidates = [
        lerobot_root / "src",
        lerobot_root / ".venv" / Path(*LEROBOT_SITE_PACKAGES),
    ]
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.append(str(path))


def configure_libero(assets_root: Path | None) -> None:
    """Write LIBERO's local path config before importing ``libero.libero``."""

    import importlib.util

    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(
            "Could not import LIBERO. Run from the root phyai uv env and pass "
            "--lerobot-root pointing at a LeRobot checkout with the libero extra."
        )

    package_root = Path(spec.origin).resolve().parent
    benchmark_root = package_root / "libero"
    bddl_files = benchmark_root / "bddl_files"
    init_states = benchmark_root / "init_files"
    datasets = package_root / "datasets"

    if assets_root is not None:
        assets_root = assets_root.expanduser().resolve()
        if not assets_root.is_dir():
            raise NotADirectoryError(f"--assets-root is not a directory: {assets_root}")

        link = benchmark_root / "assets"
        if not link.exists():
            link.symlink_to(assets_root, target_is_directory=True)

    config_dir = Path.home() / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(bddl_files),
        "init_states": str(init_states),
        "datasets": str(datasets),
    }
    if assets_root is not None:
        config["assets"] = str(assets_root)
    config_path = config_dir / "config.yaml"
    config_path.write_text("".join(f"{k}: {v}\n" for k, v in config.items()))


def prepare_robosuite_egl_import_env() -> str | None:
    """Satisfy robosuite's import-time EGL device check.

    This robosuite version checks that ``MUJOCO_EGL_DEVICE_ID`` is one of the
    literal entries in ``CUDA_VISIBLE_DEVICES`` when binding_utils is imported.
    Later, MuJoCo's EGL backend sees the remapped visible device list and wants
    an index in that smaller list. We therefore use the physical id for import,
    then switch to the visible-list index before the first environment reset.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return None
    first_visible = visible.split(",")[0].strip()
    if not first_visible or not first_visible.isdigit():
        return None

    original = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = first_visible
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return original


def prepare_robosuite_egl_runtime_env(original: str | None) -> None:
    """Switch EGL device id to MuJoCo's visible-device index."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        if original is None:
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        else:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = original
        return
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"


def make_output_dir(root: Path, task: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    out = root / stamp
    return out.with_name(f"{out.name}_{task}_phyai_pi0")


def render_first_env(env: Any) -> np.ndarray:
    """Render the first sub-env from a Gym vector env."""

    if hasattr(env, "envs"):
        return env.envs[0].render()
    rendered = env.call("render")
    return rendered[0]


def extract_successes(info: dict[str, Any], num_envs: int) -> list[bool]:
    """Handle Gymnasium vector info variants."""

    if "final_info" in info and isinstance(info["final_info"], dict):
        values = info["final_info"].get("is_success")
        if values is not None:
            return np.asarray(values).astype(bool).tolist()
    if "is_success" in info:
        values = info["is_success"]
        if hasattr(values, "tolist"):
            return np.asarray(values).astype(bool).tolist()
        return [bool(values)] * num_envs
    return [False] * num_envs


def observation_to_phyai_raw(
    observation: dict[str, Any],
    *,
    num_images: int,
    image_keys: list[str],
) -> dict[str, Any]:
    """Map LeRobot-format LIBERO observations to phyai PI0Processor input."""

    missing = [key for key in image_keys[:num_images] if key not in observation]
    if missing:
        raise KeyError(f"Missing image keys after LIBERO preprocessing: {missing}")
    if "observation.state" not in observation:
        raise KeyError("Missing observation.state after LIBERO preprocessing.")

    task = observation.get("task", [""] * observation["observation.state"].shape[0])
    return {
        "images": [observation[key] for key in image_keys[:num_images]],
        "state": observation["observation.state"],
        "task": task,
    }


def make_request(
    processor: PI0Processor,
    raw: dict[str, Any],
    *,
    plugin_cfg: PI0Config,
    device: torch.device,
    dtype: torch.dtype,
    fixed_noise: bool,
) -> PI0Request:
    processed = processor.preprocess(raw)
    noise = None
    if fixed_noise:
        gen = torch.Generator(device=device).manual_seed(0)
        noise = torch.randn(
            processed.pixel_values.shape[0],
            plugin_cfg.chunk_size,
            plugin_cfg.max_action_dim,
            dtype=dtype,
            device=device,
            generator=gen,
        )
    return PI0Request(
        pixel_values=processed.pixel_values,
        input_ids=processed.input_ids,
        lang_lens=processed.lang_lens,
        state=processed.state,
        noise=noise,
    )


def select_action_chunk(
    engine: Engine,
    processor: PI0Processor,
    raw: dict[str, Any],
    *,
    plugin_cfg: PI0Config,
    device: torch.device,
    dtype: torch.dtype,
    fixed_noise: bool,
) -> torch.Tensor:
    request = make_request(
        processor,
        raw,
        plugin_cfg=plugin_cfg,
        device=device,
        dtype=dtype,
        fixed_noise=fixed_noise,
    )
    model_actions = engine.step(request)
    return processor.postprocess(model_actions)


def rollout_one(
    *,
    env: Any,
    engine: Engine,
    processor: PI0Processor,
    env_preprocessor: Any,
    preprocess_observation: Any,
    plugin_cfg: PI0Config,
    device: torch.device,
    dtype: torch.dtype,
    fixed_noise: bool,
    image_keys: list[str],
    max_steps_override: int | None,
    save_video: bool,
    video_path: Path | None,
    write_video: Any,
) -> dict[str, Any]:
    """Run one vector-env rollout and return LeRobot-like episode metrics."""

    if hasattr(processor.preprocessor, "reset"):
        processor.preprocessor.reset()
    if hasattr(processor.postprocessor, "reset"):
        processor.postprocessor.reset()

    observation, _ = env.reset()
    frames: list[np.ndarray] = []
    if save_video:
        frames.append(render_first_env(env))

    max_steps = int(env.call("_max_episode_steps")[0])
    if max_steps_override is not None:
        max_steps = min(max_steps, int(max_steps_override))

    action_queue: deque[torch.Tensor] = deque()
    done = np.array([False] * env.num_envs)
    rewards: list[torch.Tensor] = []
    successes: list[torch.Tensor] = []
    steps = 0

    while not np.all(done) and steps < max_steps:
        lerobot_obs = preprocess_observation(observation)
        try:
            lerobot_obs["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            lerobot_obs["task"] = list(env.call("task"))
        prev_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            lerobot_obs = env_preprocessor(lerobot_obs)
        finally:
            torch.set_default_dtype(prev_default_dtype)
        raw = observation_to_phyai_raw(
            lerobot_obs,
            num_images=plugin_cfg.num_images,
            image_keys=image_keys,
        )

        if not action_queue:
            action_chunk = select_action_chunk(
                engine,
                processor,
                raw,
                plugin_cfg=plugin_cfg,
                device=device,
                dtype=dtype,
                fixed_noise=fixed_noise,
            )
            action_queue.extend(action_chunk.transpose(0, 1))

        action = action_queue.popleft()
        action_numpy = action.to("cpu", dtype=torch.float32).numpy()
        action_numpy = np.clip(action_numpy, -1.0, 1.0).astype(np.float32)
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if save_video:
            frames.append(render_first_env(env))

        step_successes = extract_successes(info, env.num_envs)
        done = terminated | truncated | done
        if steps + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        rewards.append(torch.as_tensor(reward))
        successes.append(torch.as_tensor(step_successes, dtype=torch.bool))
        steps += 1

    reward_t = torch.stack(rewards, dim=1) if rewards else torch.zeros(env.num_envs, 0)
    success_t = (
        torch.stack(successes, dim=1)
        if successes
        else torch.zeros(env.num_envs, 0, dtype=torch.bool)
    )
    video_paths: list[str] = []
    if save_video and frames and video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        write_video(str(video_path), np.stack(frames), env.unwrapped.metadata["render_fps"])
        video_paths.append(str(video_path))

    return {
        "sum_rewards": reward_t.sum(dim=1).tolist(),
        "max_rewards": reward_t.max(dim=1).values.tolist() if reward_t.numel() else [0.0],
        "successes": success_t.any(dim=1).tolist() if success_t.numel() else [False],
        "video_paths": video_paths,
        "steps": steps,
    }


def flatten_task_ids(task_ids: Iterable[int] | None) -> list[int] | None:
    return None if task_ids is None else [int(x) for x in task_ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-name",
        default=os.getenv("LEROBOT_PALIGEMMA_TOKENIZER_PATH", DEFAULT_TOKENIZER),
        help="Local PaliGemma tokenizer directory or HF id.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path(os.environ["LIBERO_ASSETS"]) if "LIBERO_ASSETS" in os.environ else None,
        help="Local LIBERO assets directory. Also writes ~/.libero/config.yaml.",
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "lerobot-main",
        help="Local LeRobot checkout whose venv has the LIBERO extra installed.",
    )
    parser.add_argument("--task", default="libero_object")
    parser.add_argument("--task-ids", type=parse_task_ids, default=[0])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-images", type=int, choices=(2, 3), default=None)
    parser.add_argument("--fixed-noise", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-offline", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    args = parser.parse_args()

    if not args.no_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")

    if not args.checkpoint.is_dir():
        raise NotADirectoryError(f"--checkpoint must be a directory: {args.checkpoint}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.n_episodes <= 0:
        raise ValueError("--n-episodes must be positive.")

    add_lerobot_paths(args.lerobot_root.expanduser().resolve())
    configure_libero(args.assets_root)
    original_egl_device_id = prepare_robosuite_egl_import_env()

    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.utils.io_utils import write_video

    plugin_cfg = load_config(args.checkpoint, PI0Config)
    if args.num_images is not None:
        plugin_cfg = replace(plugin_cfg, empty_cameras=3 - args.num_images)

    device = torch.device(args.device)
    dtype = torch.bfloat16
    engine = Engine(
        EngineArgs(
            plugin="pi0",
            plugin_args=PI0Args(
                checkpoint_dir=args.checkpoint,
                config=plugin_cfg,
                max_batch_size=args.batch_size,
            ),
            config=EngineConfig(
                device=DeviceConfig(target=args.device, params_dtype=dtype),
                runtime=RuntimeConfig(use_cuda_graph=not args.no_cuda_graph),
            ),
        )
    )

    envs = None
    try:
        processor = PI0Processor.from_pretrained(
            args.checkpoint,
            tokenizer_name=str(args.tokenizer_name),
            image_size=plugin_cfg.vision.image_size,
            num_channels=plugin_cfg.vision.num_channels,
            num_images=plugin_cfg.num_images,
            max_state_dim=plugin_cfg.max_state_dim,
            action_dim=ACTION_DIM,
            normalize_pixels=True,
            device=device,
            params_dtype=dtype,
            local_files_only=not args.no_offline,
        )

        env_cfg = LiberoEnvConfig(
            task=args.task,
            task_ids=flatten_task_ids(args.task_ids),
            max_parallel_tasks=1,
        )
        env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=None)
        envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=False)
        prepare_robosuite_egl_runtime_env(original_egl_device_id)

        output_dir = make_output_dir(args.output_dir, args.task)
        save_video = not args.no_video
        image_keys = ["observation.images.image", "observation.images.image2", "observation.images.image3"]

        per_task: list[dict[str, Any]] = []
        all_sum_rewards: list[float] = []
        all_max_rewards: list[float] = []
        all_successes: list[bool] = []
        all_video_paths: list[str] = []
        start = time.time()

        for task_group, task_map in envs.items():
            for task_id, env in task_map.items():
                task_sum: list[float] = []
                task_max: list[float] = []
                task_success: list[bool] = []
                task_video_paths: list[str] = []
                for episode_idx in range(args.n_episodes):
                    video_path = (
                        output_dir / "videos" / f"{task_group}_{task_id}" / f"eval_episode_{episode_idx}.mp4"
                        if save_video
                        else None
                    )
                    result = rollout_one(
                        env=env,
                        engine=engine,
                        processor=processor,
                        env_preprocessor=env_preprocessor,
                        preprocess_observation=preprocess_observation,
                        plugin_cfg=plugin_cfg,
                        device=device,
                        dtype=dtype,
                        fixed_noise=args.fixed_noise,
                        image_keys=image_keys,
                        max_steps_override=args.max_steps,
                        save_video=save_video,
                        video_path=video_path,
                        write_video=write_video,
                    )
                    task_sum.extend(float(x) for x in result["sum_rewards"])
                    task_max.extend(float(x) for x in result["max_rewards"])
                    task_success.extend(bool(x) for x in result["successes"])
                    task_video_paths.extend(result["video_paths"])
                    print(
                        f"episode={episode_idx} task_group={task_group} task_id={task_id} "
                        f"steps={result['steps']} success={task_success[-1]} "
                        f"sum_reward={task_sum[-1]:.3f}"
                    )

                metrics = {
                    "sum_rewards": task_sum,
                    "max_rewards": task_max,
                    "successes": task_success,
                    "video_paths": task_video_paths,
                }
                per_task.append(
                    {
                        "task_group": task_group,
                        "task_id": task_id,
                        "metrics": metrics,
                    }
                )
                all_sum_rewards.extend(task_sum)
                all_max_rewards.extend(task_max)
                all_successes.extend(task_success)
                all_video_paths.extend(task_video_paths)

        eval_s = time.time() - start
        n_episodes = len(all_successes)
        overall = {
            "avg_sum_reward": float(np.mean(all_sum_rewards)) if all_sum_rewards else 0.0,
            "avg_max_reward": float(np.mean(all_max_rewards)) if all_max_rewards else 0.0,
            "pc_success": float(np.mean(all_successes)) if all_successes else 0.0,
            "n_episodes": n_episodes,
            "eval_s": eval_s,
            "eval_ep_s": eval_s / max(n_episodes, 1),
            "video_paths": all_video_paths,
        }

        print("\nPHYAI PI0 LIBERO smoke complete")
        print(f"checkpoint         : {args.checkpoint}")
        print(f"tokenizer          : {args.tokenizer_name}")
        print(f"camera count       : {plugin_cfg.num_images}")
        print(f"task               : {args.task}")
        print(f"task_ids           : {args.task_ids}")
        print(f"output_dir         : {output_dir}")
        print("\nOverall Aggregated Metrics:")
        print(overall)
        print("\nAggregated Metrics for per_task:")
        print(per_task)
    finally:
        if envs is not None:
            close_envs(envs)
        engine.close()


if __name__ == "__main__":
    main()
