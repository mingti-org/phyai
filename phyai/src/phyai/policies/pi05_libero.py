"""Thin LIBERO adapter for pi0.5 PhyAI inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import BackendConfig, DeviceConfig, EngineConfig, RuntimeConfig
from phyai.env import envs
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request
from phyai_utils_tools.models.pi05 import PI05_DEFAULT_TOKENIZER_NAME, PI05Processor
from phyai_utils_tools.processing.transition import IMAGES, STATE, TASK

LIBERO_AGENTVIEW_KEYS: tuple[str, ...] = (
    "agentview",
    "agentview_image",
    "image",
    "observation.images.image",
)
LIBERO_WRIST_KEYS: tuple[str, ...] = (
    "wrist",
    "robot0_eye_in_hand_image",
    "wrist_image",
    "image2",
    "observation.images.image2",
)


def _lerobot_pi05_weight_remap(key: str) -> str | None:
    """Strip LeRobot's outer model prefix and drop inference-unused keys."""
    if key.startswith("model."):
        key = key[len("model.") :]
    if key == "paligemma_with_expert.gemma_expert.lm_head.weight":
        return None
    return key


class PI05LiberoPolicy:
    """Adapt vla-evaluation-harness LIBERO observations to ``PI05Processor``."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: str = "cuda",
        params_dtype: torch.dtype = torch.bfloat16,
        max_batch_size: int = 1,
        use_cuda_graph: bool = True,
        attn_backend: str = "flashinfer",
        norm_backend: str = "phyai-kernel",
        linear_backend: str | None = "flashinfer",
        flashinfer_workspace_bytes: int = 512 * 1024 * 1024,
        tokenizer_name: str | None = None,
        camera_mode: str | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.params_dtype = params_dtype
        self.max_batch_size = int(max_batch_size)
        self.config = self._read_config()
        self.image_size = self._resolve_image_size(self.config)
        self._action_dim = self._resolve_action_dim(self.config)
        self.max_action_dim = int(self.config.get("max_action_dim", 32))
        self._chunk_size = int(self.config.get("chunk_size", PI05Config().chunk_size))
        self.camera_names = self._resolve_camera_names(camera_mode)
        self.tokenizer_name = self._resolve_tokenizer_name(tokenizer_name)
        self.prompt_mode = str(
            self.config.get("phyai_prompt_mode", "lerobot_state_bins")
        )
        self.normalization_mode = str(
            self.config.get("phyai_normalization_mode", "mean_std")
        )
        self._use_phyai_compat = (
            "phyai_prompt_mode" in self.config
            or "phyai_normalization_mode" in self.config
        )
        self._normalizer_stats = self._load_processor_state(
            "policy_preprocessor.json", "normalizer_processor"
        )
        self._unnormalizer_stats = self._load_processor_state(
            "policy_postprocessor.json", "unnormalizer_processor"
        )
        if self._use_phyai_compat:
            self._validate_compat_stats()
        self._tokenizer = None
        self.processor = PI05Processor.from_pretrained(
            self.checkpoint_dir,
            tokenizer_name=self.tokenizer_name,
            image_size=self.image_size,
            num_channels=3,
            num_images=len(self.camera_names),
            action_dim=self._action_dim,
            normalize_pixels=True,
            device=device,
            params_dtype=params_dtype,
        )
        self.engine = Engine(
            EngineArgs(
                plugin="pi05",
                plugin_args=PI05Args(
                    checkpoint_dir=self.checkpoint_dir,
                    max_batch_size=self.max_batch_size,
                    weight_remap=_lerobot_pi05_weight_remap,
                    inputs_image_shape=[
                        [self.image_size, self.image_size, 3] for _ in self.camera_names
                    ],
                ),
                config=EngineConfig(
                    backends=BackendConfig(
                        attn=attn_backend, norm=norm_backend, linear=linear_backend
                    ),
                    device=DeviceConfig(target=device, params_dtype=params_dtype),
                    runtime=RuntimeConfig(
                        use_cuda_graph=use_cuda_graph,
                        flashinfer_workspace_bytes=flashinfer_workspace_bytes,
                        force_linear_kernel=linear_backend,
                    ),
                ),
            )
        )

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return int(self.processor.action_dim or self._action_dim)

    @staticmethod
    def _resolve_image_size(config: dict[str, Any]) -> int:
        resolution = config.get("image_resolution")
        if isinstance(resolution, list) and resolution:
            return int(resolution[0])
        return PI05Config().vision.image_size

    @staticmethod
    def _resolve_action_dim(config: dict[str, Any]) -> int:
        shape = config.get("output_features", {}).get("action", {}).get("shape")
        if isinstance(shape, list) and shape:
            return int(shape[-1])
        return 7

    def _read_config(self) -> dict[str, Any]:
        path = self.checkpoint_dir / "config.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_camera_names(self, camera_mode: str | None) -> list[str]:
        mode = camera_mode or envs.PHYAI_CAMERA_MODE.get() or "three_camera"
        if mode == "two_camera":
            return ["agentview", "wrist"]
        if mode == "three_camera":
            return ["agentview", "wrist", "empty"]
        raise ValueError(f"Unsupported PHYAI_CAMERA_MODE={mode!r}.")

    def _resolve_tokenizer_name(self, tokenizer_name: str | None) -> str:
        if tokenizer_name:
            return tokenizer_name
        if env_tokenizer := envs.PHYAI_TOKENIZER_PATH.get():
            return env_tokenizer
        if config_tokenizer := self.config.get("tokenizer_name"):
            return str(config_tokenizer)
        return PI05_DEFAULT_TOKENIZER_NAME

    def _load_processor_state(
        self, config_name: str, registry_name: str
    ) -> dict[str, torch.Tensor]:
        path = self.checkpoint_dir / config_name
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        for step in config.get("steps", []):
            if step.get("registry_name") != registry_name:
                continue
            state_file = step.get("state_file")
            if not state_file:
                return {}
            return load_file(str(self.checkpoint_dir / state_file))
        return {}

    def _validate_compat_stats(self) -> None:
        if self.normalization_mode == "openpi_quantile":
            normalizer_keys = ("observation.state.min", "observation.state.max")
            unnormalizer_keys = ("action.min", "action.max")
        else:
            normalizer_keys = ("observation.state.mean", "observation.state.std")
            unnormalizer_keys = ("action.mean", "action.std")
        missing = [
            f"normalizer:{key}"
            for key in normalizer_keys
            if key not in self._normalizer_stats
        ]
        missing.extend(
            f"unnormalizer:{key}"
            for key in unnormalizer_keys
            if key not in self._unnormalizer_stats
        )
        if missing:
            raise ValueError(
                f"{self.checkpoint_dir}: compat normalization requires missing stats "
                f"{', '.join(missing)}"
            )

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    def observation_to_raw(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            IMAGES: [
                self._extract_camera_tensor(obs, name) for name in self.camera_names
            ],
            STATE: self._extract_state(obs),
            TASK: [self._extract_task(obs)],
        }

    def observation_to_request_inputs(
        self, obs: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        if not self._use_phyai_compat:
            processed = self.processor.preprocess(self.observation_to_raw(obs))
            return {
                "pixel_values": processed.pixel_values,
                "input_ids": processed.input_ids,
                "lang_lens": processed.lang_lens,
            }
        pixel_values = (
            torch.stack(
                [
                    self._extract_camera_model_tensor(obs, name).squeeze(0)
                    for name in self.camera_names
                ],
                dim=0,
            )
            .unsqueeze(0)
            .to(self.device)
        )
        state = self._normalize_state(self._extract_state(obs))
        input_ids, lang_lens = self._tokenize_inputs([self._extract_task(obs)], state)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids.to(self.device),
            "lang_lens": lang_lens.to(self.device),
        }

    def _extract_camera_tensor(
        self, obs: dict[str, Any], camera_name: str
    ) -> torch.Tensor:
        image = self._extract_camera_image(obs, camera_name)
        return self._image_to_raw_tensor(image)

    def _extract_camera_model_tensor(
        self, obs: dict[str, Any], camera_name: str
    ) -> torch.Tensor:
        image = self._extract_camera_image(obs, camera_name)
        return self._image_to_model_tensor(image)

    def _extract_camera_image(
        self, obs: dict[str, Any], camera_name: str
    ) -> np.ndarray:
        if camera_name == "agentview":
            return self._extract_image(obs, LIBERO_AGENTVIEW_KEYS)
        if camera_name == "wrist":
            return self._extract_image(obs, LIBERO_WRIST_KEYS)
        if camera_name == "empty":
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        raise ValueError(f"Unsupported camera_name={camera_name!r}.")

    @staticmethod
    def _extract_image(obs: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray:
        candidates: list[Any] = []
        images = obs.get("images")
        if isinstance(images, dict):
            candidates.extend(images.get(k) for k in keys)
        candidates.extend(obs.get(k) for k in keys)
        for candidate in candidates:
            if candidate is None:
                continue
            array = np.asarray(candidate)
            if array.ndim == 4:
                array = array[0]
            if array.ndim != 3:
                continue
            if array.shape[0] == 3 and array.shape[-1] != 3:
                array = np.transpose(array, (1, 2, 0))
            if array.shape[-1] == 3:
                return array
        raise KeyError(f"LIBERO observation does not contain any image keys: {keys}.")

    @staticmethod
    def _image_to_raw_tensor(image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        if array.max(initial=0.0) > 1.0:
            array = array / 255.0
        return (
            torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
            .unsqueeze(0)
            .contiguous()
        )

    def _image_to_model_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = self._image_to_raw_tensor(image)
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = self._resize_with_pad(tensor, self.image_size, self.image_size)
        return (tensor * 2.0 - 1.0).contiguous()

    @staticmethod
    def _resize_with_pad(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
        _, _, cur_height, cur_width = images.shape
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized = F.interpolate(
            images,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        resized = resized.clamp(0.0, 1.0)
        pad_h0, rem_h = divmod(height - resized_height, 2)
        pad_w0, rem_w = divmod(width - resized_width, 2)
        return F.pad(
            resized,
            (pad_w0, pad_w0 + rem_w, pad_h0, pad_h0 + rem_h),
            mode="constant",
            value=0.0,
        )

    @staticmethod
    def _extract_state(obs: dict[str, Any]) -> torch.Tensor:
        state = obs.get("states", obs.get("state"))
        if state is None:
            raise KeyError("LIBERO observation must contain 'states' or 'state'.")
        array = np.asarray(state, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        return torch.from_numpy(np.ascontiguousarray(array))

    @staticmethod
    def _extract_task(obs: dict[str, Any]) -> str:
        task = obs.get("task_description", obs.get("task", ""))
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""
        return str(task)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.normalization_mode == "openpi_quantile":
            min_v = self._normalizer_stats.get("observation.state.min")
            max_v = self._normalizer_stats.get("observation.state.max")
            if min_v is None or max_v is None:
                return state
            return (state - min_v.to(state)) / (
                max_v.to(state) - min_v.to(state) + 1e-6
            ) * 2.0 - 1.0
        mean = self._normalizer_stats.get("observation.state.mean")
        std = self._normalizer_stats.get("observation.state.std")
        if mean is None or std is None:
            return state
        return (state - mean.to(state)) / torch.clamp(std.to(state), min=1e-8)

    def _tokenize_inputs(
        self, tasks: list[str], states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prompt_mode == "openpi_task":
            prompts = [
                task.strip().replace("_", " ").replace("\n", " ") + "\n"
                for task in tasks
            ]
        else:
            state_np = states.detach().cpu().numpy()
            bins = np.linspace(-1.0, 1.0, 257)[:-1]
            discretized = np.digitize(state_np, bins=bins) - 1
            discretized = np.clip(discretized, 0, 255)
            prompts = []
            for task, state_bins in zip(tasks, discretized):
                cleaned = task.strip().replace("_", " ").replace("\n", " ")
                state_str = " ".join(map(str, state_bins))
                prompts.append(f"Task: {cleaned}, State: {state_str};\nAction: ")
        encoded = self.tokenizer(
            prompts,
            max_length=int(self.config.get("tokenizer_max_length", 200)),
            padding="max_length",
            padding_side="right",
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(torch.int64), encoded["attention_mask"].sum(
            dim=-1
        ).to(torch.int64)

    def _postprocess_actions(self, raw_actions: torch.Tensor) -> np.ndarray:
        action = raw_actions[..., : self.action_dim].detach().float()
        if not self._use_phyai_compat:
            actions = self.processor.postprocess(action)
            if isinstance(actions, torch.Tensor):
                actions = actions.detach().cpu().numpy()
            return np.asarray(actions, dtype=np.float32)
        action = action.cpu()
        if self.normalization_mode == "openpi_quantile":
            min_v = self._unnormalizer_stats.get("action.min")
            max_v = self._unnormalizer_stats.get("action.max")
            if min_v is not None and max_v is not None:
                action = (action + 1.0) / 2.0 * (
                    max_v.to(action) - min_v.to(action) + 1e-6
                ) + min_v.to(action)
        else:
            mean = self._unnormalizer_stats.get("action.mean")
            std = self._unnormalizer_stats.get("action.std")
            if mean is not None and std is not None:
                action = action * torch.clamp(std.to(action), min=1e-8) + mean.to(
                    action
                )
        return action.numpy().astype(np.float32)

    def infer(
        self, obs: dict[str, Any], *, noise: torch.Tensor | np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        request_kwargs = self.observation_to_request_inputs(obs)
        if noise is not None:
            request_kwargs["noise"] = torch.as_tensor(noise, device=self.device)
        request = PI05Request(**request_kwargs)
        with torch.inference_mode():
            raw_actions = self.engine.step(request)
        actions = self._postprocess_actions(raw_actions)
        return {"actions": actions}

    def close(self) -> None:
        self.engine.close()
