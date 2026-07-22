"""Cosmos3 processors — text-to-video tokenizer + action/policy processor"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline
from phyai_utils_tools.processing.transition import IMAGES, TASK, Transition
from phyai_utils_tools.tokenizer import get_tokenizer

from phyai_utils_tools.models.cosmos3.steps_cosmos3 import (
    ACTION_CHUNK,
    ACTION_START_FRAME_OFFSET,
    CAPTION,
    COND_ACTION,
    COND_ACTION_INDEXES,
    DOMAIN_ID,
    EMBODIMENT_TO_DOMAIN_ID,
    EMBODIMENT_TO_RAW_ACTION_DIM,
    META_HEIGHT,
    META_WIDTH,
    MODE,
    NEG_TEXT_IDS,
    NEG_TEXT_MASK,
    RAW_ACTION_DIM,
    TEXT_IDS,
    TEXT_MASK,
    VIDEO_SHAPE,
    Cosmos3ActionPadStep,
    Cosmos3DomainResolveStep,
    Cosmos3ImagePreprocessStep,
    Cosmos3TextTokenizeStep,
    cosmos3_default_negative_prompt,
    cosmos3_generation_caption,
    resolve_domain_id,
    resolve_raw_action_dim,
)
from phyai_utils_tools.processing.transition import PIXEL_VALUES


COSMOS3_VISION_START_TOKEN = "<|vision_start|>"
COSMOS3_ROBOLAB_CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. "
    "The bottom row contains two horizontally concatenated third-person "
    "perspective views of the scene from opposite sides, with the robot visible."
)


def _flatten_chat_ids(out) -> list[int]:
    """Normalize ``apply_chat_template(tokenize=True)`` output to ``list[int]``.

    Different transformers/tokenizers versions return a ``list[int]``, a nested
    ``[[int, ...]]``, or a ``BatchEncoding`` of ``tokenizers.Encoding`` objects.
    """
    # BatchEncoding / list whose first element exposes ``.ids`` (Encoding).
    first = out[0] if len(out) > 0 else None
    if hasattr(first, "ids"):
        return list(first.ids)
    if isinstance(first, (list, tuple)):
        return [int(x) for x in first]
    # Flat list of ints.
    return [int(x) for x in out]


@dataclass
class Cosmos3TokenizedPrompt:
    """Batch-1 tokenized prompt tensors."""

    text_ids: torch.Tensor  # [1, S] int64
    text_mask: torch.Tensor  # [1, S] int64 (all ones — no padding)


@dataclass
class Cosmos3GenerationOutput:
    """CPU-ready media output from the Cosmos3 generation plugin."""

    frames: torch.Tensor  # [T, H, W, 3] uint8 RGB, CPU
    video: torch.Tensor  # [B, 3, T, H, W] or [3, T, H, W], CPU float in [0, 1]
    waveform: torch.Tensor | None = None  # CPU float in [-1, 1]
    sample_rate: int | None = None


class Cosmos3Processor:
    """Qwen2 chat-template tokenizer for Cosmos3 T2V/I2V/T2AV prompts.

    When the generation dims (``fps``/``num_frames``/``height``/``width``) are given
    and ``append_metadata`` is set, the positive prompt gets native-style
    duration/resolution metadata appended (see :func:`cosmos3_generation_caption`).
    The negative prompt defaults to the native structured "bad video" negative
    (:func:`cosmos3_default_negative_prompt`); pass an explicit string (e.g. ``""``)
    to override.
    """

    def __init__(
        self,
        tokenizer_name_or_path: str,
        *,
        use_system_prompt: bool = False,
        fps: float | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        aspect_ratio: str | None = None,
        append_metadata: bool = True,
        negative_prompt: str | None = None,
    ) -> None:
        self.tokenizer = get_tokenizer(tokenizer_name_or_path)
        self.use_system_prompt = use_system_prompt
        self.fps = fps
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.aspect_ratio = aspect_ratio
        self.append_metadata = bool(append_metadata)
        self._negative_prompt = negative_prompt
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.vision_start_token_id = int(
            self.tokenizer.convert_tokens_to_ids(COSMOS3_VISION_START_TOKEN)
        )

    def _augment(self, prompt: str) -> str:
        """Append native duration/resolution metadata when the dims are known."""
        if not self.append_metadata:
            return prompt
        if None in (self.fps, self.num_frames, self.height, self.width):
            return prompt
        return cosmos3_generation_caption(
            prompt,
            fps=self.fps,
            num_frames=self.num_frames,
            height=self.height,
            width=self.width,
            aspect_ratio=self.aspect_ratio,
        )

    def tokenize(
        self,
        prompt: str,
        *,
        device: torch.device | str = "cpu",
        augment: bool = True,
    ) -> Cosmos3TokenizedPrompt:
        """Tokenize one prompt -> ``[1, S]`` ids + all-ones mask.

        ``augment`` controls whether duration/resolution metadata is appended (on
        for the positive prompt; off for the negative, matching native).
        """
        content = self._augment(prompt) if augment else prompt
        conversation = []
        if self.use_system_prompt:
            conversation.append(
                {
                    "role": "system",
                    "content": "You are a helpful assistant who will generate videos from a given prompt.",
                }
            )
        conversation.append({"role": "user", "content": content})
        out = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True
        )
        ids = _flatten_chat_ids(out)
        ids = ids + [self.eos_token_id, self.vision_start_token_id]
        text_ids = torch.tensor([ids], dtype=torch.long, device=device)
        text_mask = torch.ones_like(text_ids)
        return Cosmos3TokenizedPrompt(text_ids=text_ids, text_mask=text_mask)

    def tokenize_pair(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> tuple[Cosmos3TokenizedPrompt, Cosmos3TokenizedPrompt]:
        """Tokenize the conditional + unconditional prompts.

        ``negative_prompt=None`` falls back to the processor's ``negative_prompt``,
        then to the native structured default. Pass ``""`` for an empty negative.
        The positive prompt is metadata-augmented; the negative is not.
        """
        if negative_prompt is None:
            negative_prompt = self._negative_prompt
        if negative_prompt is None:
            negative_prompt = cosmos3_default_negative_prompt()
        return (
            self.tokenize(prompt, device=device, augment=True),
            self.tokenize(negative_prompt, device=device, augment=False),
        )


class Cosmos3GenerationPostProcessor:
    """Postprocess and save Cosmos3 generation media.

    ``cosmos3`` engine outputs are already VAE-decoded by the plugin:
    video-only requests return pixels in ``[0, 1]`` and T2AV requests return a
    ``{"video", "sound", "sample_rate"}`` dict. This class handles the output-side
    media glue: move tensors to CPU, convert video pixels to uint8 RGB frames, and
    optionally mux video + audio into one mp4 via PyAV.
    """

    def __init__(self, fps: float) -> None:
        self.fps = float(fps)

    @staticmethod
    def _to_uint8_frames(video: torch.Tensor) -> torch.Tensor:
        """Convert ``[1,3,T,H,W]`` or ``[3,T,H,W]`` pixels to CPU uint8 frames."""
        if video.ndim == 5:
            video = video[0]
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(
                "Expected video shaped [1, 3, T, H, W] or [3, T, H, W], got "
                f"{tuple(video.shape)}."
            )
        return (
            (video.clamp(0, 1) * 255)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .cpu()
            .contiguous()
        )

    def postprocess(
        self, output: torch.Tensor | dict[str, torch.Tensor | int]
    ) -> Cosmos3GenerationOutput:
        """Move generation output to CPU and prepare frames for media encoding."""
        if isinstance(output, dict):
            video = output["video"]
            waveform = output.get("sound")
            sample_rate = output.get("sample_rate")
        else:
            video = output
            waveform = None
            sample_rate = None
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"Expected video tensor, got {type(video)!r}.")

        frames = self._to_uint8_frames(video)
        video_cpu = video.detach().cpu()
        waveform_cpu = (
            waveform.detach().clamp(-1.0, 1.0).float().cpu()
            if isinstance(waveform, torch.Tensor)
            else None
        )
        sample_rate_int = int(sample_rate) if sample_rate is not None else None
        return Cosmos3GenerationOutput(
            frames=frames,
            video=video_cpu,
            waveform=waveform_cpu,
            sample_rate=sample_rate_int,
        )

    def save_mp4(
        self,
        output: Cosmos3GenerationOutput,
        path: str | Path,
        *,
        crf: str = "18",
    ) -> None:
        """Encode frames, and optional waveform, into one mp4 via PyAV."""
        import av

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = output.frames.numpy()  # [T, H, W, 3] uint8 RGB
        with av.open(str(path), mode="w") as container:
            video_stream = container.add_stream(
                "h264", rate=Fraction(self.fps).limit_denominator(10000)
            )
            video_stream.width = int(arr.shape[2])
            video_stream.height = int(arr.shape[1])
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {"crf": crf}

            audio_stream = None
            audio_samples = None
            audio_layout = "stereo"
            if output.waveform is not None and output.sample_rate is not None:
                wav = output.waveform
                if wav.ndim == 3:
                    wav = wav[0]
                if wav.ndim == 1:
                    wav = wav.reshape(1, -1)
                if wav.ndim != 2:
                    raise ValueError(
                        "Expected waveform shaped [1, channels, samples], "
                        "[channels, samples], or [samples], got "
                        f"{tuple(output.waveform.shape)}."
                    )
                audio_samples = wav.numpy()
                audio_layout = "stereo" if audio_samples.shape[0] >= 2 else "mono"
                audio_stream = container.add_stream("aac", rate=int(output.sample_rate))
                audio_stream.layout = audio_layout

            for frame_data in arr:
                frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            for packet in video_stream.encode():
                container.mux(packet)

            if audio_stream is not None:
                audio_frame = av.AudioFrame.from_ndarray(
                    audio_samples, format="fltp", layout=audio_layout
                )
                audio_frame.sample_rate = int(output.sample_rate)
                for packet in audio_stream.encode(audio_frame):
                    container.mux(packet)
                for packet in audio_stream.encode():
                    container.mux(packet)


@dataclass
class Cosmos3PolicyProcessedInputs:
    """Preprocessed inputs for the Cosmos3 action/policy path."""

    pixel_values: torch.Tensor
    caption: str
    text_ids: torch.Tensor
    text_mask: torch.Tensor
    neg_text_ids: torch.Tensor
    neg_text_mask: torch.Tensor
    cond_action: torch.Tensor | None
    domain_id: int
    mode: str
    action_chunk: int
    raw_action_dim: int
    video_shape: tuple[int, int, int]
    video_num_frames: int
    content_size: tuple[int, int]
    cond_frame_indexes: tuple[int, ...] | None = None
    cond_action_indexes: tuple[int, ...] = ()
    action_start_frame_offset: int = 1


class Cosmos3PolicyProcessor(BaseModelProcessor):
    """Cosmos3 action/policy pre/post processor.

    Preprocessing: image resize/normalize, text tokenize, action pad, domain resolve.
    Postprocessing: slice action to raw_action_dim, move to CPU.
    """

    def __init__(
        self,
        *,
        tokenizer_name_or_path: str,
        height: int = 480,
        width: int = 832,
        num_frames: int = 17,
        mode: str = "policy",
        domain_name: str | int = "agibotworld",
        action_chunk_size: int = 16,
        raw_action_dim: int | None = None,
        action_dim: int = 64,
        negative_prompt: str = "",
        fps: float = 24.0,
        image_size: int | None = None,
        append_metadata: bool = True,
        prompt_format: str = "plain",
        view_point: str = "ego_view",
        additional_view_description: str | None = None,
        cond_frame_indexes: tuple[int, ...] | None = None,
        action_history_length: int = 0,
        flip_gripper: bool = False,
        action_stats_path: str | None = None,
        action_normalization: str = "minmax",
        device: torch.device | str = "cpu",
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.mode = mode
        self.domain_name = domain_name
        self.action_chunk_size = int(action_chunk_size)
        # note(chenghua): An explicit width is authoritative because one domain ID
        # can serve checkpoints trained with different physical action schemas.
        self.raw_action_dim = self._resolve_raw_action_dim(raw_action_dim, domain_name)
        self.action_dim = int(action_dim)
        self.negative_prompt = negative_prompt
        self.fps = float(fps)
        self.image_size = int(image_size) if image_size is not None else None
        self.append_metadata = bool(append_metadata)
        self.prompt_format = prompt_format
        self.view_point = view_point
        self.additional_view_description = additional_view_description
        self.cond_frame_indexes = (
            tuple(cond_frame_indexes) if cond_frame_indexes is not None else None
        )
        self.action_history_length = int(action_history_length)
        if self.action_history_length < 0:
            raise ValueError("action_history_length must be non-negative.")
        self.flip_gripper = bool(flip_gripper)
        self.action_normalization = action_normalization
        self.device = device
        self.params_dtype = params_dtype
        # Optional output denormalization: tensors built once from an external stats
        # JSON, applied in postprocess.
        self._action_mean: torch.Tensor | None = None
        self._action_std: torch.Tensor | None = None
        self._action_min: torch.Tensor | None = None
        self._action_range: torch.Tensor | None = None
        if action_stats_path is not None:
            self._load_action_stats(action_stats_path)
        super().__init__()

    def _resolve_raw_action_dim(
        self, raw_action_dim: int | None, domain_name: str | int
    ) -> int:
        """Use an explicit raw width or infer it from the embodiment table."""
        if raw_action_dim is None:
            return resolve_raw_action_dim(domain_name)
        return int(raw_action_dim)

    @staticmethod
    def _flip_gripper_channel(action: Any) -> torch.Tensor:
        """Copy an action tensor and invert its final gripper channel."""
        tensor = torch.as_tensor(action, dtype=torch.float32).clone()
        if tensor.ndim not in (2, 3) or tensor.shape[-1] < 1:
            raise ValueError(
                "Gripper flipping expects action shaped [T,D] or [B,T,D], got "
                f"{tuple(tensor.shape)}."
            )
        tensor[..., -1] = 1.0 - tensor[..., -1]
        return tensor

    def _load_action_stats(self, stats_path: str) -> None:
        """Load output-denormalization tensors from an external stats JSON.

        The JSON is a dict, optionally with a ``"global"`` (or ``"global_raw"`` for
        ``quantile_rot``) block. ``meanstd`` uses ``mean``/``std``; ``minmax`` uses
        ``min``/``max``; ``quantile``/``quantile_rot`` use ``q01``/``q99``.
        """
        import json

        with open(stats_path) as f:
            raw_stats = json.load(f)
        if not isinstance(raw_stats, dict):
            raise ValueError(f"Action stats file must contain a dict: {stats_path}")
        stats_key = (
            "global_raw" if self.action_normalization == "quantile_rot" else "global"
        )
        stats = raw_stats.get(stats_key, raw_stats)
        if self.action_normalization == "meanstd":
            self._action_mean = torch.tensor(stats["mean"], dtype=torch.float32)
            self._action_std = torch.clamp(
                torch.tensor(stats["std"], dtype=torch.float32), min=1e-8
            )
        elif self.action_normalization in ("quantile", "quantile_rot"):
            self._action_min = torch.tensor(stats["q01"], dtype=torch.float32)
            q99 = torch.tensor(stats["q99"], dtype=torch.float32)
            self._action_range = torch.clamp(q99 - self._action_min, min=1e-6)
        elif self.action_normalization == "minmax":
            self._action_min = torch.tensor(stats["min"], dtype=torch.float32)
            amax = torch.tensor(stats["max"], dtype=torch.float32)
            self._action_range = torch.clamp(amax - self._action_min, min=1e-6)
        else:
            raise ValueError(
                "action_normalization must be one of 'meanstd', 'minmax', "
                f"'quantile', 'quantile_rot'; got {self.action_normalization!r}."
            )

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Invert the configured normalization on the raw-dim action channels.

        Slice to the stats width, then ``x*std+mean`` (meanstd) or
        ``(x+1)/2*range+min`` (minmax / quantile). No-op when no stats are loaded.
        """
        if self._action_mean is not None and self._action_std is not None:
            dim = self._action_mean.shape[0]
            mean = self._action_mean.to(action.device)
            std = self._action_std.to(action.device)
            return action[..., :dim] * std + mean
        if self._action_min is not None and self._action_range is not None:
            dim = self._action_min.shape[0]
            amin = self._action_min.to(action.device)
            arange = self._action_range.to(action.device)
            return (action[..., :dim] + 1.0) / 2.0 * arange + amin
        return action

    def _to_transition(self, raw: dict[str, Any]) -> Transition:
        """Adapt caller's raw dict into the canonical transition."""
        t: Transition = {}
        t[IMAGES] = raw.get("images")
        t[TASK] = raw.get("task", raw.get("prompt", ""))
        cond_action = raw.get("action")
        if cond_action is None:
            cond_action = raw.get("cond_action")
        if cond_action is not None and self.flip_gripper:
            cond_action = self._flip_gripper_channel(cond_action)
        t[COND_ACTION] = cond_action
        t[DOMAIN_ID] = raw.get("domain_name", raw.get("domain_id", self.domain_name))
        t[MODE] = raw.get("mode", self.mode)
        return t

    def _to_output(self, transition: Transition) -> Cosmos3PolicyProcessedInputs:
        """Extract typed output from the final transition."""
        return Cosmos3PolicyProcessedInputs(
            pixel_values=transition[PIXEL_VALUES],
            caption=transition[CAPTION],
            text_ids=transition[TEXT_IDS],
            text_mask=transition[TEXT_MASK],
            neg_text_ids=transition[NEG_TEXT_IDS],
            neg_text_mask=transition[NEG_TEXT_MASK],
            cond_action=transition.get(COND_ACTION),
            domain_id=transition[DOMAIN_ID],
            mode=transition[MODE],
            action_chunk=transition[ACTION_CHUNK],
            raw_action_dim=transition[RAW_ACTION_DIM],
            video_shape=transition[VIDEO_SHAPE],
            video_num_frames=self.action_chunk_size + 1,
            content_size=(transition[META_HEIGHT], transition[META_WIDTH]),
            cond_frame_indexes=self.cond_frame_indexes,
            cond_action_indexes=transition[COND_ACTION_INDEXES],
            action_start_frame_offset=transition[ACTION_START_FRAME_OFFSET],
        )

    def build_preprocessor(self) -> ProcessorPipeline:
        steps = [
            Cosmos3ImagePreprocessStep(
                height=self.height,
                width=self.width,
                mode=self.mode,
                image_size=self.image_size,
            ),
            Cosmos3TextTokenizeStep(
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                negative_prompt=self.negative_prompt,
                append_metadata=self.append_metadata,
                prompt_format=self.prompt_format,
                view_point=self.view_point,
                additional_view_description=self.additional_view_description,
                fps=self.fps,
                num_frames=self.action_chunk_size + 1,
            ),
            Cosmos3ActionPadStep(
                action_chunk_size=self.action_chunk_size,
                raw_action_dim=self.raw_action_dim,
                action_dim=self.action_dim,
                mode=self.mode,
                action_history_length=self.action_history_length,
            ),
            Cosmos3DomainResolveStep(),
        ]
        return ProcessorPipeline(
            steps=steps,
            name="cosmos3_policy_preprocessor",
            to_transition=self._to_transition,
            to_output=self._to_output,
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            steps=[],
            name="cosmos3_policy_postprocessor",
            to_transition=lambda x: x,
            to_output=lambda x: x,
        )

    def postprocess(self, output: dict[str, Any] | torch.Tensor) -> dict[str, Any]:
        """Slice action to raw_action_dim, optionally denormalize, move to CPU.

        When action stats were loaded (``action_stats_path``), the sliced action is
        denormalized back to physical units before moving to CPU; otherwise it is
        returned in the model's (normalized) action space.
        """
        if isinstance(output, torch.Tensor):
            action = self._postprocess_action(output)
            return {"action": action.cpu()}
        result: dict[str, Any] = {}
        if "action" in output:
            action = self._postprocess_action(output["action"])
            result["action"] = action.cpu()
        if "pixels" in output:
            result["pixels"] = output["pixels"].cpu()
        if "video" in output:
            result["video"] = output["video"].cpu()
        return result

    def _postprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        """Remove clean history, denormalize, and restore gripper convention."""
        action = action[:, self.action_history_length :, : self.raw_action_dim]
        action = self._denormalize_action(action)
        if self.flip_gripper:
            action = self._flip_gripper_channel(action)
        return action


class Cosmos3RoboLabPolicyProcessor(Cosmos3PolicyProcessor):
    """Native-compatible adapter for the Cosmos3 RoboLab/OpenPI request schema.

    The adapter composes the optional three-camera observation, performs the
    server's fixed 540x640 bilinear resize, conditions on joint-position history,
    and restores the external gripper convention on output.
    """

    def __init__(
        self,
        *,
        tokenizer_name_or_path: str,
        format_prompt_as_json: bool = False,
        image_height: int = 540,
        image_width: int = 640,
        action_chunk_size: int = 32,
        history_length: int = 1,
        domain_name: str | int = "droid_lerobot",
        raw_action_dim: int = 8,
        action_dim: int = 64,
        fps: float = 15.0,
        negative_prompt: str = "",
        device: torch.device | str = "cpu",
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.robolab_image_height = int(image_height)
        self.robolab_image_width = int(image_width)
        super().__init__(
            tokenizer_name_or_path=tokenizer_name_or_path,
            height=544,
            width=736,
            num_frames=action_chunk_size + 1,
            mode="policy",
            domain_name=domain_name,
            action_chunk_size=action_chunk_size,
            raw_action_dim=raw_action_dim,
            action_dim=action_dim,
            negative_prompt=negative_prompt,
            fps=fps,
            image_size=480,
            append_metadata=True,
            prompt_format="json" if format_prompt_as_json else "plain",
            view_point="concat_view",
            additional_view_description=COSMOS3_ROBOLAB_CONCAT_VIEW_DESCRIPTION,
            cond_frame_indexes=(0,),
            action_history_length=history_length,
            flip_gripper=True,
            device=device,
            params_dtype=params_dtype,
        )

    @staticmethod
    def _as_rgb_uint8(value: Any, key: str) -> np.ndarray:
        """Validate one RoboLab RGB observation."""
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key!r} must have shape [H,W,3], got {image.shape}.")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    @staticmethod
    def _resize_rgb_uint8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Match the RoboLab server's float bilinear resize and uint8 cast."""
        import torch.nn.functional as F

        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

    def _compose_roboarena_views(self, raw: dict[str, Any]) -> np.ndarray | None:
        """Compose wrist-over-two-exterior cameras using the native layout."""
        keys = (
            "observation/wrist_image_left",
            "observation/exterior_image_1_left",
            "observation/exterior_image_2_left",
        )
        if not all(key in raw for key in keys):
            return None
        wrist = self._as_rgb_uint8(raw[keys[0]], keys[0])
        left = self._as_rgb_uint8(raw[keys[1]], keys[1])
        right = self._as_rgb_uint8(raw[keys[2]], keys[2])
        half_size = (wrist.shape[0] // 2, wrist.shape[1] // 2)
        left = self._resize_rgb_uint8(left, half_size)
        right = self._resize_rgb_uint8(right, half_size)
        return np.concatenate([wrist, np.concatenate([left, right], axis=1)], axis=0)

    def _extract_robolab_image(self, raw: dict[str, Any]) -> np.ndarray:
        """Extract the direct or composed RoboLab observation image."""
        if "observation/image" in raw:
            return self._as_rgb_uint8(raw["observation/image"], "observation/image")
        if "images" in raw:
            return self._as_rgb_uint8(raw["images"], "images")
        image = self._compose_roboarena_views(raw)
        if image is not None:
            return image
        raise ValueError(
            "RoboLab input requires 'observation/image', 'images', or all three "
            "wrist/exterior camera keys."
        )

    @staticmethod
    def _as_history(value: Any, key: str, width: int | None = None) -> np.ndarray:
        """Convert a RoboLab state history to contiguous float32 rows."""
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or (width is not None and array.shape[-1] != width):
            expected = f"[T,{width}]" if width is not None else "[T,D]"
            raise ValueError(f"{key!r} must have shape {expected}, got {array.shape}.")
        return np.ascontiguousarray(array)

    def _extract_joint_history(self, raw: dict[str, Any]) -> np.ndarray:
        """Build oldest-to-newest joint/gripper rows in external convention."""
        length = self.action_history_length
        joints = self._as_history(
            raw["observation/joint_position"],
            "observation/joint_position",
            self.raw_action_dim - 1,
        )
        gripper = np.asarray(raw["observation/gripper_position"], dtype=np.float32)
        if gripper.ndim == 0:
            gripper = gripper.reshape(1, 1)
        elif gripper.ndim == 1:
            gripper = gripper[:, None]
        if gripper.ndim != 2 or gripper.shape[-1] != 1:
            raise ValueError(
                "'observation/gripper_position' must have shape [T,1], [T], or "
                f"scalar, got {gripper.shape}."
            )
        if len(joints) < length or len(gripper) < length:
            raise ValueError(
                f"RoboLab history_length={length} requires at least {length} state rows."
            )
        return np.concatenate([joints[-length:], gripper[-length:]], axis=-1)

    def preprocess(self, raw: dict[str, Any]) -> Cosmos3PolicyProcessedInputs:
        """Adapt one RoboLab/OpenPI observation and run the common processor."""
        image = self._extract_robolab_image(raw)
        size = (self.robolab_image_height, self.robolab_image_width)
        if image.shape[:2] != size:
            image = self._resize_rgb_uint8(image, size)

        adapted: dict[str, Any] = {
            "images": image,
            "task": raw.get("prompt", raw.get("task", "")),
            "domain_name": raw.get("domain_name", self.domain_name),
        }
        cond_action = raw.get("cond_action")
        if cond_action is None and self.action_history_length > 0:
            cond_action = self._extract_joint_history(raw)
        if cond_action is not None:
            adapted["cond_action"] = cond_action
        return super().preprocess(adapted)


__all__ = [
    "Cosmos3GenerationOutput",
    "Cosmos3GenerationPostProcessor",
    "Cosmos3PolicyProcessedInputs",
    "Cosmos3PolicyProcessor",
    "Cosmos3RoboLabPolicyProcessor",
    "Cosmos3Processor",
    "Cosmos3TokenizedPrompt",
    "COSMOS3_VISION_START_TOKEN",
    "COSMOS3_ROBOLAB_CONCAT_VIEW_DESCRIPTION",
    "EMBODIMENT_TO_DOMAIN_ID",
    "EMBODIMENT_TO_RAW_ACTION_DIM",
    "cosmos3_default_negative_prompt",
    "cosmos3_generation_caption",
    "resolve_domain_id",
    "resolve_raw_action_dim",
]
