"""GR00T-N1.7 native model boundaries.

This module intentionally mirrors the decomposition in Isaac-GR00T N1.7
without importing the reference implementation:

* Qwen3-VL / Cosmos backbone: produces V-L token features.
* Action head: state encoder, action encoder, optional VL self-attn,
  DiT, action decoder.
* Top-level container: owns parameters only; runners/scheduler own
  runtime state, random noise, denoising loops, and graph capture.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.layers.attention import Attention
from phyai.layers.layer_norm import LayerNorm as PhyAILayerNorm
from phyai.layers.linear.layers import ReplicatedLinear
from phyai.models.gr00t_n17.configuration_gr00t_n17 import (
    GR00TN17Config,
    GR00TN17DiTConfig,
    GR00TN17VLSelfAttentionConfig,
)
from phyai.models.gr00t_n17.qwen3_vl_adapter import GR00TN17Qwen3VLBackbone
from phyai.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from phyai.utils import load_config
from phyai.weights.shards import replicated


# ============================================================================ #
# Shared primitives / PhyAI wrappers                                            #
# ============================================================================ #


def _resolve_backbone_config_dir(
    model_name_or_path: str,
    *,
    local_files_only: bool = False,
    revision: str | None = None,
) -> str:
    """Return a local directory holding the backbone ``config.json``.

    A local directory is used as-is. Otherwise the path is treated as a
    HuggingFace repo id and resolved against the local hub cache via
    ``huggingface_hub`` (no ``transformers`` model machinery). With
    ``local_files_only`` the cached snapshot is used without any network
    access.
    """
    from pathlib import Path

    candidate = Path(model_name_or_path)
    if candidate.is_dir():
        return str(candidate)

    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_name_or_path,
        revision=revision,
        allow_patterns=["config.json"],
        local_files_only=local_files_only,
    )


class GR00TN17NativeImplementationError(NotImplementedError):
    """Raised when a GR00T-N1.7 native submodule has not been ported yet."""


@dataclass(frozen=True)
class GR00TN17BackboneOutput:
    """Backbone output consumed by the action head.

    ``image_mask`` keeps the historical field name, but it is a visual-token
    mask: image tokens and video tokens are both ``True`` when present.
    """

    backbone_features: torch.Tensor
    backbone_attention_mask: torch.Tensor
    image_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class GR00TN17ActionInput:
    """Action-head inputs after preprocessing.

    ``action_mask`` is applied to the final normalized action chunk. Supported
    shapes cover valid action dimensions and/or horizon steps:
    ``(action_dim,)``, ``(B, action_dim)``, ``(B, action_horizon)``, and
    ``(B, action_horizon, action_dim)``.
    """

    state: torch.Tensor
    embodiment_id: torch.Tensor
    action_mask: torch.Tensor | None = None
    action: torch.Tensor | None = None


class GR00TN17Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GR00TN17Dropout(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.p
        mask = torch.empty_like(x).bernoulli_(keep_prob)
        return x * mask / keep_prob


class GR00TN17LayerNorm(PhyAILayerNorm):
    def __init__(
        self,
        normalized_shape: int,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        if elementwise_affine:
            super().__init__(
                int(normalized_shape),
                eps=eps,
                backend="phyai-kernel",
                bias=True,
                prefix="",
            )
            self.elementwise_affine = True
            return

        nn.Module.__init__(self)
        self.elementwise_affine = False
        self.backend = "phyai-kernel"
        self.hidden_size = int(normalized_shape)
        self.variance_epsilon = float(eps)
        self.has_bias = False
        self.prefix = ""
        self.register_buffer("weight", torch.ones(self.hidden_size), persistent=False)
        self.register_buffer("_zero_beta", None, persistent=False)
        self._layernorm = self._load_kernel(self.backend)

    def _apply(self, fn):
        super()._apply(fn)
        if not self.elementwise_affine:
            weight = self._buffers.get("weight")
            if weight is not None and weight.dtype != torch.float32:
                self._buffers["weight"] = weight.float()
        return self

    @staticmethod
    def _torch_layer_norm(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        var = (x_float - mean).square().mean(dim=-1, keepdim=True)
        out = (x_float - mean) * torch.rsqrt(var + eps)
        out = out.to(dtype=x.dtype)
        if weight is not None:
            out = out * weight.to(device=x.device, dtype=x.dtype)
        if bias is not None:
            out = out + bias.to(device=x.device, dtype=x.dtype)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            weight = self.weight if self.elementwise_affine else None
            bias = self.bias if self.elementwise_affine else None
            return self._torch_layer_norm(x, weight, bias, self.variance_epsilon)

        if self.elementwise_affine:
            return super().forward(x)

        needs_reshape = x.dim() != 2
        if needs_reshape:
            orig_shape = x.shape
            x = x.contiguous().reshape(-1, orig_shape[-1])
        if self.weight.device != x.device or self.weight.dtype != torch.float32:
            raise RuntimeError(
                "non-affine GR00TN17LayerNorm weight must be moved to the input "
                "device and kept in fp32 before forward."
            )
        out = self._layernorm(
            x,
            self.weight,
            None,
            self.variance_epsilon,
        )
        if needs_reshape:
            out = out.reshape(orig_shape)
        return out


class GR00TN17ReplicatedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim))
        self.weight.hf_keys = []
        self.weight.weight_loader = replicated()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]


# ============================================================================ #
# Qwen3-VL backbone boundary                                                    #
# ============================================================================ #


class GR00TN17Backbone(nn.Module):
    """Qwen3-VL / Cosmos-Reason2 backbone boundary."""

    def __init__(
        self,
        config: GR00TN17Config,
        *,
        qwen3vl_model: nn.Module | None = None,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        transformers_loading_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config.backbone
        self.params_dtype = params_dtype
        self.target_device = torch.device(device) if device is not None else None
        self.transformers_loading_kwargs = dict(transformers_loading_kwargs or {})
        self.qwen3vl_model = self._build_qwen3vl_model(qwen3vl_model)

    def prepare_input(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return dict(batch)

    def _truncate_language_layers(self, qwen3vl_model: nn.Module) -> None:
        if self.config.select_layer < 0:
            return
        language_model = getattr(qwen3vl_model, "language_model", None)
        if language_model is None:
            nested_model = getattr(qwen3vl_model, "model", None)
            language_model = getattr(nested_model, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers is None:
            return
        while len(layers) > self.config.select_layer:
            layers.pop(-1)

    def _attach_qwen3vl_weight_keys(self, qwen3vl_model: nn.Module) -> None:
        for name, param in qwen3vl_model.named_parameters():
            _attach_replicated_hf_key(param, f"backbone.model.{name}")

    def _build_qwen3vl_model(self, qwen3vl_model: nn.Module | None) -> nn.Module:
        if qwen3vl_model is not None:
            self._truncate_language_layers(qwen3vl_model)
            self._attach_qwen3vl_weight_keys(qwen3vl_model)
            return qwen3vl_model

        qwen3vl_config = self.config.qwen3vl
        if qwen3vl_config is None:
            backbone_dir = _resolve_backbone_config_dir(
                self.config.model_name,
                local_files_only=self.transformers_loading_kwargs.get(
                    "local_files_only", False
                ),
                revision=self.config.model_revision,
            )
            qwen3vl_config = load_config(backbone_dir, Qwen3VLConfig)
        dtype = (
            torch.bfloat16
            if self.config.load_bf16
            else self.params_dtype or torch.get_default_dtype()
        )
        return GR00TN17Qwen3VLBackbone(
            qwen3vl_config,
            select_layer=self.config.select_layer,
            params_dtype=dtype,
            device=self.target_device,
            attention_backend=self.config.attention_backend,
        ).eval()

    def _load_qwen3vl_model(self) -> nn.Module:
        """Return the already-built Qwen3-VL backbone.

        Kept for existing callers that need to force construction before weight
        loading; construction now happens in ``__init__`` so this method has no
        forward-time side effects.
        """
        return self.qwen3vl_model

    def prepare_position_ids(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        vl_input = self.prepare_input(batch)
        required_keys = ("input_ids", "attention_mask")
        missing = [key for key in required_keys if key not in vl_input]
        if missing:
            raise KeyError(f"backbone position-id inputs missing keys: {missing}")
        has_image = "image_grid_thw" in vl_input
        has_video = "video_grid_thw" in vl_input
        if not has_image and not has_video:
            raise KeyError(
                "backbone position-id inputs require image_grid_thw, "
                "video_grid_thw, or both."
            )
        qwen3vl_model = self._load_qwen3vl_model()
        native_model = getattr(qwen3vl_model, "model", None)
        if native_model is None or not hasattr(native_model, "get_rope_index"):
            return None
        input_ids = vl_input["input_ids"]
        mm_token_type_ids = vl_input.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            image_token_id = qwen3vl_model.config.image_token_id
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
            mm_token_type_ids = mm_token_type_ids.masked_fill(
                input_ids == image_token_id, 1
            )
            video_token_id = getattr(qwen3vl_model.config, "video_token_id", None)
            if video_token_id is not None:
                mm_token_type_ids = mm_token_type_ids.masked_fill(
                    input_ids == int(video_token_id), 2
                )
        position_ids, _ = native_model.get_rope_index(
            input_ids,
            image_grid_thw=vl_input.get("image_grid_thw"),
            video_grid_thw=vl_input.get("video_grid_thw"),
            attention_mask=vl_input["attention_mask"],
            mm_token_type_ids=mm_token_type_ids,
        )
        return position_ids

    def _prepare_model_inputs(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        vl_input = self.prepare_input(inputs)
        required_keys = ("input_ids", "attention_mask")
        optional_keys = (
            "mm_token_type_ids",
            "position_ids",
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        )
        missing = [key for key in required_keys if key not in vl_input]
        if missing:
            raise KeyError(f"backbone inputs missing keys: {missing}")
        has_image = "pixel_values" in vl_input or "image_grid_thw" in vl_input
        has_video = "pixel_values_videos" in vl_input or "video_grid_thw" in vl_input
        if has_image and not {"pixel_values", "image_grid_thw"} <= vl_input.keys():
            raise KeyError(
                "backbone image inputs require both pixel_values and image_grid_thw."
            )
        if (
            has_video
            and not {"pixel_values_videos", "video_grid_thw"} <= vl_input.keys()
        ):
            raise KeyError(
                "backbone video inputs require both pixel_values_videos and "
                "video_grid_thw."
            )
        if not has_image and not has_video:
            raise KeyError(
                "backbone inputs require image tensors, video tensors, or both."
            )
        model_inputs = {key: vl_input[key] for key in required_keys}
        model_inputs.update(
            {key: vl_input[key] for key in optional_keys if key in vl_input}
        )
        return model_inputs

    def _build_output(
        self,
        backbone_features: torch.Tensor,
        model_inputs: dict[str, torch.Tensor],
    ) -> GR00TN17BackboneOutput:
        qwen3vl_model = self._load_qwen3vl_model()
        if "mm_token_type_ids" in model_inputs:
            visual_mask = model_inputs["mm_token_type_ids"] != 0
        else:
            image_token_id = qwen3vl_model.config.image_token_id
            visual_mask = model_inputs["input_ids"] == image_token_id
            video_token_id = getattr(qwen3vl_model.config, "video_token_id", None)
            if video_token_id is not None:
                visual_mask = visual_mask | (
                    model_inputs["input_ids"] == int(video_token_id)
                )
        return GR00TN17BackboneOutput(
            backbone_features=backbone_features,
            backbone_attention_mask=model_inputs["attention_mask"] == 1,
            image_mask=visual_mask,
        )

    def backbone_graph_plan(self, inputs: dict[str, torch.Tensor]):
        del inputs
        return None

    def build_graph_output(
        self,
        backbone_features: torch.Tensor,
        model_inputs: dict[str, torch.Tensor],
    ) -> GR00TN17BackboneOutput:
        """Wrap a captured-core ``pre_norm_hidden_state`` into a backbone output."""
        return self._build_output(backbone_features, model_inputs)

    def forward(self, inputs: dict[str, torch.Tensor]) -> GR00TN17BackboneOutput:
        model_inputs = self._prepare_model_inputs(inputs)
        outputs = self._load_qwen3vl_model()(**model_inputs)
        # Native Qwen3-VL returns the pre-final-norm tensor directly when the
        # GR00T adapter requests it. Keep the object fallback for injected test
        # doubles and older backbone wrappers.
        if torch.is_tensor(outputs):
            backbone_features = outputs
        else:
            backbone_features = getattr(outputs, "pre_norm_hidden_state", None)
            if backbone_features is None:
                backbone_features = outputs.hidden_states[-1]
        return self._build_output(backbone_features, model_inputs)


# ============================================================================ #
# Action head: encoders + DiT + decoder                                         #
# ============================================================================ #


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
        )
    )


def _pad_last_dim(x: torch.Tensor, pad_right: int) -> torch.Tensor:
    if pad_right == 0:
        return x
    pad_shape = (*x.shape[:-1], pad_right)
    return torch.cat((x, x.new_zeros(pad_shape)), dim=-1)


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.dim() < 2:
        raise ValueError("fan in and fan out require a tensor with at least 2 dims.")
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        for size in tensor.shape[2:]:
            receptive_field_size *= size
    return (
        num_input_fmaps * receptive_field_size,
        num_output_fmaps * receptive_field_size,
    )


def _init_kaiming_uniform_(tensor: torch.Tensor, a: float) -> None:
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    gain = math.sqrt(2.0 / (1.0 + a**2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-bound, bound)


def _init_uniform_(tensor: torch.Tensor, low: float, high: float) -> None:
    with torch.no_grad():
        tensor.uniform_(low, high)


def _init_normal_(tensor: torch.Tensor, mean: float, std: float) -> None:
    with torch.no_grad():
        tensor.normal_(mean=mean, std=std)


def _init_zeros_(tensor: torch.Tensor) -> None:
    with torch.no_grad():
        tensor.zero_()


def _attach_replicated_hf_key(param: nn.Parameter, hf_key: str) -> None:
    param.hf_keys = [(hf_key, None)]
    param.weight_loader = replicated()


def _check_integer_category_ids(cat_ids: torch.Tensor, *, name: str) -> None:
    if (
        torch.is_floating_point(cat_ids)
        or torch.is_complex(cat_ids)
        or cat_ids.dtype == torch.bool
    ):
        raise TypeError(f"{name} must be an integer tensor, got {cat_ids.dtype}.")


def _invalid_category_values(
    cat_ids: torch.Tensor,
    *,
    num_categories: int,
) -> torch.Tensor:
    bad = (cat_ids < 0) | (cat_ids >= num_categories)
    return cat_ids[bad].detach().cpu()


def _replicated_linear(linear: ReplicatedLinear, x: torch.Tensor) -> torch.Tensor:
    out, bias = linear(x)
    if bias is not None:
        return out + bias
    return out


class GR00TN17Linear(ReplicatedLinear):
    """Replicated PhyAI linear with the same tensor-return API as Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias, prefix="")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is None:
            return
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        _init_uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, bias = super().forward(x)
        if bias is not None:
            return out + bias
        return out


def _gr00t_n17_action_head_hf_key(local_name: str, prefix: str) -> str:
    """Map native module parameter names to Isaac-GR00T checkpoint keys."""
    hf_name = local_name
    if hf_name.startswith("model.timestep_encoder."):
        hf_name = hf_name.replace(
            "model.timestep_encoder.linear_",
            "model.timestep_encoder.timestep_embedder.linear_",
            1,
        )
    hf_name = hf_name.replace(".attn1.to_out.", ".attn1.to_out.0.")
    hf_name = hf_name.replace(".ff.fc1.", ".ff.net.0.proj.")
    hf_name = hf_name.replace(".ff.fc2.", ".ff.net.2.")
    return f"{prefix}.{hf_name}" if prefix else hf_name


def _timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    *,
    max_period: int = 10000,
    downscale_freq_shift: float = 1.0,
    flip_sin_to_cos: bool = True,
) -> torch.Tensor:
    """Diffusers-style sinusoidal timestep features."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half, device=timesteps.device, dtype=torch.float32
    )
    exponent = exponent / (half - downscale_freq_shift)
    args = timesteps.float()[:, None] * torch.exp(exponent)[None]
    emb = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat((emb[:, half:], emb[:, :half]), dim=-1)
    if dim % 2 == 1:
        emb = _pad_last_dim(emb, 1)
    return emb


class GR00TN17SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding for action timesteps."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 2:
            raise ValueError(
                "timesteps must have shape (batch, sequence), got "
                f"{tuple(timesteps.shape)}."
            )
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent * (math.log(10000.0) / half_dim)
        freqs = timesteps.float().unsqueeze(-1) * exponent.exp()
        enc = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=-1)
        if self.embedding_dim % 2 == 1:
            enc = _pad_last_dim(enc, 1)
        return enc


class GR00TN17CategorySpecificLinear(nn.Module):
    """Linear layer with independent weights per embodiment id."""

    def __init__(self, num_categories: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.num_categories = int(num_categories)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.linears = nn.ModuleList(
            [
                ReplicatedLinear(
                    self.input_dim,
                    self.output_dim,
                    bias=True,
                    prefix="",
                )
                for _ in range(self.num_categories)
            ]
        )
        for linear in self.linears:
            _init_normal_(linear.weight, mean=0.0, std=0.02)
            if linear.bias is not None:
                _init_zeros_(linear.bias)

    def attach_hf_keys(self, prefix: str) -> None:
        """Load official stacked W/b tensors into per-embodiment linears."""

        def _load_weight(
            _param: nn.Parameter, tensor: torch.Tensor, _shard_id: object
        ) -> None:
            if tuple(tensor.shape) != (
                self.num_categories,
                self.input_dim,
                self.output_dim,
            ):
                raise ValueError(
                    f"{prefix}.W shape mismatch: expected "
                    f"{(self.num_categories, self.input_dim, self.output_dim)}, "
                    f"got {tuple(tensor.shape)}."
                )
            for cat_id, linear in enumerate(self.linears):
                linear.weight.data.copy_(
                    tensor[cat_id].T.to(
                        device=linear.weight.device,
                        dtype=linear.weight.dtype,
                    )
                )

        def _load_bias(
            _param: nn.Parameter, tensor: torch.Tensor, _shard_id: object
        ) -> None:
            if tuple(tensor.shape) != (self.num_categories, self.output_dim):
                raise ValueError(
                    f"{prefix}.b shape mismatch: expected "
                    f"{(self.num_categories, self.output_dim)}, got {tuple(tensor.shape)}."
                )
            for cat_id, linear in enumerate(self.linears):
                if linear.bias is None:
                    continue
                linear.bias.data.copy_(
                    tensor[cat_id].to(
                        device=linear.bias.device,
                        dtype=linear.bias.dtype,
                    )
                )

        first = self.linears[0]
        first.weight.hf_keys = [(f"{prefix}.W", None)]
        first.weight.weight_loader = _load_weight
        if first.bias is not None:
            first.bias.hf_keys = [(f"{prefix}.b", None)]
            first.bias.weight_loader = _load_bias

    def forward(
        self,
        x: torch.Tensor,
        cat_ids: torch.Tensor,
        *,
        static_cat_ids: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, T, C), got {tuple(x.shape)}.")
        if cat_ids.ndim != 1 or cat_ids.shape[0] != x.shape[0]:
            raise ValueError(
                "cat_ids must have shape (B,), got "
                f"{tuple(cat_ids.shape)} for batch {x.shape[0]}."
            )
        _check_integer_category_ids(cat_ids, name="cat_ids")
        if static_cat_ids is not None:
            if len(static_cat_ids) != x.shape[0]:
                raise ValueError(
                    "static category id count must match batch size: "
                    f"{len(static_cat_ids)} vs {x.shape[0]}."
                )
            if any(
                cat_id < 0 or cat_id >= self.num_categories for cat_id in static_cat_ids
            ):
                raise ValueError(
                    f"static category ids must be in [0, {self.num_categories})."
                )
            first_cat = static_cat_ids[0]
            if all(cat_id == first_cat for cat_id in static_cat_ids):
                return _replicated_linear(self.linears[first_cat], x)
            outs = [
                _replicated_linear(self.linears[cat_id], x[row : row + 1])
                for row, cat_id in enumerate(static_cat_ids)
            ]
            return torch.cat(outs, dim=0)
        cat_ids = cat_ids.long()
        invalid = _invalid_category_values(
            cat_ids,
            num_categories=self.num_categories,
        )
        if invalid.numel() > 0:
            raise ValueError(
                f"cat_ids must be in [0, {self.num_categories}); got invalid "
                f"values: {invalid.tolist()}."
            )
        out = x.new_empty(x.shape[0], x.shape[1], self.output_dim)
        for cat_id, linear in enumerate(self.linears):
            indices = torch.nonzero(cat_ids == cat_id, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            cat_x = x.index_select(0, indices)
            cat_out = _replicated_linear(linear, cat_x)
            out.index_copy_(0, indices, cat_out)
        return out


class GR00TN17CategorySpecificMLP(nn.Module):
    """Two-layer embodiment-conditioned MLP."""

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.layer1 = GR00TN17CategorySpecificLinear(
            num_categories, input_dim, hidden_dim
        )
        self.layer2 = GR00TN17CategorySpecificLinear(
            num_categories, hidden_dim, output_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        cat_ids: torch.Tensor,
        *,
        static_cat_ids: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        hidden = self.layer1(x, cat_ids, static_cat_ids=static_cat_ids)
        return self.layer2(torch.relu(hidden), cat_ids, static_cat_ids=static_cat_ids)


class GR00TN17MultiEmbodimentActionEncoder(nn.Module):
    """Action encoder with embodiment-specific projections and time features."""

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.W1 = GR00TN17CategorySpecificLinear(
            num_embodiments, action_dim, hidden_size
        )
        self.W2 = GR00TN17CategorySpecificLinear(
            num_embodiments, 2 * hidden_size, hidden_size
        )
        self.W3 = GR00TN17CategorySpecificLinear(
            num_embodiments, hidden_size, hidden_size
        )
        self.pos_encoding = GR00TN17SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
        *,
        static_cat_ids: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(
                f"actions must have shape (B, T, C), got {tuple(actions.shape)}."
            )
        batch, horizon, _ = actions.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != batch:
            raise ValueError(
                "timesteps must have shape (B,), got "
                f"{tuple(timesteps.shape)} for batch {batch}."
            )
        a_emb = self.W1(actions, cat_ids, static_cat_ids=static_cat_ids)
        tau = timesteps[:, None].expand(batch, horizon)
        tau_emb = self.pos_encoding(tau).to(dtype=a_emb.dtype)
        x = torch.cat((a_emb, tau_emb), dim=-1)
        x = _swish(self.W2(x, cat_ids, static_cat_ids=static_cat_ids))
        return self.W3(x, cat_ids, static_cat_ids=static_cat_ids)


class GR00TN17TimestepEncoder(nn.Module):
    """Sinusoidal timestep projection followed by a two-layer MLP."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.linear_1 = GR00TN17Linear(256, embedding_dim)
        self.linear_2 = GR00TN17Linear(embedding_dim, embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = self.linear_1.weight.dtype
        emb = _timestep_embedding(timesteps, 256).to(dtype=dtype)
        return self.linear_2(_swish(self.linear_1(emb)))


class GR00TN17AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm conditioned on timestep embeddings."""

    def __init__(self, embedding_dim: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.linear = GR00TN17Linear(embedding_dim, 2 * embedding_dim)
        self.norm = GR00TN17LayerNorm(
            embedding_dim, eps=norm_eps, elementwise_affine=False
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(_swish(temb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class GR00TN17Attention(nn.Module):
    """Small backend-selectable attention module for DiT self/cross attention.

    The default backend is ``"sdpa"`` (phyai's recommended path, CUDA-graph
    captureable). ``"eager"`` is an opt-in fp32-softmax path reached **only** when
    that backend is selected explicitly; it is kept solely because it is the
    tightest match to the Isaac-GR00T reference numerics (parity validation).
    ``"flashinfer"`` covers the unmasked self-attention path and falls back to the
    ``"sdpa"`` masked path for cross-attention (GR00T's cross-attention uses
    key-only masks, which flashinfer's prefill kernel does not take).
    """

    _SUPPORTED_BACKENDS = frozenset({"eager", "sdpa", "flashinfer"})

    def __init__(
        self,
        query_dim: int,
        *,
        num_heads: int,
        head_dim: int,
        cross_attention_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.attention_backend = str(backend).lower().replace("_", "-")
        if self.attention_backend not in self._SUPPORTED_BACKENDS:
            supported = ", ".join(sorted(self._SUPPORTED_BACKENDS))
            raise ValueError(
                f"GR00TN17Attention backend must be one of: {supported}; "
                f"got {backend!r}."
            )
        kv_dim = int(cross_attention_dim or query_dim)
        self.to_q = GR00TN17Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = GR00TN17Linear(kv_dim, self.inner_dim, bias=bias)
        self.to_v = GR00TN17Linear(kv_dim, self.inner_dim, bias=bias)
        self.to_out = GR00TN17Linear(self.inner_dim, query_dim, bias=True)
        self.dropout = float(dropout)
        backend_kwargs = (
            {"compile": False} if self.attention_backend == "sdpa" else None
        )
        self.prefill_attention = Attention(
            self.num_heads,
            self.head_dim,
            causal=False,
            backend=self.attention_backend,
            backend_kwargs=backend_kwargs,
        )

    @staticmethod
    def _expand_key_mask(
        attention_mask: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        attn_mask = attention_mask.to(device=device, dtype=torch.bool)
        return attn_mask[:, None, None, :]

    def _eager_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        if attention_mask is not None:
            attn = attn * attention_mask.to(dtype=attn.dtype)
        if self.dropout > 0.0 and self.training:
            keep_prob = 1.0 - self.dropout
            attn = attn * torch.empty_like(attn).bernoulli_(keep_prob) / keep_prob
        return torch.matmul(attn, v)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        dropout_p = self.dropout if self.training else 0.0
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=self.scale,
        )

    def _backend_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        use_prefill = (
            encoder_hidden_states is None
            and attention_mask is None
            and (not self.training or self.dropout == 0.0)
            and self.attention_backend != "eager"
            and (self.attention_backend != "flashinfer" or q.is_cuda)
        )
        if use_prefill:
            return self.prefill_attention(q, k, v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.attention_backend == "eager":
            out = self._eager_attention(q, k, v, attention_mask=attention_mask)
            return out.transpose(1, 2)
        # "sdpa" and the "flashinfer" masked cross-attn fallback both land
        # here: flashinfer's prefill kernel only covers the unmasked
        # self-attention path, so its masked cross-attention falls back to
        # SDPA rather than the fp32 eager path (which is reserved for the
        # explicit "eager" parity-validation backend).
        out = self._sdpa_attention(q, k, v, attention_mask=attention_mask)
        return out.transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        key_value_states = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )
        batch, target_len, _ = hidden_states.shape

        q = self.to_q(hidden_states)
        q = q.view(batch, target_len, self.num_heads, self.head_dim)
        if encoder_kv is None:
            k, v = self.project_kv(key_value_states)
        else:
            k, v = encoder_kv
        attn_mask = self._expand_key_mask(attention_mask, device=hidden_states.device)
        out = self._backend_attention(
            q,
            k,
            v,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attn_mask,
        )
        out = out.contiguous().view(batch, target_len, self.inner_dim)
        return self.to_out(out)

    def project_kv(
        self, key_value_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, source_len, _ = key_value_states.shape
        k = self.to_k(key_value_states)
        v = self.to_v(key_value_states)
        k = k.view(batch, source_len, self.num_heads, self.head_dim)
        v = v.view(batch, source_len, self.num_heads, self.head_dim)
        return k, v


class GR00TN17FeedForward(nn.Module):
    """DiT feed-forward block."""

    def __init__(self, dim: int, *, dropout: float, final_dropout: bool) -> None:
        super().__init__()
        self.fc1 = GR00TN17Linear(dim, 4 * dim)
        self.fc2 = GR00TN17Linear(4 * dim, dim)
        self.dropout = GR00TN17Dropout(dropout)
        self.final_dropout = (
            GR00TN17Dropout(dropout) if final_dropout else GR00TN17Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _gelu_tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.final_dropout(x)


class GR00TN17BasicTransformerBlock(nn.Module):
    """DiT transformer block matching the GR00T action-head topology."""

    def __init__(
        self,
        dim: int,
        config: GR00TN17DiTConfig,
        *,
        cross_attention_dim: int | None,
    ) -> None:
        super().__init__()
        self.norm_type = config.norm_type
        self.is_cross_attention = cross_attention_dim is not None
        if config.norm_type == "ada_norm":
            self.norm1 = GR00TN17AdaLayerNorm(dim)
        else:
            self.norm1 = GR00TN17LayerNorm(
                dim, eps=1e-5, elementwise_affine=config.norm_elementwise_affine
            )
        self.attn1 = GR00TN17Attention(
            dim,
            num_heads=config.num_attention_heads,
            head_dim=config.attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            dropout=config.dropout,
            bias=True,
            backend=config.attention_backend,
        )
        self.norm3 = GR00TN17LayerNorm(
            dim, eps=1e-5, elementwise_affine=config.norm_elementwise_affine
        )
        self.ff = GR00TN17FeedForward(
            dim, dropout=config.dropout, final_dropout=config.final_dropout
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_kv=encoder_kv,
        )
        hidden_states = hidden_states + attn_output
        return hidden_states + self.ff(self.norm3(hidden_states))


class GR00TN17DiT(nn.Module):
    """Native diffusion transformer used by the GR00T-N1.7 action head."""

    def __init__(
        self,
        config: GR00TN17DiTConfig,
        *,
        cross_attention_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.inner_dim = config.num_attention_heads * config.attention_head_dim
        self.output_dim = config.output_dim
        self.timestep_encoder = GR00TN17TimestepEncoder(self.inner_dim)
        blocks = []
        for idx in range(config.num_layers):
            use_self_attn = idx % 2 == 1 and config.interleave_self_attention
            blocks.append(
                GR00TN17BasicTransformerBlock(
                    self.inner_dim,
                    config,
                    cross_attention_dim=None if use_self_attn else cross_attention_dim,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)
        self.norm_out = GR00TN17LayerNorm(
            self.inner_dim, eps=1e-6, elementwise_affine=False
        )
        self.proj_out_1 = GR00TN17Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = GR00TN17Linear(self.inner_dim, self.output_dim)

    def precompute_encoder_kv(
        self,
        encoder_hidden_states: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        """Project fixed cross-attention K/V once per action inference."""
        cache: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for block in self.transformer_blocks:
            if block.is_cross_attention:
                cache.append(block.attn1.project_kv(encoder_hidden_states))
            else:
                cache.append(None)
        return cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if hidden_states.shape[-1] != self.inner_dim:
            raise ValueError(
                "hidden_states last dim must match DiT inner_dim "
                f"{self.inner_dim}, got {hidden_states.shape[-1]}."
            )
        temb = self.timestep_encoder(timestep)
        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            encoder_kv = encoder_kv_cache[idx] if encoder_kv_cache is not None else None
            if not block.is_cross_attention:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_kv=encoder_kv,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        shift, scale = self.proj_out_1(_swish(temb)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        output = self.proj_out_2(hidden_states)
        if return_all_hidden_states:
            return output, all_hidden_states
        return output


class GR00TN17AlternateVLDiT(GR00TN17DiT):
    """DiT variant alternating text-token and visual-token cross attention."""

    def __init__(
        self,
        config: GR00TN17DiTConfig,
        *,
        cross_attention_dim: int,
        attend_text_every_n_blocks: int,
    ) -> None:
        super().__init__(config, cross_attention_dim=cross_attention_dim)
        self.attend_text_every_n_blocks = int(attend_text_every_n_blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        return_all_hidden_states: bool = False,
        image_mask: torch.Tensor | None = None,
        backbone_attention_mask: torch.Tensor | None = None,
        image_attention_mask: torch.Tensor | None = None,
        non_image_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        temb = self.timestep_encoder(timestep)
        if image_attention_mask is None or non_image_attention_mask is None:
            if image_mask is None:
                raise ValueError(
                    "image_mask is required when use_alternate_vl_dit=True."
                )
            if backbone_attention_mask is None:
                if encoder_attention_mask is None:
                    raise ValueError(
                        "backbone_attention_mask is required when "
                        "use_alternate_vl_dit=True."
                    )
                else:
                    backbone_attention_mask = encoder_attention_mask
            image_attention_mask = image_mask.bool() & backbone_attention_mask.bool()
            non_image_attention_mask = (
                ~image_mask.bool()
            ) & backbone_attention_mask.bool()
        all_hidden_states = [hidden_states]
        cross_idx = 0
        for idx, block in enumerate(self.transformer_blocks):
            encoder_kv = encoder_kv_cache[idx] if encoder_kv_cache is not None else None
            if not block.is_cross_attention:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                if cross_idx % self.attend_text_every_n_blocks == 0:
                    curr_mask = non_image_attention_mask
                else:
                    curr_mask = image_attention_mask
                cross_idx += 1
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_mask,
                    encoder_kv=encoder_kv,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        shift, scale = self.proj_out_1(_swish(temb)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        output = self.proj_out_2(hidden_states)
        if return_all_hidden_states:
            return output, all_hidden_states
        return output


class GR00TN17SelfAttentionTransformer(nn.Module):
    """Optional VL self-attention stack before action cross-attention."""

    def __init__(self, config: GR00TN17VLSelfAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.inner_dim = config.num_attention_heads * config.attention_head_dim
        if config.positional_embeddings is not None:
            raise GR00TN17NativeImplementationError(
                "GR00T-N1.7 VL self-attention positional embeddings are not "
                "implemented yet."
            )
        self.transformer_blocks = nn.ModuleList(
            [
                GR00TN17SelfAttentionBlock(
                    self.inner_dim,
                    num_heads=config.num_attention_heads,
                    head_dim=config.attention_head_dim,
                    dropout=config.dropout,
                    final_dropout=config.final_dropout,
                    attention_backend=config.attention_backend,
                )
                for _ in range(config.num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        return hidden_states


class GR00TN17SelfAttentionBlock(nn.Module):
    """LayerNorm + self-attention + FFN block used by VL self-attention."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        head_dim: int,
        dropout: float,
        final_dropout: bool,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.norm1 = GR00TN17LayerNorm(dim, eps=1e-5, elementwise_affine=True)
        self.attn1 = GR00TN17Attention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            cross_attention_dim=None,
            dropout=dropout,
            bias=True,
            backend=attention_backend,
        )
        self.norm3 = GR00TN17LayerNorm(dim, eps=1e-5, elementwise_affine=True)
        self.ff = GR00TN17FeedForward(dim, dropout=dropout, final_dropout=final_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn1(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
        )
        return hidden_states + self.ff(self.norm3(hidden_states))


class GR00TN17ActionHead(nn.Module):
    """Flow-matching action head boundary."""

    def __init__(self, config: GR00TN17Config) -> None:
        super().__init__()
        self.config = config.action_head
        self.backbone_config = config.backbone
        self.hidden_size = int(self.config.hidden_size)
        self.input_embedding_dim = int(self.config.input_embedding_dim)
        self.action_dim = int(self.config.max_action_dim)
        self.action_horizon = int(self.config.action_horizon)
        self.num_inference_timesteps = int(self.config.num_inference_timesteps)
        self.num_timestep_buckets = int(self.config.num_timestep_buckets)
        dit_inner_dim = (
            self.config.dit.num_attention_heads * self.config.dit.attention_head_dim
        )
        if dit_inner_dim != self.input_embedding_dim:
            raise ValueError(
                "action_head.input_embedding_dim must equal "
                "dit.num_attention_heads * dit.attention_head_dim "
                f"({dit_inner_dim}), got {self.input_embedding_dim}."
            )

        if self.config.use_alternate_vl_dit:
            self.model = GR00TN17AlternateVLDiT(
                self.config.dit,
                cross_attention_dim=self.backbone_config.backbone_embedding_dim,
                attend_text_every_n_blocks=self.config.attend_text_every_n_blocks,
            )
        else:
            self.model = GR00TN17DiT(
                self.config.dit,
                cross_attention_dim=self.backbone_config.backbone_embedding_dim,
            )
        self.state_encoder = GR00TN17CategorySpecificMLP(
            num_categories=self.config.max_num_embodiments,
            input_dim=self.config.max_state_dim * self.config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = GR00TN17MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=self.config.max_num_embodiments,
        )
        self.action_decoder = GR00TN17CategorySpecificMLP(
            num_categories=self.config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.vlln = (
            GR00TN17LayerNorm(self.backbone_config.backbone_embedding_dim)
            if self.config.use_vlln
            else GR00TN17Identity()
        )
        if (
            self.config.vl_self_attention is not None
            and self.config.vl_self_attention.num_layers > 0
        ):
            self.vl_self_attention = GR00TN17SelfAttentionTransformer(
                self.config.vl_self_attention
            )
        else:
            self.vl_self_attention = GR00TN17Identity()
        if self.config.add_pos_embed:
            self.position_embedding = GR00TN17ReplicatedEmbedding(
                self.config.max_seq_len, self.input_embedding_dim
            )
            _init_normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def attach_hf_keys(self, prefix: str = "action_head") -> None:
        """Attach Isaac-GR00T safetensors keys to every action-head parameter."""
        category_param_ids: set[int] = set()
        for local_name, module in self.named_modules():
            if isinstance(module, GR00TN17CategorySpecificLinear):
                module.attach_hf_keys(_gr00t_n17_action_head_hf_key(local_name, prefix))
                category_param_ids.update(id(param) for param in module.parameters())
        for local_name, param in self.named_parameters():
            if id(param) in category_param_ids:
                continue
            _attach_replicated_hf_key(
                param, _gr00t_n17_action_head_hf_key(local_name, prefix)
            )

    def process_backbone_output(
        self, backbone_output: GR00TN17BackboneOutput
    ) -> GR00TN17BackboneOutput:
        backbone_features = self.vlln(backbone_output.backbone_features)
        if isinstance(self.vl_self_attention, GR00TN17SelfAttentionTransformer):
            backbone_features = self.vl_self_attention(
                backbone_features,
                attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            backbone_features = self.vl_self_attention(backbone_features)
        return GR00TN17BackboneOutput(
            backbone_features=backbone_features,
            backbone_attention_mask=backbone_output.backbone_attention_mask,
            image_mask=backbone_output.image_mask,
        )

    def _encode_features(
        self,
        backbone_output: GR00TN17BackboneOutput,
        action_input: GR00TN17ActionInput,
        *,
        static_cat_ids: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backbone_output = self.process_backbone_output(backbone_output)
        state = action_input.state
        if state.ndim != 3:
            raise ValueError(
                "state must have shape (B, state_history_length, max_state_dim), "
                f"got {tuple(state.shape)}."
            )
        if state.shape[1] != self.config.state_history_length:
            raise ValueError(
                "state history length mismatch: expected "
                f"{self.config.state_history_length}, got {state.shape[1]}."
            )
        state = state.reshape(state.shape[0], 1, -1)
        state_features = self.state_encoder(
            state,
            action_input.embodiment_id,
            static_cat_ids=static_cat_ids,
        )
        return backbone_output.backbone_features, state_features

    def _positioned_action_features(
        self,
        action_features: torch.Tensor,
        *,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.add_pos_embed:
            return action_features
        if action_features.shape[1] > self.config.max_seq_len:
            raise ValueError(
                "action horizon exceeds max_seq_len for position embedding: "
                f"{action_features.shape[1]} > {self.config.max_seq_len}."
            )
        return action_features + self.position_embedding(position_ids).unsqueeze(0)

    def validate_embodiment_id(self, embodiment_id: torch.Tensor) -> None:
        if embodiment_id.ndim != 1:
            raise ValueError(
                f"embodiment_id must have shape (B,), got {tuple(embodiment_id.shape)}."
            )
        _check_integer_category_ids(embodiment_id, name="embodiment_id")
        invalid = _invalid_category_values(
            embodiment_id.long(),
            num_categories=self.config.max_num_embodiments,
        )
        if invalid.numel() > 0:
            raise ValueError(
                "embodiment_id must be in "
                f"[0, {self.config.max_num_embodiments}); got invalid values: "
                f"{invalid.tolist()}."
            )

    @staticmethod
    def _action_mask_shape_error(
        action_mask: torch.Tensor,
        *,
        batch_size: int,
        action_horizon: int,
        action_dim: int,
    ) -> ValueError:
        return ValueError(
            "action_mask must have shape (action_dim,), (B, action_dim), "
            "(B, action_horizon), or (B, action_horizon, action_dim); got "
            f"{tuple(action_mask.shape)} for expected actions "
            f"{(batch_size, action_horizon, action_dim)}."
        )

    def validate_action_mask(
        self,
        action_mask: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> None:
        if action_mask is None:
            return
        h = self.action_horizon
        d = self.action_dim
        shape = tuple(action_mask.shape)
        valid = (
            (action_mask.ndim == 1 and shape == (d,))
            or (action_mask.ndim == 2 and shape == (batch_size, d))
            or (action_mask.ndim == 2 and shape == (batch_size, h))
            or (action_mask.ndim == 3 and shape == (batch_size, h, d))
        )
        if not valid:
            raise self._action_mask_shape_error(
                action_mask,
                batch_size=batch_size,
                action_horizon=h,
                action_dim=d,
            )

    def apply_action_mask(
        self,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if action_mask is None:
            return actions
        mask = action_mask.to(device=actions.device)
        batch_size, horizon, action_dim = actions.shape
        if mask.ndim == 1 and mask.shape[0] == action_dim:
            return actions * mask.to(dtype=actions.dtype)[None, None, :]
        if mask.ndim == 2 and tuple(mask.shape) == (batch_size, action_dim):
            return actions * mask.to(dtype=actions.dtype)[:, None, :]
        if mask.ndim == 2 and tuple(mask.shape) == (batch_size, horizon):
            return actions * mask.to(dtype=actions.dtype).unsqueeze(-1)
        if mask.ndim == 3 and tuple(mask.shape) == tuple(actions.shape):
            return actions * mask.to(dtype=actions.dtype)
        raise self._action_mask_shape_error(
            action_mask,
            batch_size=batch_size,
            action_horizon=horizon,
            action_dim=action_dim,
        )

    def prepare_initial_actions(
        self,
        backbone_features: torch.Tensor,
        *,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = backbone_features.shape[0]
        device = backbone_features.device
        expected = (batch_size, self.action_horizon, self.action_dim)
        if tuple(noise.shape) != expected:
            raise ValueError(
                f"noise must have shape {expected}, got {tuple(noise.shape)}."
            )
        return noise.to(device=device, dtype=backbone_features.dtype)

    def denoise_step(
        self,
        actions: torch.Tensor,
        step: int,
        *,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: GR00TN17BackboneOutput,
        action_input: GR00TN17ActionInput,
        timesteps: torch.Tensor,
        dt: float,
        action_position_ids: torch.Tensor,
        encoder_kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        static_cat_ids: tuple[int, ...] | None = None,
        image_mask: torch.Tensor | None = None,
        image_attention_mask: torch.Tensor | None = None,
        non_image_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action_input.action is not None:
            raise GR00TN17NativeImplementationError(
                "RTC/inpainting action inputs are not implemented in the native "
                "GR00T-N1.7 action-head path yet."
            )

        action_features = self.action_encoder(
            actions,
            timesteps,
            embodiment_id,
            static_cat_ids=static_cat_ids,
        )
        action_features = self._positioned_action_features(
            action_features,
            position_ids=action_position_ids,
        )
        sa_embs = torch.cat((state_features, action_features), dim=1)
        if self.config.use_alternate_vl_dit:
            if image_mask is None:
                raise ValueError(
                    "image_mask must be provided by the runner when "
                    "use_alternate_vl_dit=True."
                )
            model_output = self.model(
                sa_embs,
                backbone_features,
                timestep=timesteps,
                encoder_kv_cache=encoder_kv_cache,
                image_mask=image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
                image_attention_mask=image_attention_mask,
                non_image_attention_mask=non_image_attention_mask,
            )
        else:
            model_output = self.model(
                sa_embs,
                backbone_features,
                timestep=timesteps,
                encoder_attention_mask=backbone_output.backbone_attention_mask,
                encoder_kv_cache=encoder_kv_cache,
            )
        pred = self.action_decoder(
            model_output,
            embodiment_id,
            static_cat_ids=static_cat_ids,
        )
        pred_velocity = pred[:, -self.action_horizon :]
        return actions + dt * pred_velocity


# ============================================================================ #
# Top-level GR00TN17Model                                                       #
# ============================================================================ #


class GR00TN17Model(nn.Module):
    """GR00T-N1.7 parameter container.

    This class does not own scheduler state and does not expose a monolithic
    ``forward`` or ``get_action``. Runners call the backbone and action-head
    pieces independently, and own denoising control flow.
    """

    def __init__(
        self,
        config: GR00TN17Config,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        backbone_qwen3vl_model: nn.Module | None = None,
        backbone_transformers_loading_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.action_head = GR00TN17ActionHead(config)
        self.action_head.attach_hf_keys("action_head")
        if device is not None or params_dtype is not None:
            self.action_head.to(device=device, dtype=self.params_dtype)
        self.backbone = GR00TN17Backbone(
            config,
            qwen3vl_model=backbone_qwen3vl_model,
            params_dtype=self.params_dtype,
            device=device,
            transformers_loading_kwargs=backbone_transformers_loading_kwargs,
        )
        if device is not None:
            self.backbone.to(device=device)

    def prepare_backbone_input(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return self.backbone.prepare_input(inputs)

    def prepare_action_input(
        self, inputs: dict[str, torch.Tensor]
    ) -> GR00TN17ActionInput:
        batch = dict(inputs)
        required = ("state", "embodiment_id")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"action inputs missing keys: {missing}")
        return GR00TN17ActionInput(
            state=batch["state"],
            embodiment_id=batch["embodiment_id"],
            action_mask=batch.get("action_mask"),
            action=batch.get("action"),
        )

    def prepare_input(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        device: torch.device | str | None = None,
    ) -> tuple[dict[str, torch.Tensor], GR00TN17ActionInput]:
        explicit_device = torch.device(device) if device is not None else None
        backbone_device = explicit_device or self.backbone_device
        action_device = explicit_device or self.action_device
        backbone_dtype = self.backbone_dtype
        action_dtype = self.action_dtype
        action_keys = {"state", "action", "action_mask", "embodiment_id"}

        def move(key: str, value: torch.Tensor) -> torch.Tensor:
            target_device = action_device if key in action_keys else backbone_device
            if torch.is_floating_point(value):
                target_dtype = action_dtype if key in action_keys else backbone_dtype
                return value.to(device=target_device, dtype=target_dtype)
            return value.to(device=target_device)

        moved = {
            key: move(key, value) if isinstance(value, torch.Tensor) else value
            for key, value in dict(inputs).items()
        }
        return self.prepare_backbone_input(moved), self.prepare_action_input(moved)

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        for tensor in module.parameters():
            return tensor.device
        for tensor in module.buffers():
            return tensor.device
        return torch.device("cpu")

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        for tensor in module.parameters():
            if torch.is_floating_point(tensor):
                return tensor.dtype
        for tensor in module.buffers():
            if torch.is_floating_point(tensor):
                return tensor.dtype
        return torch.get_default_dtype()

    @property
    def backbone_device(self) -> torch.device:
        return self._module_device(self.backbone)

    @property
    def backbone_dtype(self) -> torch.dtype:
        if self.config.backbone.load_bf16:
            return torch.bfloat16
        return self._module_dtype(self.backbone)

    @property
    def action_device(self) -> torch.device:
        return self._module_device(self.action_head)

    @property
    def action_dtype(self) -> torch.dtype:
        return self._module_dtype(self.action_head)

    @property
    def device(self) -> torch.device:
        return self.action_device

    @property
    def dtype(self) -> torch.dtype:
        return self.action_dtype


__all__ = [
    "GR00TN17ActionHead",
    "GR00TN17ActionInput",
    "GR00TN17Backbone",
    "GR00TN17BackboneOutput",
    "GR00TN17Model",
    "GR00TN17NativeImplementationError",
]
