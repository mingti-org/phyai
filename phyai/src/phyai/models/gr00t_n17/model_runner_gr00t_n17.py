"""GR00T-N1.7 model runner boundaries.

Runners will eventually own CUDA graph capture for the backbone and
action head. The first native drop keeps these boundaries explicit and
raises before silently falling back to the reference implementation.
"""

from __future__ import annotations

import os

import torch

from phyai.models.gr00t_n17.modeling_gr00t_n17 import (
    GR00TN17ActionInput,
    GR00TN17BackboneOutput,
    GR00TN17Model,
    gr00t_n17_static_category_ids,
)
from phyai.runtime.cuda_graph_manager import CudaGraph, CudaGraphRegistry
from phyai.runtime.model_runner import ModelRunner


_BACKBONE_CUDA_GRAPH_ENV = "PHYAI_GR00T_BACKBONE_CUDA_GRAPH"


def _backbone_cuda_graph_enabled() -> bool:
    """Whether to CUDA-graph the Qwen3-VL backbone (ViT / LLM cores).

    This flag is off by default to keep a reference/eager comparison path. The
    optimized GR00T benchmark path enables it with a truthy
    ``PHYAI_GR00T_BACKBONE_CUDA_GRAPH`` (1/on/true/yes), so the fixed-shape
    ViT -> multimodal-fuse -> LLM stream is captured and replayed.
    """
    return os.environ.get(_BACKBONE_CUDA_GRAPH_ENV, "").strip().lower() in {
        "1",
        "on",
        "true",
        "yes",
    }


class GR00TN17BackboneRunner(ModelRunner):
    """Runs the Qwen3-VL backbone and owns its CUDA graph.

    Gated by ``PHYAI_GR00T_BACKBONE_CUDA_GRAPH``: the default path is eager for
    reference parity, while the benchmark/optimized path enables graph replay.
    When enabled and graph-eligible, the whole ViT -> multimodal-fuse -> LLM
    decoder is captured/replayed **here** as one shape-keyed graph: the modeling
    exposes a pure preamble + core via
    ``GR00TN17Backbone.backbone_graph_plan`` and holds no graph state .

    Lifecycle: no per-request mutable state, hence no ``reset()`` (the base
    :class:`ModelRunner` defines only ``setup``/``forward``; ``reset`` is not part
    of the contract and no runner here implements one). The only cross-call state
    is the shape-keyed graph registry, intentionally reused across requests and
    released in :meth:`close`.
    """

    def __init__(self, model: GR00TN17Model) -> None:
        self.model = model
        self.graphs = CudaGraphRegistry()

    def setup(self) -> None:
        return None

    def _replay_or_capture(self, core_fn, inputs: dict, key: tuple) -> torch.Tensor:
        graph = self.graphs.get(key)
        if graph is None:
            graph = CudaGraph()
            graph.capture(core_fn, inputs)
            self.graphs.register(key, graph)
        out = graph.replay(inputs)
        # Replay output aliases the static capture buffer; clone the pre-norm
        # features that leave to the action head this step.
        return out.pre_norm_hidden_state.clone()

    def forward(self, inputs) -> GR00TN17BackboneOutput:
        backbone_inputs = dict(inputs)
        if "position_ids" not in backbone_inputs:
            position_ids = self.model.backbone.prepare_position_ids(backbone_inputs)
            if position_ids is not None:
                backbone_inputs["position_ids"] = position_ids
        backbone = self.model.backbone
        if _backbone_cuda_graph_enabled():
            plan = backbone.backbone_graph_plan(backbone_inputs)
            if plan is not None:
                core_fn, buffers, key, model_inputs = plan
                features = self._replay_or_capture(core_fn, buffers, key)
                return backbone.build_graph_output(features, model_inputs)
        return backbone.forward(backbone_inputs)

    def close(self) -> None:
        self.graphs = CudaGraphRegistry()
        return None


class GR00TN17ActionHeadRunner(ModelRunner):
    """Runs the state/action encoders and DiT denoising loop.

    Lifecycle: like the backbone runner, it keeps no per-request mutable state
    (the constructor flags and the shape-keyed graph registry are the only state,
    reused across requests and released in :meth:`close`), so it needs no
    ``reset()`` — which the base :class:`ModelRunner` contract does not define.
    """

    @staticmethod
    def _supports_cuda_graph_attention(model: GR00TN17Model) -> bool:
        action_config = model.action_head.config
        if action_config.dit.attention_backend == "flashinfer":
            return False
        vl_self_attention = action_config.vl_self_attention
        if (
            vl_self_attention is not None
            and vl_self_attention.attention_backend == "flashinfer"
        ):
            return False
        return True

    def __init__(
        self,
        model: GR00TN17Model,
        *,
        max_batch_size: int = 1,
        device: torch.device | str | None = None,
        use_cuda_graph: bool = True,
    ) -> None:
        self.model = model
        self.max_batch_size = int(max_batch_size)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.use_cuda_graph = (
            bool(use_cuda_graph)
            and self.device.type == "cuda"
            and self._supports_cuda_graph_attention(model)
        )
        self.graphs = CudaGraphRegistry()

    def setup(self) -> None:
        return None

    def _shape_key(
        self,
        backbone_output: GR00TN17BackboneOutput,
        action_input: GR00TN17ActionInput,
        noise: torch.Tensor,
    ) -> tuple[object, ...]:
        return (
            tuple(backbone_output.backbone_features.shape),
            backbone_output.backbone_features.dtype,
            tuple(backbone_output.backbone_attention_mask.shape),
            tuple(action_input.state.shape),
            action_input.state.dtype,
            tuple(action_input.embodiment_id.shape),
            tuple(noise.shape),
            noise.dtype,
        )

    @staticmethod
    def _category_key(action_input: GR00TN17ActionInput) -> tuple[int, ...] | None:
        cat_ids = action_input.embodiment_id
        if cat_ids.ndim != 1:
            return None
        return tuple(int(v) for v in cat_ids.detach().cpu().tolist())

    def _prepare_noise(
        self,
        backbone_output: GR00TN17BackboneOutput,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.model.action_head.prepare_initial_actions(
            backbone_output.backbone_features,
            noise=noise,
        )

    def _image_mask_tensor(
        self,
        backbone_output: GR00TN17BackboneOutput,
    ) -> torch.Tensor:
        if backbone_output.image_mask is not None:
            return backbone_output.image_mask
        return torch.zeros_like(
            backbone_output.backbone_attention_mask,
            dtype=torch.bool,
        )

    def _fwd_loop(
        self,
        *,
        backbone_features: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        backbone_output = GR00TN17BackboneOutput(
            backbone_features=backbone_features,
            backbone_attention_mask=backbone_attention_mask,
            image_mask=image_mask,
        )
        action_input = GR00TN17ActionInput(
            state=state,
            embodiment_id=embodiment_id,
        )
        action_head = self.model.action_head
        backbone_features, state_features = action_head._encode_features(
            backbone_output, action_input
        )
        actions = action_head.prepare_initial_actions(backbone_features, noise=noise)
        encoder_kv_cache = action_head.precompute_dit_encoder_kv(backbone_features)
        for step in range(action_head.num_inference_timesteps):
            actions = action_head.denoise_step(
                actions,
                step,
                backbone_features=backbone_features,
                state_features=state_features,
                embodiment_id=action_input.embodiment_id,
                backbone_output=backbone_output,
                action_input=action_input,
                encoder_kv_cache=encoder_kv_cache,
            )
        return actions

    def _static_category_fwd_loop(
        self,
        *,
        category_key: tuple[int, ...],
        backbone_features: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        with gr00t_n17_static_category_ids(category_key):
            return self._fwd_loop(
                backbone_features=backbone_features,
                backbone_attention_mask=backbone_attention_mask,
                image_mask=image_mask,
                state=state,
                embodiment_id=embodiment_id,
                noise=noise,
            )

    def _graph_inputs(
        self,
        backbone_output: GR00TN17BackboneOutput,
        action_input: GR00TN17ActionInput,
        noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "backbone_features": backbone_output.backbone_features,
            "backbone_attention_mask": backbone_output.backbone_attention_mask,
            "image_mask": self._image_mask_tensor(backbone_output),
            "state": action_input.state,
            "embodiment_id": action_input.embodiment_id,
            "noise": noise,
        }

    def forward(
        self,
        backbone_output: GR00TN17BackboneOutput,
        action_input: GR00TN17ActionInput,
        *,
        noise=None,
    ):
        if action_input.action is not None:
            return self.model.action_head.get_action(
                backbone_output,
                action_input,
                noise=noise,
            )
        if backbone_output.backbone_features.shape[0] > self.max_batch_size:
            raise ValueError(
                "GR00T-N1.7 action runner batch exceeds max_batch_size: "
                f"{backbone_output.backbone_features.shape[0]} > {self.max_batch_size}."
            )
        noise = self._prepare_noise(backbone_output, noise)
        inputs = self._graph_inputs(backbone_output, action_input, noise)
        if self.use_cuda_graph:
            category_key = self._category_key(action_input)
            key = self._shape_key(backbone_output, action_input, noise) + (
                category_key,
            )
            graph = self.graphs.get(key)
            if graph is None:
                graph = CudaGraph()
                if category_key is None:
                    graph.capture(self._fwd_loop, inputs)
                else:
                    graph.capture(
                        lambda **kwargs: self._static_category_fwd_loop(
                            category_key=category_key,
                            **kwargs,
                        ),
                        inputs,
                    )
                self.graphs.register(key, graph)
            return graph.replay(inputs)
        return self._fwd_loop(**inputs)

    def close(self) -> None:
        self.graphs = CudaGraphRegistry()
        return None


__all__ = ["GR00TN17ActionHeadRunner", "GR00TN17BackboneRunner"]
