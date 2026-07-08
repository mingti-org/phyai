"""GR00T-N1.7 model runner boundaries.

Runners own CUDA graph capture, request-scoped sampling state, and the
multi-step action denoising loop. The model modules remain parameter
containers plus single-step math.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import torch

from phyai.layers.attention.utils import (
    get_global_fi_workspace,
    resolve_prefill_backend,
)
from phyai.models.gr00t_n17.modeling_gr00t_n17 import (
    GR00TN17ActionInput,
    GR00TN17BackboneOutput,
    GR00TN17Model,
)
from phyai.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb_vision,
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)
from phyai.runtime.cuda_graph_manager import CudaGraph, CudaGraphRegistry
from phyai.runtime.model_runner import ModelRunner


def _tensor_graph_key(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
    )


def _right_pad_sequence(
    tensor: torch.Tensor,
    target_len: int,
    *,
    value: int,
) -> torch.Tensor:
    pad_len = int(target_len) - tensor.shape[1]
    if pad_len <= 0:
        return tensor
    pad = tensor.new_full((tensor.shape[0], pad_len), value)
    return torch.cat((tensor, pad), dim=1)


def _right_pad_position_ids(
    position_ids: torch.Tensor,
    target_len: int,
) -> torch.Tensor:
    pad_len = int(target_len) - position_ids.shape[-1]
    if pad_len <= 0:
        return position_ids
    pad = position_ids.new_zeros((*position_ids.shape[:-1], pad_len))
    return torch.cat((position_ids, pad), dim=-1)


class GR00TN17BackboneRunner(ModelRunner):
    """Runs the Qwen3-VL backbone and owns shape-keyed CUDA graphs."""

    def __init__(
        self,
        model: GR00TN17Model,
        *,
        device: torch.device | str | None = None,
        use_cuda_graph: bool = True,
    ) -> None:
        self.model = model
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        self.graphs = CudaGraphRegistry()
        self._vision_graph_states: dict[tuple, tuple] = {}

    def setup(self) -> None:
        return None

    def _select_graph_seq_len_bucket(self, seq_len: int) -> int | None:
        for bucket in self.model.backbone.config.graph_seq_len_buckets:
            if seq_len <= bucket:
                return int(bucket)
        return None

    def _replay_or_capture(self, core_fn, inputs: dict, key: tuple) -> torch.Tensor:
        graph = self.graphs.get(key)
        if graph is None:
            graph = CudaGraph()
            self._vision_graph_states[key] = self._make_vision_graph_state(inputs)
            core_fn = partial(
                core_fn,
                vision_graph_state=self._vision_graph_states[key],
            )
            graph.capture(core_fn, inputs)
            self.graphs.register(key, graph)
        out = graph.replay(inputs)
        # Replay output aliases the static capture buffer; clone the pre-norm
        # features that leave to the action head this step.
        return out.pre_norm_hidden_state.clone()

    def _backbone_graph_plan(self, inputs: dict[str, torch.Tensor]):
        backbone = self.model.backbone
        model_inputs = backbone._prepare_model_inputs(inputs)
        if "pixel_values_videos" in model_inputs or "video_grid_thw" in model_inputs:
            return None

        qwen3vl_model = backbone._load_qwen3vl_model()
        qwen_model = qwen3vl_model.model
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        image_grid_thw = model_inputs["image_grid_thw"]
        bucket_seq_len = self._select_graph_seq_len_bucket(input_ids.shape[1])
        if bucket_seq_len is None:
            return None
        position_ids = model_inputs.get("position_ids")
        if position_ids is None:
            position_ids, _ = qwen_model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )

        pad_token_id = int(getattr(qwen3vl_model.config, "pad_token_id", 0) or 0)
        input_ids = _right_pad_sequence(
            input_ids,
            bucket_seq_len,
            value=pad_token_id,
        )
        attention_mask = _right_pad_sequence(
            attention_mask,
            bucket_seq_len,
            value=0,
        )
        position_ids = _right_pad_position_ids(position_ids, bucket_seq_len)
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = attention_mask
        model_inputs["position_ids"] = position_ids
        visual = qwen_model.visual
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            image_grid_thw,
            visual.num_grid_per_side,
            visual.spatial_merge_size,
        )
        vision_position_ids = get_vision_position_ids(
            image_grid_thw, visual.spatial_merge_size
        )
        cu_seqlens = get_vision_cu_seqlens(image_grid_thw)
        cos, sin = qwen_model.language_model.rotary_emb.get_cos_sin(position_ids)

        # The graph path currently handles image-only visual inputs. Video
        # tensors fall back before this point, so image-token indices are the
        # full visual insertion set for the captured core.
        image_mask = input_ids == qwen3vl_model.config.image_token_id
        visual_index = image_mask.reshape(-1).nonzero(as_tuple=True)[0]
        key = (
            _tensor_graph_key(input_ids),
            _tensor_graph_key(model_inputs["pixel_values"]),
            _tensor_graph_key(attention_mask),
            tuple(int(v) for v in image_grid_thw.detach().cpu().flatten().tolist()),
            _tensor_graph_key(visual_index),
        )
        buffers = {
            "input_ids": input_ids,
            "pixel_values": model_inputs["pixel_values"],
            "bilinear_indices": bilinear_indices,
            "bilinear_weights": bilinear_weights,
            "vision_position_ids": vision_position_ids,
            "cu_seqlens": cu_seqlens,
            "cos": cos,
            "sin": sin,
            "visual_index": visual_index,
        }
        return self._backbone_core, buffers, key, model_inputs

    def _make_vision_graph_state(self, inputs: dict[str, torch.Tensor]) -> tuple:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

        qwen_model = self.model.backbone._load_qwen3vl_model().model
        visual = qwen_model.visual
        cu_seqlens = inputs["cu_seqlens"]
        workspace = get_global_fi_workspace(cu_seqlens.device)
        wrappers = []
        for block in visual.blocks:
            layer = block.attn.attn
            qo_indptr_buf = torch.empty_like(cu_seqlens, dtype=torch.int32)
            kv_indptr_buf = torch.empty_like(cu_seqlens, dtype=torch.int32)
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                workspace,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=qo_indptr_buf,
                kv_indptr_buf=kv_indptr_buf,
                backend=resolve_prefill_backend(),
            )
            wrapper.plan(
                cu_seqlens.to(torch.int32),
                cu_seqlens.to(torch.int32),
                num_qo_heads=layer.num_heads,
                num_kv_heads=layer.num_kv_heads,
                head_dim_qk=layer.head_dim,
                causal=layer.causal,
                sm_scale=layer.scale,
                window_left=(
                    -1 if layer.sliding_window is None else layer.sliding_window - 1
                ),
                logits_soft_cap=layer.logits_soft_cap,
                q_data_type=visual.dtype,
                kv_data_type=visual.dtype,
            )
            wrappers.append(wrapper)
        return tuple(wrappers)

    @staticmethod
    def _run_vision_attention_with_wrapper(
        block,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        wrapper,
    ) -> torch.Tensor:
        attn = block.attn
        seq_length = hidden_states.shape[0]
        qkv, _ = attn.qkv(hidden_states)
        q, k, v = (
            qkv.reshape(seq_length, 3, attn.num_heads, attn.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        out = wrapper.run(q, k, v)
        out = out.reshape(seq_length, -1)
        out, _ = attn.proj(out)
        return out

    def _run_vision_block_with_wrapper(
        self,
        block,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        wrapper,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self._run_vision_attention_with_wrapper(
            block,
            block.norm1(hidden_states),
            position_embeddings,
            wrapper,
        )
        hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
        return hidden_states

    def _backbone_core(
        self,
        vision_graph_state: tuple,
        *,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        bilinear_indices: torch.Tensor,
        bilinear_weights: torch.Tensor,
        vision_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        visual_index: torch.Tensor,
    ) -> SimpleNamespace:
        qwen_model = self.model.backbone._load_qwen3vl_model().model
        visual = qwen_model.visual
        hidden_states = visual.patch_embed(pixel_values.to(dtype=visual.dtype))
        pos_embeds = (
            visual.pos_embed_weight[bilinear_indices] * bilinear_weights[:, :, None]
        ).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary_pos_emb = visual.rotary_pos_emb(vision_position_ids)
        seq_len = hidden_states.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_visual_embeds: list[torch.Tensor] = []
        for layer_num, block in enumerate(visual.blocks):
            hidden_states = self._run_vision_block_with_wrapper(
                block,
                hidden_states,
                position_embeddings,
                vision_graph_state[layer_num],
            )
            if layer_num in visual.deepstack_visual_indexes:
                idx = visual.deepstack_visual_indexes.index(layer_num)
                deepstack_visual_embeds.append(
                    visual.deepstack_merger_list[idx](hidden_states)
                )
        image_embeds = visual.merger(hidden_states)

        language_model = qwen_model.language_model
        inputs_embeds = language_model.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds.reshape(-1, inputs_embeds.shape[-1]).index_copy_(
            0,
            visual_index,
            image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
        )

        cos = cos.to(inputs_embeds.dtype)
        sin = sin.to(inputs_embeds.dtype)
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(language_model.layers):
            hidden_states = layer(hidden_states, cos=cos, sin=sin, attn_ctx=None)
            if layer_idx < len(deepstack_visual_embeds):
                hidden_states = hidden_states.clone()
                hidden_states.reshape(-1, hidden_states.shape[-1]).index_add_(
                    0,
                    visual_index,
                    deepstack_visual_embeds[layer_idx].to(
                        hidden_states.device, hidden_states.dtype
                    ),
                )
        return SimpleNamespace(
            last_hidden_state=language_model.norm(hidden_states),
            pre_norm_hidden_state=hidden_states,
            hidden_states=None,
        )

    def forward(self, inputs) -> GR00TN17BackboneOutput:
        backbone_inputs = dict(inputs)
        if "position_ids" not in backbone_inputs:
            position_ids = self.model.backbone.prepare_position_ids(backbone_inputs)
            if position_ids is not None:
                backbone_inputs["position_ids"] = position_ids
        backbone = self.model.backbone
        if self.use_cuda_graph:
            plan = self._backbone_graph_plan(backbone_inputs)
            if plan is None:
                return backbone.forward(backbone_inputs)
            core_fn, buffers, key, model_inputs = plan
            features = self._replay_or_capture(core_fn, buffers, key)
            return backbone.build_graph_output(features, model_inputs)
        return backbone.forward(backbone_inputs)

    def close(self) -> None:
        self.graphs = CudaGraphRegistry()
        self._vision_graph_states = {}
        return None


class GR00TN17ActionHeadRunner(ModelRunner):
    """Runs action denoising and owns shape/category-keyed CUDA graphs."""

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
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.use_cuda_graph = (
            bool(use_cuda_graph)
            and self.device.type == "cuda"
            and self._supports_cuda_graph_attention(model)
        )
        self.graphs = CudaGraphRegistry()
        self._step_timesteps: torch.Tensor | None = None
        self._action_position_ids: torch.Tensor | None = None
        self._dt: float = 1.0
        self._empty_image_masks: dict[tuple[object, ...], torch.Tensor] = {}

    def setup(self) -> None:
        self._init_scheduler_buffers()
        return None

    def _init_scheduler_buffers(self) -> None:
        action_head = self.model.action_head
        step_ids = [
            int(
                (step / float(action_head.num_inference_timesteps))
                * action_head.num_timestep_buckets
            )
            for step in range(action_head.num_inference_timesteps)
        ]
        self._step_timesteps = (
            torch.tensor(step_ids, dtype=torch.long, device=self.device)
            .unsqueeze(1)
            .expand(action_head.num_inference_timesteps, self.max_batch_size)
            .clone()
        )
        self._action_position_ids = torch.arange(
            action_head.action_horizon,
            dtype=torch.long,
            device=self.device,
        )
        self._dt = 1.0 / action_head.num_inference_timesteps

    def _ensure_scheduler_buffers(self) -> None:
        if (
            self._step_timesteps is None
            or self._action_position_ids is None
            or self._step_timesteps.device != self.device
            or self._action_position_ids.device != self.device
        ):
            self._init_scheduler_buffers()

    @staticmethod
    def _shape_key(inputs: dict[str, torch.Tensor]) -> tuple[object, ...]:
        return tuple(
            (name, _tensor_graph_key(tensor)) for name, tensor in sorted(inputs.items())
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
        action_head = self.model.action_head
        batch_size = backbone_output.backbone_features.shape[0]
        if noise is None:
            return torch.randn(
                batch_size,
                action_head.action_horizon,
                action_head.action_dim,
                dtype=backbone_output.backbone_features.dtype,
                device=backbone_output.backbone_features.device,
            )
        return noise.to(
            device=backbone_output.backbone_features.device,
            dtype=backbone_output.backbone_features.dtype,
        )

    def _image_mask_tensor(
        self,
        backbone_output: GR00TN17BackboneOutput,
    ) -> torch.Tensor:
        if backbone_output.image_mask is not None:
            return backbone_output.image_mask
        attention_mask = backbone_output.backbone_attention_mask
        if attention_mask is None:
            raise ValueError("image_mask is required when backbone mask is compacted.")
        key = (
            tuple(attention_mask.shape),
            attention_mask.device.type,
            attention_mask.device.index,
        )
        image_mask = self._empty_image_masks.get(key)
        if image_mask is None:
            image_mask = torch.zeros(
                attention_mask.shape,
                dtype=torch.bool,
                device=attention_mask.device,
            )
            self._empty_image_masks[key] = image_mask
        return image_mask

    @staticmethod
    def _alternate_vl_attention_masks(
        backbone_attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backbone_mask = backbone_attention_mask.bool()
        image_mask = image_mask.bool()
        return image_mask & backbone_mask, (~image_mask) & backbone_mask

    def _fwd_loop(
        self,
        *,
        backbone_features: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
        image_attention_mask: torch.Tensor,
        non_image_attention_mask: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        noise: torch.Tensor,
        image_token_indices: torch.Tensor | None = None,
        non_image_token_indices: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        static_cat_ids: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        self._ensure_scheduler_buffers()
        backbone_output = GR00TN17BackboneOutput(
            backbone_features=backbone_features,
            backbone_attention_mask=backbone_attention_mask,
            image_mask=image_mask,
        )
        action_input = GR00TN17ActionInput(
            state=state,
            embodiment_id=embodiment_id,
            action_mask=action_mask,
        )
        action_head = self.model.action_head
        backbone_features, state_features = action_head._encode_features(
            backbone_output,
            action_input,
            static_cat_ids=static_cat_ids,
        )
        effective_backbone_attention_mask = (
            None if backbone_attention_mask.numel() == 0 else backbone_attention_mask
        )
        backbone_output = GR00TN17BackboneOutput(
            backbone_features=backbone_features,
            backbone_attention_mask=effective_backbone_attention_mask,
            image_mask=image_mask,
        )
        actions = action_head.prepare_initial_actions(backbone_features, noise=noise)
        if self.use_cuda_graph:
            actions = actions.clone()
        encoder_kv_cache_is_masked = False
        if (
            action_head.config.use_alternate_vl_dit
            and image_token_indices is not None
            and non_image_token_indices is not None
        ):
            encoder_kv_cache = action_head.model.precompute_masked_encoder_kv(
                backbone_features,
                image_token_indices=image_token_indices,
                non_image_token_indices=non_image_token_indices,
            )
            encoder_kv_cache_is_masked = True
        else:
            encoder_kv_cache = action_head.model.precompute_encoder_kv(
                backbone_features
            )
        batch_size = actions.shape[0]
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
                encoder_kv_cache_is_masked=encoder_kv_cache_is_masked,
                static_cat_ids=static_cat_ids,
                timesteps=self._step_timesteps[step, :batch_size],
                action_position_ids=self._action_position_ids,
                image_mask=image_mask,
                image_attention_mask=image_attention_mask,
                non_image_attention_mask=non_image_attention_mask,
                dt=self._dt,
            )
        # Isaac-GR00T inference does not feed action_mask into the denoise loop.
        # Apply it only to the returned normalized chunk so padding/dim masks do
        # not perturb the sampled trajectory used for official parity.
        return action_head.apply_action_mask(actions, action_input.action_mask)

    def _static_category_fwd_loop(
        self,
        *,
        category_key: tuple[int, ...],
        backbone_features: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
        image_attention_mask: torch.Tensor,
        non_image_attention_mask: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        noise: torch.Tensor,
        image_token_indices: torch.Tensor | None = None,
        non_image_token_indices: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._fwd_loop(
            backbone_features=backbone_features,
            backbone_attention_mask=backbone_attention_mask,
            image_mask=image_mask,
            image_attention_mask=image_attention_mask,
            non_image_attention_mask=non_image_attention_mask,
            state=state,
            embodiment_id=embodiment_id,
            noise=noise,
            image_token_indices=image_token_indices,
            non_image_token_indices=non_image_token_indices,
            action_mask=action_mask,
            static_cat_ids=category_key,
        )

    @staticmethod
    def _require_shared_mask(mask: torch.Tensor, name: str) -> None:
        if mask.ndim != 2:
            raise ValueError(f"{name} must be 2-D, got {tuple(mask.shape)}.")
        if mask.shape[0] > 1 and not torch.equal(mask, mask[:1].expand_as(mask)):
            raise ValueError(
                "GR00T-N1.7 CUDA graph compacted mask path requires identical "
                f"{name} rows across the batch."
            )

    def _graph_inputs(
        self,
        backbone_output: GR00TN17BackboneOutput,
        action_input: GR00TN17ActionInput,
        noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_mask = self._image_mask_tensor(backbone_output)
        if self.use_cuda_graph:
            if backbone_output.backbone_attention_mask is None:
                raise ValueError("CUDA graph path requires a backbone attention mask.")
            backbone_mask = backbone_output.backbone_attention_mask.bool()
            image_mask = image_mask.bool()
            self._require_shared_mask(backbone_mask, "backbone_attention_mask")
            self._require_shared_mask(image_mask, "image_mask")
            valid_indices = backbone_mask[0].nonzero(as_tuple=True)[0]
            if valid_indices.numel() == 0:
                raise ValueError("backbone_attention_mask has no valid tokens.")
            backbone_features = backbone_output.backbone_features.index_select(
                1, valid_indices
            )
            compact_image_mask = image_mask.index_select(1, valid_indices)
            compact_backbone_mask = backbone_mask.new_empty((backbone_mask.shape[0], 0))
            compact_valid_mask = torch.ones_like(compact_image_mask, dtype=torch.bool)
            image_attention_mask, non_image_attention_mask = (
                self._alternate_vl_attention_masks(
                    compact_valid_mask,
                    compact_image_mask,
                )
            )
            inputs = {
                "backbone_features": backbone_features,
                "backbone_attention_mask": compact_backbone_mask,
                "image_mask": compact_image_mask,
                "image_attention_mask": image_attention_mask,
                "non_image_attention_mask": non_image_attention_mask,
                "state": action_input.state,
                "embodiment_id": action_input.embodiment_id,
                "noise": noise,
            }
            if self.model.action_head.config.use_alternate_vl_dit:
                inputs["image_token_indices"] = compact_image_mask[0].nonzero(
                    as_tuple=True
                )[0]
                inputs["non_image_token_indices"] = (~compact_image_mask[0]).nonzero(
                    as_tuple=True
                )[0]
            if action_input.action_mask is not None:
                inputs["action_mask"] = action_input.action_mask
            return inputs
        image_attention_mask, non_image_attention_mask = (
            self._alternate_vl_attention_masks(
                backbone_output.backbone_attention_mask,
                image_mask,
            )
        )
        inputs = {
            "backbone_features": backbone_output.backbone_features,
            "backbone_attention_mask": backbone_output.backbone_attention_mask,
            "image_mask": image_mask,
            "image_attention_mask": image_attention_mask,
            "non_image_attention_mask": non_image_attention_mask,
            "state": action_input.state,
            "embodiment_id": action_input.embodiment_id,
            "noise": noise,
        }
        if action_input.action_mask is not None:
            inputs["action_mask"] = action_input.action_mask
        return inputs

    def forward(
        self,
        backbone_output: GR00TN17BackboneOutput,
        action_input: GR00TN17ActionInput,
        *,
        noise=None,
    ):
        action_head = self.model.action_head
        action_head.validate_embodiment_id(action_input.embodiment_id)
        if backbone_output.backbone_features.shape[0] > self.max_batch_size:
            raise ValueError(
                "GR00T-N1.7 action runner batch exceeds max_batch_size: "
                f"{backbone_output.backbone_features.shape[0]} > {self.max_batch_size}."
            )
        action_head.validate_action_mask(
            action_input.action_mask,
            batch_size=backbone_output.backbone_features.shape[0],
        )
        runtime_device = backbone_output.backbone_features.device
        if self.device != runtime_device:
            self.device = runtime_device
            self._init_scheduler_buffers()
        self._ensure_scheduler_buffers()
        noise = self._prepare_noise(backbone_output, noise)
        inputs = self._graph_inputs(backbone_output, action_input, noise)
        if self.use_cuda_graph:
            category_key = self._category_key(action_input)
            if category_key is None:
                return self._fwd_loop(**inputs)
            key = self._shape_key(inputs) + (("category_key", category_key),)
            graph = self.graphs.get(key)
            if graph is None:
                graph = CudaGraph()
                graph.capture(
                    lambda **kwargs: self._static_category_fwd_loop(
                        category_key=category_key,
                        **kwargs,
                    ),
                    inputs,
                )
                self.graphs.register(key, graph)
            return graph.replay(inputs).clone()
        return self._fwd_loop(**inputs)

    def close(self) -> None:
        self.graphs = CudaGraphRegistry()
        self._empty_image_masks = {}
        return None


__all__ = ["GR00TN17ActionHeadRunner", "GR00TN17BackboneRunner"]
