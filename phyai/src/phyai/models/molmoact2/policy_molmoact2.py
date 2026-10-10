"""Stateless composition of the MolmoAct2 backbone and action expert."""

from dataclasses import dataclass

import torch
from torch import nn

from phyai.layers.linear import ReplicatedLinear
from phyai.models.molmoact2.modeling_molmoact2 import (
    MolmoAct2TextModel,
    MolmoAct2VisionBackbone,
)
from phyai.models.molmoact2.modeling_action_expert import ActionExpert
from phyai.models.molmoact2.configuration_molmoact2 import MolmoAct2Config


def batch_images(
    config: MolmoAct2Config,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_token_pooling: torch.Tensor,
    image_grids: torch.Tensor,
    image_num_crops: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack the processor's image crops and image-local pooling indices by sample."""
    raw_counts = (input_ids == config.image_end_token_id).sum(1).tolist()
    image_count = image_grids.shape[0]
    if sum(raw_counts) == image_count:
        counts = raw_counts
    elif sum(raw_counts) == 2 * image_count and all(
        count % 2 == 0 for count in raw_counts
    ):
        counts = [count // 2 for count in raw_counts]
    else:
        raise ValueError("Image end tokens do not match the supplied image grids.")
    crops = image_num_crops.tolist()
    pools = (image_grids[:, :2].prod(-1) + image_grids[:, 2:].prod(-1)).tolist()
    if len(crops) != image_count or sum(crops) != pixel_values.shape[0]:
        raise ValueError("Image crop counts do not match pixel_values.")
    if sum(pools) != image_token_pooling.shape[0] or image_count == 0:
        raise ValueError("Image grids do not match pooling indices.")
    images_by_sample, pooling_by_sample = [], []
    image_offset = crop_offset = pool_offset = 0
    for count in counts:
        sample_crops = sum(crops[image_offset : image_offset + count])
        images_by_sample.append(pixel_values[crop_offset : crop_offset + sample_crops])
        local_offset = 0
        local_pool = []
        for image_index in range(image_offset, image_offset + count):
            n_pool = pools[image_index]
            indices = image_token_pooling[pool_offset : pool_offset + n_pool]
            local_pool.append(
                torch.where(indices >= 0, indices + local_offset, indices)
            )
            local_offset += crops[image_index] * pixel_values.shape[1]
            pool_offset += n_pool
        pooling_by_sample.append(
            torch.cat(local_pool) if local_pool else image_token_pooling[:0]
        )
        crop_offset += sample_crops
        image_offset += count
    max_crops = max(part.shape[0] for part in images_by_sample)
    max_pool = max(part.shape[0] for part in pooling_by_sample)
    images = pixel_values.new_full(
        (len(counts), max_crops, *pixel_values.shape[1:]), -1
    )
    pooling = image_token_pooling.new_full(
        (len(counts), max_pool, image_token_pooling.shape[1]), -1
    )
    for index, (image, pool) in enumerate(zip(images_by_sample, pooling_by_sample)):
        images[index, : image.shape[0]] = image
        pooling[index, : pool.shape[0]] = pool
    return images, pooling


@dataclass(frozen=True)
class MolmoAct2Output:
    last_hidden_state: torch.Tensor
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    image_hidden_states: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    logits: torch.Tensor | None = None


class MolmoAct2Model(nn.Module):
    def __init__(
        self,
        config: MolmoAct2Config,
        *,
        params_dtype: torch.dtype = torch.bfloat16,
        vision_params_dtype: torch.dtype | None = None,
        device=None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.transformer = MolmoAct2TextModel(
            config.text_config,
            params_dtype=params_dtype,
            device=device,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
        )
        self.vision_backbone = MolmoAct2VisionBackbone(
            config.vit_config,
            config.adapter_config,
            params_dtype=vision_params_dtype or params_dtype,
            device=device,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
        )
        self.action_expert = (
            ActionExpert(
                config.action_expert_config,
                llm_dim=config.text_config.hidden_size,
                llm_kv_dim=config.text_config.num_key_value_heads
                * config.text_config.head_dim,
                llm_num_layers=config.text_config.num_hidden_layers,
                params_dtype=params_dtype,
                device=device,
                attn_backend=attn_backend,
                norm_backend=norm_backend,
                prefix="model.action_expert",
            )
            if config.add_action_expert
            else None
        )

    def build_input_embeddings(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        token_pooling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        safe_ids = input_ids.masked_fill(input_ids == -1, 0)
        embeddings = self.transformer.wte(safe_ids)
        features = None
        if images is not None:
            if token_pooling is None:
                raise ValueError("Image crops require token_pooling.")
            features = self.vision_backbone(images, token_pooling).to(embeddings.device)
            is_patch = safe_ids == self.config.image_patch_id
            if int(is_patch.sum()) != features.shape[0]:
                raise ValueError(
                    "Image patch tokens do not match pooled image features."
                )
            embeddings = embeddings.clone()
            embeddings[is_patch] += features
        return embeddings, features

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        token_pooling: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        output_hidden_states: bool = False,
    ) -> MolmoAct2Output:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids and inputs_embeds.")
        features = None
        if pixel_values is not None:
            if (
                input_ids is None
                or image_token_pooling is None
                or image_grids is None
                or image_num_crops is None
            ):
                raise ValueError(
                    "pixel_values requires IDs, pooling indices, grids, and crop counts."
                )
            if images is not None:
                raise ValueError("Supply either pixel_values or packed images.")
            images, token_pooling = batch_images(
                self.config,
                input_ids,
                pixel_values,
                image_token_pooling,
                image_grids,
                image_num_crops,
            )
        if inputs_embeds is None:
            inputs_embeds, features = self.build_input_embeddings(
                input_ids, images, token_pooling
            )
        elif images is not None:
            raise ValueError("Images cannot accompany inputs_embeds.")
        output = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
        )
        return MolmoAct2Output(
            output.last_hidden_state,
            output.past_key_values,
            features,
            output.hidden_states,
        )


class MolmoAct2ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: MolmoAct2Config,
        *,
        params_dtype=torch.bfloat16,
        device=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = MolmoAct2Model(
            config, params_dtype=params_dtype, device=device, **kwargs
        )
        self.lm_head = ReplicatedLinear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix="lm_head",
        )

    def forward(
        self, input_ids=None, *, logits_to_keep: int = 0, **kwargs
    ) -> MolmoAct2Output:
        output = self.model(input_ids, **kwargs)
        hidden = (
            output.last_hidden_state[:, -logits_to_keep:]
            if logits_to_keep
            else output.last_hidden_state
        )
        return MolmoAct2Output(
            output.last_hidden_state,
            output.past_key_values,
            output.image_hidden_states,
            output.hidden_states,
            self.lm_head(hidden)[0],
        )


def molmoact2_weight_remap(name: str) -> str | None:
    if name == "model.transformer.rotary_emb.inv_freq":
        return None  # The shared rotary layer reconstructs this constant from config.
    return name
