"""Stateless GoalWAM forward using PhyAI's Cosmos3 towers and TP layers."""

from dataclasses import fields, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from phyai.parallel import use_mesh
from phyai.layers.linear import ReplicatedLinear
from phyai.weights.shards import replicated
from phyai.layers.layer_norm import RMSNorm
from phyai.layers.mlp.dense_mlp import DenseMLP
from phyai.models.goalwam.text_goalwam import (
    GoalWAMRMSNorm,
    GoalWAMTextProjection,
    register_goalwam_text_mesh,
)
from phyai.models.cosmos3.modeling_cosmos3 import (
    Cosmos3Condition,
    DomainAwareLinear,
    Cosmos3Transformer,
    Cosmos3LanguageModel,
)
from phyai.models.cosmos3.configuration_cosmos3 import Cosmos3Config
from phyai.models.goalwam.configuration_goalwam import GoalWAMConfig
from phyai.models.goalwam.modeling_kernels_goalwam import (
    add_goalwam_norm,
    prepare_goalwam_attention,
)


class SharedEEFLinear(nn.Module):
    """Shared Cartesian channels plus independently masked joint channels."""

    def __init__(
        self,
        config: GoalWAMConfig,
        *,
        output: bool,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.output = output
        self.joint_dim = config.joint_dim
        self.shared_dim = config.shared_dim
        self.action_dim = config.action_dim
        hidden = config.hidden_size
        self.shared = ReplicatedLinear(
            hidden if output else config.shared_dim,
            config.shared_dim if output else hidden,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.shared",
        )
        self.joint = DomainAwareLinear(
            hidden if output else config.joint_dim,
            config.joint_dim if output else hidden,
            config.num_embodiment_domains,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.joint",
        )

    def forward(
        self,
        value: torch.Tensor,
        domains: torch.Tensor,
        valid: torch.Tensor,
        joint_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Row selection is prepared by the runner before graph capture."""
        if valid.shape != (value.shape[0], self.action_dim):
            raise ValueError("Action validity must match the projection row count")
        joint_input = (
            value
            if self.output
            else value[:, : self.joint_dim].masked_fill(~valid[:, : self.joint_dim], 0)
        )
        selected = self.joint(
            joint_input.index_select(0, joint_rows), domains.index_select(0, joint_rows)
        )
        joint = value.new_zeros((value.shape[0], self.joint.output_size)).index_copy(
            0, joint_rows, selected
        )
        if self.output:
            # The 19-channel bridge is not aligned for every GEMM backend.
            shared = F.linear(value, self.shared.weight, self.shared.bias)
            padding = value.new_zeros(
                (value.shape[0], self.action_dim - self.joint_dim - self.shared_dim)
            )
            return torch.cat((joint, shared, padding), dim=-1).masked_fill(~valid, 0)
        section = slice(self.joint_dim, self.joint_dim + self.shared_dim)
        shared = F.linear(
            value[:, section].masked_fill(~valid[:, section], 0),
            self.shared.weight,
            self.shared.bias,
        )
        return joint + shared


@dataclass(frozen=True)
class GoalWAMLayout:
    """Immutable row metadata supplied by the runner, independent of values."""

    goal_rows: torch.Tensor
    video_noisy_rows: torch.Tensor
    action_noisy_rows: torch.Tensor
    action_valid: torch.Tensor
    domains: torch.Tensor
    input_joint_rows: torch.Tensor
    output_joint_rows: torch.Tensor


class GoalWAMTransformer(Cosmos3Transformer):
    def __init__(
        self,
        config: GoalWAMConfig,
        *,
        params_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
    ) -> None:
        # Construct the common towers without allocating unused action bridges.
        common = {
            field.name: getattr(config, field.name) for field in fields(Cosmos3Config)
        }
        super().__init__(
            Cosmos3Config(**{**common, "action_gen": False}),
            params_dtype=params_dtype,
            device=device,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
        )
        # The reference computes one full text prefill before slicing its KV.
        # A separate logical mesh keeps common UND layers unsharded even when
        # the Generator uses TP. It creates no extra process groups.
        del self.language_model
        with use_mesh(register_goalwam_text_mesh()):
            self.language_model = Cosmos3LanguageModel(
                config,
                attn_backend=attn_backend,
                norm_backend=norm_backend,
                params_dtype=params_dtype,
                device=device,
            )
            for index, layer in enumerate(self.language_model.layers):
                layer.self_attn.qkv_proj = GoalWAMTextProjection(
                    config.hidden_size,
                    config.head_dim,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    bias=False,
                    params_dtype=params_dtype,
                    device=device,
                    hf_legs={"q": "to_q", "k": "to_k", "v": "to_v"},
                    prefix=f"language_model.layers.{index}.self_attn.qkv_proj",
                )
        self.config = config
        self.action_gen = True
        target = self.proj_in.weight.device
        self.action_proj_in = SharedEEFLinear(
            config,
            output=False,
            params_dtype=params_dtype,
            device=target,
            prefix="action_proj_in",
        )
        self.action_proj_out = SharedEEFLinear(
            config,
            output=True,
            params_dtype=params_dtype,
            device=target,
            prefix="action_proj_out",
        )
        for name in ("goal_vision_embed", "action_modality_embed"):
            parameter = nn.Parameter(
                torch.zeros(config.hidden_size, dtype=params_dtype, device=target),
                requires_grad=False,
            )
            parameter.hf_keys = [(name, None)]
            parameter.weight_loader = replicated()
            self.register_parameter(name, parameter)
        self.configure_rounding()

    def configure_rounding(self) -> None:
        """Use public layer options for the checkpoint's rounding boundaries."""
        replacements = []
        for name, module in self.named_modules():
            if isinstance(module, DenseMLP):
                module.cast_before_multiply = True
            if isinstance(module, RMSNorm):
                replacements.append(
                    (
                        name,
                        GoalWAMRMSNorm(
                            module.hidden_size,
                            eps=module.variance_epsilon,
                            backend=module.backend,
                            dtype=module.weight.dtype,
                            device=module.weight.device,
                            prefix=module.prefix,
                            cast_before_affine=True,
                        ),
                    )
                )
        for name, module in replacements:
            parent, _, leaf = name.rpartition(".")
            setattr(self.get_submodule(parent), leaf, module)

    def forward(
        self,
        video_patches: torch.Tensor,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        condition: Cosmos3Condition,
        layout: GoalWAMLayout,
        *,
        time_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict velocities for noisy visual patches and action rows only."""
        dtype = self.proj_in.weight.dtype
        visual, _ = self.proj_in(video_patches.to(dtype))
        visual = visual + layout.goal_rows.to(dtype)[:, None] * self.goal_vision_embed
        action = self.action_proj_in(
            actions.to(dtype),
            layout.domains,
            layout.action_valid,
            layout.input_joint_rows,
        )
        action = action + self.action_modality_embed
        if time_embedding is None:
            time = timestep.float().reshape(1) * self.timestep_scale
            time_embedding = (
                self.time_embedder(time.expand(layout.video_noisy_rows.numel())).to(
                    dtype
                ),
                self.time_embedder(time.expand(layout.action_noisy_rows.numel())).to(
                    dtype
                ),
            )
        visual = visual.index_add(
            0,
            layout.video_noisy_rows,
            time_embedding[0],
        )
        action = action.index_add(
            0,
            layout.action_noisy_rows,
            time_embedding[1],
        )
        hidden = torch.cat((visual, action), dim=0).unsqueeze(0)
        rows = torch.cat(
            (layout.video_noisy_rows, layout.action_noisy_rows + visual.shape[0])
        )
        selected = self.forward_selected(hidden, condition, rows).squeeze(0)
        video_hidden, action_hidden = selected.split(
            (layout.video_noisy_rows.numel(), layout.action_noisy_rows.numel())
        )
        video_output, _ = self.proj_out(video_hidden)
        action_rows = layout.action_noisy_rows
        action_output = self.action_proj_out(
            action_hidden,
            layout.domains.index_select(0, action_rows),
            layout.action_valid.index_select(0, action_rows),
            layout.output_joint_rows,
        )
        return video_output, action_output

    def forward_selected(
        self,
        hidden: torch.Tensor,
        condition: Cosmos3Condition,
        output_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Keep full K/V; only the final block narrows queries and output work."""
        normalized = self.gen_layers[0].input_layernorm(hidden)
        cos, sin = condition.freqs_gen
        last = len(self.gen_layers) - 1
        for index, (layer, (key, value)) in enumerate(
            zip(self.gen_layers, condition.cached_kv, strict=True)
        ):
            attention = layer.cross_attention
            query, key, value = prepare_goalwam_attention(
                attention, normalized, key, value, cos, sin
            )
            if index == last:
                query = query.index_select(1, output_rows)
                hidden = hidden.index_select(1, output_rows)
            context = (
                None
                if condition.attention_contexts is None
                else condition.attention_contexts[index]
            )
            attended = attention.attn(query, key, value, context).flatten(-2)
            projected, _ = attention.to_out(attended)
            hidden, normalized = add_goalwam_norm(
                hidden, projected, layer.post_attention_layernorm
            )
            projected = layer.mlp(normalized)
            next_norm = (
                self.norm_moe_gen
                if index == last
                else self.gen_layers[index + 1].input_layernorm
            )
            hidden, normalized = add_goalwam_norm(hidden, projected, next_norm)
        return normalized
