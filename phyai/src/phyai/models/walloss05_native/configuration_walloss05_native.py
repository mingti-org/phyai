"""Configuration for the experimental WALL-OSS-0.5 native PHYAI port."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from phyai.models.configuration import PretrainedConfig


def _as_int_dict(value: Mapping[str, Any] | None) -> dict[str, int]:
    if value is None:
        return {}
    return {str(k): int(v) for k, v in value.items()}


@dataclass(frozen=True)
class WallOSS05NativeConfig(PretrainedConfig):
    """Frozen config with checkpoint fields plus inference/train-config overlay.

    The public WALL-OSS-0.5 config.json does not contain dof_config or
    agent_pos_config. Official Wall-X inference injects those fields from the
    train config before constructing ActionProcessor. This class mirrors that
    behavior for the native PHYAI port.
    """

    model_type: str = "qwen2_5_vl"

    vocab_size: int = 151936
    hidden_size: int = 2048
    action_hidden_size: int = 1024
    state_hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    max_position_embeddings: int = 128000
    rope_theta: float = 1000000.0
    rope_scaling: Mapping[str, Any] | None = None
    attention_dropout: float = 0.0
    _attn_implementation: str = "flash_mask"
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    num_experts: int = 2
    experts: Any = None
    dim_inputs: tuple[int, ...] = (2048, 1024)

    attention_moe: bool = True
    mlp_moe: bool = True
    norm_moe: bool = True
    mot_opt: bool = True
    causal_action_attention_mask: bool = True

    use_state_string_representation: bool = False
    use_adarms: bool = False
    adarms_cond_dim: int | None = None
    proj_with_mask: bool = True
    use_flow_action_expert: bool = True
    use_x_pred: bool = False
    use_x_loss: bool = True
    flow_loss_weight: float = 1.0
    ar_loss_weight: float = 0.01

    action_horizon: int = 10
    action_horizon_flow: int = 10
    norm_key: str = "x2_normal"

    noise_scheduler: Mapping[str, Any] = field(
        default_factory=lambda: {
            "beta_alpha": 1.5,
            "beta_beta": 1.0,
            "s": 0.999,
            "num_inference_timesteps": 10,
        }
    )

    dof_config: Mapping[str, int] = field(default_factory=dict)
    agent_pos_config: Mapping[str, int] = field(default_factory=dict)

    vision_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.dim_inputs) != self.num_experts:
            raise ValueError(
                f"dim_inputs length {len(self.dim_inputs)} must equal num_experts {self.num_experts}"
            )
        if self.action_hidden_size <= 0 or self.hidden_size <= 0:
            raise ValueError("hidden sizes must be positive")
        if self.dof_config and self.action_dim_internal <= 0:
            raise ValueError("dof_config must define a positive action dimension")
        if self.agent_pos_config and self.propri_dim_internal <= 0:
            raise ValueError("agent_pos_config must define a positive proprio dimension")

    @property
    def action_dim_internal(self) -> int:
        return sum(int(v) for v in self.dof_config.values())

    @property
    def propri_dim_internal(self) -> int:
        return sum(int(v) for v in self.agent_pos_config.values())

    @classmethod
    def from_checkpoint_and_train_config(
        cls,
        checkpoint_config: Mapping[str, Any],
        train_config: Mapping[str, Any],
        *,
        norm_key: str = "x2_normal",
    ) -> "WallOSS05NativeConfig":
        """Build config from checkpoint config.json and LIBERO-style train config."""
        base = cls.from_dict(dict(checkpoint_config))
        return base.with_train_config_overlay(train_config, norm_key=norm_key)

    def with_train_config_overlay(
        self,
        train_config: Mapping[str, Any],
        *,
        norm_key: str = "x2_normal",
    ) -> "WallOSS05NativeConfig":
        task = dict(train_config.get("task", {}) or {})
        model = dict(train_config.get("model", {}) or {})
        data = dict(train_config.get("data", {}) or {})

        dof_config = _as_int_dict(train_config.get("dof_config") or task.get("dof_config"))
        agent_pos_config = _as_int_dict(
            train_config.get("agent_pos_config") or task.get("agent_pos_config")
        )

        action_horizon = int(
            train_config.get("action_horizon")
            or task.get("action_horizon")
            or self.action_horizon
        )
        action_horizon_flow = int(
            train_config.get("action_horizon_flow")
            or data.get("action_horizon_flow")
            or task.get("action_horizon_flow")
            or self.action_horizon_flow
        )

        use_state_string_representation = bool(
            train_config.get("use_state_string_representation")
            if "use_state_string_representation" in train_config
            else task.get(
                "use_state_string_representation",
                data.get("use_state_string_representation", self.use_state_string_representation),
            )
        )

        attn_impl = getattr(self, "_attn_implementation", "flash_mask")
        if train_config.get("_attn_implementation", None) is not None:
            attn_impl = str(train_config["_attn_implementation"])

        return replace(
            self,
            dof_config=dof_config,
            agent_pos_config=agent_pos_config,
            action_horizon=action_horizon,
            action_horizon_flow=action_horizon_flow,
            use_state_string_representation=use_state_string_representation,
            flow_loss_weight=float(model.get("flow_loss_weight", self.flow_loss_weight)),
            ar_loss_weight=float(model.get("ar_loss_weight", self.ar_loss_weight)),
            norm_key=str(norm_key),
            _attn_implementation=attn_impl,
        )


__all__ = ["WallOSS05NativeConfig"]
