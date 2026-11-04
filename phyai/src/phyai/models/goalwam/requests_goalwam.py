"""Single-request inputs for the GoalWAM inference stages."""

from dataclasses import dataclass

import torch


@dataclass
class GoalWAMTextRequest:
    text_ids: torch.Tensor
    text_mask: torch.Tensor
    offload_after: bool = True


@dataclass
class GoalWAMEncodeRequest:
    """Independent RGB frames [1,3,1,H,W], normalized to [-1,1]."""

    pixels: tuple[torch.Tensor, ...]


@dataclass
class GoalWAMDenoiseRequest:
    """Normalized visual/action conditions with explicit packed positions.

    Visual streams are [1,C,T,H,W], ordered goal, histories, rollout. Action
    row zero is the measured state; subsequent rows are the action horizon.
    Noisy masks and domain IDs remain on CPU for layout validation. RTC prefixes
    are supplied as clean action rows. Positions are [3,1,packed_tokens].
    """

    vision: tuple[torch.Tensor, ...]
    video_noisy_masks: tuple[torch.Tensor, ...]
    video_roles: tuple[int, ...]
    action: torch.Tensor
    action_noisy_mask: torch.Tensor
    action_valid_mask: torch.Tensor
    action_domains: torch.Tensor
    gen_position_ids: torch.Tensor
    seed: int = 0


@dataclass
class GoalWAMObservationRequest:
    """Prepared camera frames and normalized actions from GoalWAMProcessor.

    Pixels are independent [1,3,1,H,W] goal/history/current frames. Action
    values already use the checkpoint's normalization and coordinate system.
    """

    pixels: tuple[torch.Tensor, ...]
    history_ages_seconds: tuple[float, ...]
    action: torch.Tensor
    action_noisy_mask: torch.Tensor
    action_valid_mask: torch.Tensor
    action_domains: torch.Tensor
    seed: int = 0
    video_fps: float = 7.5
    action_fps: float = 30.0


GoalWAMStageRequest = (
    GoalWAMTextRequest
    | GoalWAMEncodeRequest
    | GoalWAMDenoiseRequest
    | GoalWAMObservationRequest
)
