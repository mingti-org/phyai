"""Goal, history, rollout, and action positions for HumanWAM-Goal."""

import math

import torch

from phyai_utils_tools.models.cosmos3.positions_cosmos3 import (
    compute_mrope_position_ids_vision,
)


def humanwam_goal_position_ids(
    *,
    text_tokens: int,
    grids: tuple[tuple[int, int, int], ...],
    history_ages_seconds: tuple[float, ...],
    action_rows: int,
    video_fps: float,
    action_fps: float,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    temporal_modality_margin: int = 15000,
) -> torch.Tensor:
    """Return [3,1,S] positions for unpooled, independently encoded histories."""
    if len(grids) != len(history_ages_seconds) + 2:
        raise ValueError("Provide goal, histories, then the rollout grid")
    if text_tokens < 1 or action_rows < 1 or temporal_compression_factor < 1:
        raise ValueError(
            "Text length, action rows, and temporal compression must be positive"
        )
    if any(
        not math.isfinite(value) or value <= 0
        for value in (video_fps, action_fps, base_fps)
    ):
        raise ValueError("Frame rates must be finite and positive")
    if any(len(grid) != 3 or min(grid) < 1 for grid in grids):
        raise ValueError("Each patch grid must contain positive T,H,W dimensions")
    if any(grid[0] != 1 for grid in grids[:-1]):
        raise ValueError("Goal and histories must be independent single-frame grids")
    if any(not math.isfinite(age) or age <= 0 for age in history_ages_seconds):
        raise ValueError("History ages must be finite and positive")
    offset = text_tokens + temporal_modality_margin
    common = {
        "fps": video_fps,
        "base_fps": base_fps,
        "temporal_compression_factor": temporal_compression_factor,
        "enable_fps_modulation": True,
    }
    goal, origin = compute_mrope_position_ids_vision(
        *grids[0], temporal_offset=offset, **common
    )
    positions = [goal]
    for grid, age in zip(grids[1:-1], history_ages_seconds, strict=True):
        history, _ = compute_mrope_position_ids_vision(
            *grid,
            temporal_offset=origin,
            start_frame_offset=-age * video_fps / temporal_compression_factor,
            **common,
        )
        positions.append(history)
    rollout, _ = compute_mrope_position_ids_vision(
        *grids[-1], temporal_offset=origin, **common
    )
    actions, _ = compute_mrope_position_ids_vision(
        action_rows,
        1,
        1,
        temporal_offset=origin,
        fps=action_fps,
        base_fps=base_fps,
        temporal_compression_factor=1,
        base_temporal_compression_factor=temporal_compression_factor,
        enable_fps_modulation=True,
        start_frame_offset=0,
    )
    return torch.cat((*positions, rollout, actions), dim=1).unsqueeze(1)
