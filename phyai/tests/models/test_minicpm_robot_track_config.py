"""Configuration validation for MiniCPM-RobotTrack."""

from __future__ import annotations

import pytest
from phyai.models.minicpm_robot_track.configuration_minicpm_robot_track import (
    MiniCPM4TrackConfig,
    MiniCPMRobotTrackConfig,
)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("coarse_tokens_per_frame", 0),
        ("coarse_tokens_per_frame", -1),
        ("coarse_tokens_per_frame", 3),
        ("fine_tokens_current_frame", 0),
        ("fine_tokens_current_frame", 63),
    ),
)
def test_visual_token_counts_must_be_positive_squares(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="positive square number"):
        MiniCPMRobotTrackConfig(**{field: value})


def test_longrope_maximum_cannot_be_smaller_than_original() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        MiniCPM4TrackConfig(
            max_position_embeddings=4096,
            original_max_position_embeddings=8192,
        )


def test_text_capacity_cannot_exceed_checkpoint_limit() -> None:
    with pytest.raises(ValueError, match="exceeds the checkpoint"):
        MiniCPMRobotTrackConfig(input_seq_length=512)
