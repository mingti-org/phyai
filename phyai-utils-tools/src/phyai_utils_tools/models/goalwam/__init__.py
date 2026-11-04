"""GoalWAM inference input processing."""

from phyai_utils_tools.models.goalwam.positions_goalwam import goalwam_position_ids
from phyai_utils_tools.models.goalwam.processor_goalwam import (
    GoalWAMProcessor,
    GoalWAMProcessorConfig,
)

__all__ = ["GoalWAMProcessor", "GoalWAMProcessorConfig", "goalwam_position_ids"]
