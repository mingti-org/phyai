"""HumanWAM-Goal inference input processing."""

from phyai_utils_tools.models.humanwam_goal.positions_humanwam_goal import (
    humanwam_goal_position_ids,
)
from phyai_utils_tools.models.humanwam_goal.processor_humanwam_goal import (
    HumanWAMGoalProcessor,
    HumanWAMGoalProcessorConfig,
)

__all__ = [
    "HumanWAMGoalProcessor",
    "HumanWAMGoalProcessorConfig",
    "humanwam_goal_position_ids",
]
