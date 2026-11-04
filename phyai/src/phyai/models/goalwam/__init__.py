"""Goal-image-conditioned action inference with Shared-EEF projections."""

from phyai.models.goalwam.requests_goalwam import (
    GoalWAMTextRequest,
    GoalWAMEncodeRequest,
    GoalWAMDenoiseRequest,
    GoalWAMObservationRequest,
)
from phyai.models.goalwam.configuration_goalwam import (
    GoalWAMConfig,
    GoalWAMSamplingConfig,
)

__all__ = [
    "GoalWAMConfig",
    "GoalWAMDenoiseRequest",
    "GoalWAMEncodeRequest",
    "GoalWAMObservationRequest",
    "GoalWAMSamplingConfig",
    "GoalWAMTextRequest",
]
