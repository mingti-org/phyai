"""Build immutable inference coefficients for the UniPC sampler."""

from dataclasses import replace

import torch

from phyai.models.humanwam_goal.configuration_humanwam_goal import (
    HumanWAMGoalSamplingConfig,
)
from phyai.models.cosmos3.model_runner_rollout_cosmos3 import (
    Cosmos3UniPCSchedule,
    build_native_unipc_schedule,
)


def build_humanwam_goal_schedule(
    config: HumanWAMGoalSamplingConfig, device: torch.device
) -> Cosmos3UniPCSchedule:
    schedule = build_native_unipc_schedule(
        num_steps=config.num_steps,
        flow_shift=config.flow_shift,
        use_karras_sigmas=False,
        device=device,
        framework_compat=True,
    )
    if config.corrector_weights is None:
        return schedule
    steps = []
    for step, weights in zip(schedule.steps, config.corrector_weights, strict=True):
        corrector = step.corrector
        if corrector is not None:
            corrector = replace(
                corrector,
                rhos=torch.tensor(weights, dtype=torch.float32, device=device),
            )
        steps.append(replace(step, corrector=corrector))
    return replace(schedule, steps=tuple(steps))
