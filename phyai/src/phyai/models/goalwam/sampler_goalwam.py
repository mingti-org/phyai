"""Load immutable inference coefficients into the shared UniPC sampler."""

from dataclasses import replace

import torch

from phyai.models.goalwam.configuration_goalwam import GoalWAMSamplingConfig
from phyai.models.cosmos3.model_runner_rollout_cosmos3 import (
    Cosmos3UniPCSchedule,
    build_native_unipc_schedule,
)


def build_goalwam_schedule(
    config: GoalWAMSamplingConfig, device: torch.device
) -> Cosmos3UniPCSchedule:
    if config.corrector_weights is None:
        raise ValueError(
            "sampling.json must include the reference UniPC corrector_weights "
            "for its num_steps and flow_shift"
        )
    schedule = build_native_unipc_schedule(
        num_steps=config.num_steps,
        flow_shift=config.flow_shift,
        use_karras_sigmas=False,
        device=device,
        framework_compat=True,
    )
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
