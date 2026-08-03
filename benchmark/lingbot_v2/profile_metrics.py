from __future__ import annotations

from collections.abc import Mapping


def normalized_share_pct(values: Mapping[str, float]) -> dict[str, float]:
    """Normalize non-overlapping profiled stages to a 100 percent total."""

    total = sum(values.values())
    if total <= 0:
        return {name: 0.0 for name in values}
    return {name: 100.0 * value / total for name, value in values.items()}


def profile_diagnostics(
    baseline_mean_ms: float,
    stages: Mapping[str, float],
    *,
    max_relative_delta: float = 0.15,
) -> dict[str, float | bool | str]:
    """Check whether instrumentation materially perturbed one inference."""

    if baseline_mean_ms <= 0:
        raise ValueError("baseline_mean_ms must be positive.")
    if max_relative_delta < 0:
        raise ValueError("max_relative_delta must be non-negative.")

    profiled_stage_sum_ms = sum(stages.values())
    overhead_ratio = profiled_stage_sum_ms / baseline_mean_ms
    relative_delta = abs(profiled_stage_sum_ms - baseline_mean_ms) / baseline_mean_ms
    valid = profiled_stage_sum_ms > 0 and relative_delta <= max_relative_delta
    reason = (
        "instrumented scheduler-stage sum is within the allowed baseline delta"
        if valid
        else "instrumentation perturbation exceeds the allowed baseline delta"
    )
    return {
        "valid": valid,
        "reason": reason,
        "baseline_mean_ms": baseline_mean_ms,
        "profiled_stage_sum_ms": profiled_stage_sum_ms,
        "overhead_ratio": overhead_ratio,
        "relative_delta": relative_delta,
        "max_relative_delta": max_relative_delta,
    }


def scale_details_to_parent_stages(
    details: Mapping[str, float],
    instrumented_parents: Mapping[str, float],
    target_parents: Mapping[str, float],
    detail_parents: Mapping[str, str],
) -> dict[str, float]:
    """Remove detail-hook overhead with one scale factor per parent stage."""

    result: dict[str, float] = {}
    for detail, value in details.items():
        parent = detail_parents[detail]
        source = instrumented_parents[parent]
        target = target_parents[parent]
        result[detail] = value * target / source if source > 0 else 0.0
    return result
