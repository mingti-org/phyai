"""Audit RoboTwin results against the released LingBot V2 model card."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from benchmark.lingbot_v2.robotwin.summarize_success import (
    ANSI_RE,
    TaskResult,
    collect_results,
)

MODEL_CARD_ROW = re.compile(
    r"^\|\s*`?(?P<task>[a-z0-9_]+)`?\s*\|\s*" r"(?P<clean>[0-9]+(?:\.[0-9]+)?)%\s*\|",
    re.MULTILINE,
)
INSTRUCTION_RE = re.compile(r"Eval Instruction:\s*(seen|unseen)", re.IGNORECASE)


def parse_model_card(path: Path) -> dict[str, float]:
    content = path.read_text(encoding="utf-8", errors="replace")
    rates = {
        match.group("task"): float(match.group("clean")) / 100.0
        for match in MODEL_CARD_ROW.finditer(content)
    }
    if not rates:
        raise ValueError(f"no RoboTwin Clean result rows found in {path}")
    return rates


def binomial_two_sided(success: int, episodes: int, expected_rate: float) -> float:
    """Return the exact two-sided binomial p-value for one observed count."""

    observed_probability = (
        math.comb(episodes, success)
        * expected_rate**success
        * (1.0 - expected_rate) ** (episodes - success)
    )
    tolerance = observed_probability * 1e-12 + 1e-18
    result = 0.0
    for count in range(episodes + 1):
        probability = (
            math.comb(episodes, count)
            * expected_rate**count
            * (1.0 - expected_rate) ** (episodes - count)
        )
        if probability <= observed_probability + tolerance:
            result += probability
    return min(result, 1.0)


def wilson_interval(
    success: int, episodes: int, z: float = 1.95996398454
) -> tuple[float, float]:
    rate = success / episodes
    denominator = 1.0 + z * z / episodes
    center = (rate + z * z / (2.0 * episodes)) / denominator
    radius = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / episodes + z * z / (4.0 * episodes * episodes)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def log_instruction_type(result: TaskResult) -> str | None:
    content = ANSI_RE.sub("", result.log.read_text(encoding="utf-8", errors="replace"))
    match = INSTRUCTION_RE.search(content)
    return match.group(1).lower() if match else None


def git_revision(root: Path) -> str | None:
    git_dir = root / ".git"
    if not git_dir.exists():
        return None
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8", errors="replace").strip()
    if not value.startswith("ref: "):
        return value
    ref = git_dir / value.removeprefix("ref: ")
    return (
        ref.read_text(encoding="utf-8", errors="replace").strip()
        if ref.is_file()
        else None
    )


def protocol_findings(
    *,
    lingbot_root: Path,
    robotwin_root: Path,
    results: dict[str, TaskResult],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    launcher = lingbot_root / "experiment/robotwin/start_robotwin_infer_and_eval.sh"
    legacy_client = robotwin_root / "script/eval_polict_client_openpi.py"
    legacy_config = robotwin_root / "policy/ACT/deploy_policy.yml"
    current_client = robotwin_root / "scripts/eval_policy_xpolicylab.py"

    if not launcher.is_file():
        findings.append(
            {
                "severity": "error",
                "code": "official_launcher_missing",
                "detail": str(launcher),
            }
        )
    if not legacy_client.is_file():
        findings.append(
            {
                "severity": "error",
                "code": "official_eval_client_missing",
                "detail": (
                    "The released LingBot V2 launcher calls "
                    "script/eval_polict_client_openpi.py, but that file is absent "
                    "from this RoboTwin tree. The published protocol is not exactly "
                    "reproducible from these two public revisions."
                ),
            }
        )
    if not legacy_config.is_file():
        findings.append(
            {
                "severity": "error",
                "code": "official_act_harness_missing",
                "detail": str(legacy_config),
            }
        )
    if current_client.is_file():
        findings.append(
            {
                "severity": "info",
                "code": "current_xpolicylab_client",
                "detail": str(current_client),
            }
        )

    instruction_types = {
        value for result in results.values() if (value := log_instruction_type(result))
    }
    if instruction_types and instruction_types != {"unseen"}:
        findings.append(
            {
                "severity": "error",
                "code": "instruction_protocol_mismatch",
                "detail": (
                    f"result logs use {sorted(instruction_types)}; the released "
                    "RoboTwin ACT harness config uses instruction_type=unseen"
                ),
            }
        )
    if any(result.episodes != 100 for result in results.values()):
        findings.append(
            {
                "severity": "warning",
                "code": "episode_count_mismatch",
                "detail": (
                    "the model-card rows use 100 accepted episodes per task; "
                    "one or more audited logs use a different denominator"
                ),
            }
        )
    revision = git_revision(robotwin_root)
    if revision is None:
        findings.append(
            {
                "severity": "warning",
                "code": "robotwin_revision_unverifiable",
                "detail": "RoboTwin source has no readable .git revision metadata",
            }
        )
    return findings


def build_report(
    *,
    model_card_rates: dict[str, float],
    results: dict[str, TaskResult],
    lingbot_root: Path,
    robotwin_root: Path,
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for task, result in sorted(results.items()):
        if task not in model_card_rates:
            continue
        expected_rate = model_card_rates[task]
        low, high = wilson_interval(result.success, result.episodes)
        tasks.append(
            {
                "task": task,
                "success": result.success,
                "episodes": result.episodes,
                "observed_rate": result.rate,
                "model_card_rate": expected_rate,
                "difference_pp": (result.rate - expected_rate) * 100.0,
                "wilson_95": [low, high],
                "exact_binomial_p": binomial_two_sided(
                    result.success, result.episodes, expected_rate
                ),
                "instruction_type": log_instruction_type(result),
            }
        )
    if not tasks:
        raise ValueError("no result tasks matched the model-card task table")

    total_success = sum(task["success"] for task in tasks)
    total_episodes = sum(task["episodes"] for task in tasks)
    expected_success = sum(task["episodes"] * task["model_card_rate"] for task in tasks)
    return {
        "provenance": {
            "lingbot_revision": git_revision(lingbot_root),
            "robotwin_revision": git_revision(robotwin_root),
        },
        "summary": {
            "tasks": len(tasks),
            "success": total_success,
            "episodes": total_episodes,
            "observed_rate": total_success / total_episodes,
            "model_card_weighted_rate": expected_success / total_episodes,
            "difference_pp": (total_success - expected_success)
            / total_episodes
            * 100.0,
        },
        "tasks": tasks,
        "findings": protocol_findings(
            lingbot_root=lingbot_root,
            robotwin_root=robotwin_root,
            results=results,
        ),
    }


def print_report(report: dict[str, Any]) -> None:
    provenance = report["provenance"]
    print("===== Source provenance =====")
    print(
        f"LingBot V2 : {provenance['lingbot_revision'] or 'unverifiable'}\n"
        f"RoboTwin   : {provenance['robotwin_revision'] or 'unverifiable'}"
    )
    summary = report["summary"]
    print("\n===== Success gap =====")
    print(
        f"matched tasks : {summary['tasks']}\n"
        f"observed      : {summary['success']}/{summary['episodes']} "
        f"({summary['observed_rate'] * 100:.2f}%)\n"
        f"model card    : {summary['model_card_weighted_rate'] * 100:.2f}%\n"
        f"difference    : {summary['difference_pp']:+.2f} pp"
    )
    print("\n===== Per task =====")
    for task in report["tasks"]:
        low, high = task["wilson_95"]
        print(
            f"{task['task']:<24} "
            f"{task['success']:>3}/{task['episodes']:<3} "
            f"observed={task['observed_rate'] * 100:6.2f}% "
            f"card={task['model_card_rate'] * 100:6.2f}% "
            f"diff={task['difference_pp']:+7.2f}pp "
            f"CI95=[{low * 100:.1f},{high * 100:.1f}] "
            f"p={task['exact_binomial_p']:.4g} "
            f"instruction={task['instruction_type'] or 'unknown'}"
        )
    print("\n===== Protocol findings =====")
    for finding in report["findings"]:
        print(
            f"[{finding['severity'].upper():7}] "
            f"{finding['code']}: {finding['detail']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-card", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    model_card_rates = parse_model_card(args.model_card)
    results, incomplete = collect_results(args.results_dir)
    if incomplete:
        raise RuntimeError(
            "incomplete result logs: " + ", ".join(str(path) for path in incomplete)
        )
    report = build_report(
        model_card_rates=model_card_rates,
        results=results,
        lingbot_root=args.lingbot_root,
        robotwin_root=args.robotwin_root,
    )
    print_report(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nJSON report: {args.json}")


if __name__ == "__main__":
    main()
