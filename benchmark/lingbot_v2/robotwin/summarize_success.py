"""Summarize Official and PHYAI RoboTwin success logs as CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SUCCESS_RE = re.compile(
    r"(?:Final\s+)?Success\s+rate:\s*(?P<success>\d+)\s*/\s*"
    r"(?P<episodes>\d+)\s*(?:=>|=)\s*(?P<reported>[0-9.eE+-]+)",
    re.IGNORECASE,
)
INVALID_LOG_MARKERS = ("Policy rollout error:",)


@dataclass(frozen=True)
class TaskResult:
    """Completed success counts and source log for one evaluated task."""

    task: str
    success: int
    episodes: int
    log: Path

    @property
    def rate(self) -> float:
        """Return the task's successful-episode fraction."""

        return self.success / self.episodes


def collect_results(root: Path) -> tuple[dict[str, TaskResult], list[Path]]:
    """Collect completed task results and identify incomplete log files."""

    if not root.is_dir():
        raise FileNotFoundError(f"result directory does not exist: {root}")
    results: dict[str, TaskResult] = {}
    incomplete: list[Path] = []
    logs = sorted(root.rglob("eval_logs/*.log"))
    if not logs:
        raise FileNotFoundError(f"no eval_logs/*.log files found under {root}")
    for log in logs:
        content = ANSI_RE.sub("", log.read_text(encoding="utf-8", errors="replace"))
        if any(marker in content for marker in INVALID_LOG_MARKERS):
            incomplete.append(log)
            continue
        matches = list(SUCCESS_RE.finditer(content))
        if not matches:
            incomplete.append(log)
            continue
        match = matches[-1]
        episodes = int(match.group("episodes"))
        if episodes <= 0:
            incomplete.append(log)
            continue
        task = log.stem
        result = TaskResult(
            task=task,
            success=int(match.group("success")),
            episodes=episodes,
            log=log,
        )
        previous = results.get(task)
        if previous is not None and previous.log != log:
            raise RuntimeError(
                f"multiple completed logs found for task {task!r}: "
                f"{previous.log}, {log}"
            )
        results[task] = result
    return results, incomplete


def aggregate(results: dict[str, TaskResult]) -> tuple[float, float, int, int]:
    """Compute macro accuracy, micro accuracy, and aggregate counts."""

    if not results:
        raise ValueError("cannot aggregate an empty result set.")
    total_success = sum(result.success for result in results.values())
    total_episodes = sum(result.episodes for result in results.values())
    macro = sum(result.rate for result in results.values()) / len(results)
    micro = total_success / total_episodes
    return macro, micro, total_success, total_episodes


def write_csv(
    path: Path,
    official: dict[str, TaskResult],
    phyai: dict[str, TaskResult],
) -> None:
    """Write per-task Official/PHYAI results as a CSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted(set(official) | set(phyai))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "task",
                "official_success",
                "official_episodes",
                "official_rate",
                "phyai_success",
                "phyai_episodes",
                "phyai_rate",
                "phyai_minus_official",
            )
        )
        for task in tasks:
            left = official.get(task)
            right = phyai.get(task)
            writer.writerow(
                (
                    task,
                    "" if left is None else left.success,
                    "" if left is None else left.episodes,
                    "" if left is None else f"{left.rate:.6f}",
                    "" if right is None else right.success,
                    "" if right is None else right.episodes,
                    "" if right is None else f"{right.rate:.6f}",
                    (
                        ""
                        if left is None or right is None
                        else f"{right.rate - left.rate:+.6f}"
                    ),
                )
            )


def write_markdown(
    path: Path,
    official: dict[str, TaskResult],
    phyai: dict[str, TaskResult],
) -> None:
    """Write the paired backend comparison as a Markdown report."""

    left_macro, left_micro, left_success, left_episodes = aggregate(official)
    right_macro, right_micro, right_success, right_episodes = aggregate(phyai)
    lines = [
        "# LingBot V2 RoboTwin success comparison",
        "",
        "| Backend | Tasks | Success / Episodes | Micro Avg Acc | Macro Avg Acc |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Official | {len(official)} | {left_success} / {left_episodes} | "
            f"{left_micro * 100:.2f}% | {left_macro * 100:.2f}% |"
        ),
        (
            f"| PHYAI | {len(phyai)} | {right_success} / {right_episodes} | "
            f"{right_micro * 100:.2f}% | {right_macro * 100:.2f}% |"
        ),
        "",
        f"PHYAI - Official micro Avg Acc: **{(right_micro - left_micro) * 100:+.2f} pp**",
        "",
        "| Task | Official | PHYAI | Difference |",
        "|---|---:|---:|---:|",
    ]
    for task in sorted(set(official) | set(phyai)):
        left = official.get(task)
        right = phyai.get(task)
        left_text = "missing" if left is None else f"{left.rate * 100:.2f}%"
        right_text = "missing" if right is None else f"{right.rate * 100:.2f}%"
        difference = (
            "n/a"
            if left is None or right is None
            else f"{(right.rate - left.rate) * 100:+.2f} pp"
        )
        lines.append(f"| {task} | {left_text} | {right_text} | {difference} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Parse result directories and write both summary report formats."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--phyai-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    official, official_incomplete = collect_results(args.official_dir)
    phyai, phyai_incomplete = collect_results(args.phyai_dir)
    if (official_incomplete or phyai_incomplete) and not args.allow_incomplete:
        lines = ["incomplete RoboTwin logs were found:"]
        lines.extend(f"  official: {path}" for path in official_incomplete)
        lines.extend(f"  phyai: {path}" for path in phyai_incomplete)
        raise RuntimeError("\n".join(lines))
    if set(official) != set(phyai) and not args.allow_incomplete:
        raise RuntimeError(
            "Official and PHYAI task sets differ: "
            f"official_only={sorted(set(official) - set(phyai))}, "
            f"phyai_only={sorted(set(phyai) - set(official))}"
        )

    write_csv(args.csv, official, phyai)
    write_markdown(args.markdown, official, phyai)
    print(args.markdown.read_text(encoding="utf-8"))
    print(f"CSV      : {args.csv}")
    print(f"Markdown : {args.markdown}")


if __name__ == "__main__":
    main()
