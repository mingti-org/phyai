from pathlib import Path


ROBOTWIN_DIR = Path(__file__).resolve().parents[1]


def test_runners_default_to_official_instruction_split() -> None:
    for name in ("run_current_robotwin.sh", "run_pair.sh"):
        content = (ROBOTWIN_DIR / name).read_text(encoding="utf-8")
        assert "instruction_type=unseen" in content
        assert "default: unseen (Official ACT config)" in content
        assert "execution_mode=chunk" in content


def test_current_runner_passes_official_instruction_split() -> None:
    content = (ROBOTWIN_DIR / "run_current_robotwin.sh").read_text(encoding="utf-8")
    assert '--instruction_type "$instruction_type"' in content
    assert "instruction_args" not in content


def test_documented_server_mode_matches_official_cli_default() -> None:
    readme = (ROBOTWIN_DIR / "README.md").read_text(encoding="utf-8")
    assert "--chunk_ret true" in readme
    assert "complete 50-step action chunk" in readme


def test_run_pair_header_reports_the_effective_task_selection(tmp_path) -> None:
    """``--tasks`` overrides ``--num-tasks``, so the header must show the list.

    Only the header printed before the first backend run is under test, so the
    runner and summarizer are stubbed and the simulator tree is empty.
    """

    import subprocess

    phyai_root = tmp_path / "phyai"
    runner_dir = phyai_root / "benchmark" / "lingbot_v2" / "robotwin"
    runner_dir.mkdir(parents=True)
    (runner_dir / "run_current_robotwin.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    (runner_dir / "summarize_success.py").write_text("", encoding="utf-8")

    robotwin_root = tmp_path / "robotwin"
    (robotwin_root / "scripts").mkdir(parents=True)
    (robotwin_root / "scripts" / "eval_policy_xpolicylab.py").write_text(
        "", encoding="utf-8"
    )

    def header(*extra: str) -> str:
        result = subprocess.run(
            [
                "bash",
                str(ROBOTWIN_DIR / "run_pair.sh"),
                "--phyai-root",
                str(phyai_root),
                "--robotwin-root",
                str(robotwin_root),
                "--output",
                str(tmp_path / "out"),
                *extra,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    # An explicit list is what the run will actually evaluate.
    explicit = header("--tasks", "lift_pot,open_microwave")
    assert "tasks            : lift_pot,open_microwave" in explicit

    # Without --tasks the count is the selection.
    default = header("--num-tasks", "3")
    assert "tasks            : 3" in default


def test_protocol_manifest_fields_are_documented() -> None:
    """Every field the runner writes into ``protocol.env`` must be documented.

    The README previously listed four of them; a mismatch in any of the others
    also rejects the directory, so the omission was misleading.
    """

    runner = (ROBOTWIN_DIR / "run_current_robotwin.sh").read_text(encoding="utf-8")
    readme = (ROBOTWIN_DIR / "README.md").read_text(encoding="utf-8")

    block = runner.split('protocol_content="$(cat <<EOF', 1)[1].split("EOF", 1)[0]
    fields = [
        line.split("=", 1)[0].strip() for line in block.splitlines() if "=" in line
    ]
    assert fields, "no manifest fields parsed from run_current_robotwin.sh"

    missing = [name for name in fields if f"`{name}`" not in readme]
    assert not missing, f"README does not document manifest fields: {missing}"

    assert "whole manifest is compared" in readme
