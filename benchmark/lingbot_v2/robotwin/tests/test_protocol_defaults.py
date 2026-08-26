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
