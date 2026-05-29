---
name: phyai-local-env-report
description: Generate a local phyai environment report for debugging system, Python, CUDA/GPU, dependency, workspace package, git, and PHYAI_* configuration issues. Use when asked to inspect, summarize, export, or diagnose the current phyai development/runtime environment.
---

# phyai Local Environment Report

Use this skill to produce a reproducible local environment report for the
`phyai` monorepo. Prefer the bundled script over ad hoc command collection.

## Quick Start

From the repository root:

```bash
uv run python .claude/skills/phyai-local-env-report/scripts/collect_env_report.py
```

To save a report:

```bash
uv run python .claude/skills/phyai-local-env-report/scripts/collect_env_report.py --output reports/local-env.md
```

If `uv` is unavailable, run the same script with the active Python:

```bash
python .claude/skills/phyai-local-env-report/scripts/collect_env_report.py
```

## Workflow

1. Run the script from the `phyai` repository root unless the user explicitly
   names another checkout.
2. Share the report path when `--output` is used, or summarize the most relevant
   findings from stdout.
3. For installation/CUDA failures, look first at:
   - `Diagnostics`
   - `CUDA / GPU`
   - `Python Packages`
   - `phyai Runtime Environment`
4. Do not install packages, mutate the repo, or run heavyweight builds just to
   create the report unless the user explicitly asks.

## Script Options

```bash
uv run python .claude/skills/phyai-local-env-report/scripts/collect_env_report.py --help
```

Common options:

- `--output PATH`: write the Markdown or JSON report to a file.
- `--format markdown|json`: choose the output format; Markdown is default.
- `--no-gpu-detail`: skip slower GPU detail commands such as
  `nvidia-smi topo -m`.

## Report Coverage

The script captures:

- host, OS, Python executable/version, and key tool paths;
- uv workspace package versions and import status for phyai packages;
- selected Python package versions relevant to phyai, CUDA, kernels, and models;
- CUDA toolkit/runtime hints, Torch CUDA state, GPUs, driver data, and topology;
- registered `PHYAI_*` env vars from `phyai.env` plus extra process-level
  `PHYAI_*` variables used elsewhere in the codebase;
- git branch, commit, and dirty state;
- existing `phyai-kernel show-env` output when available.
