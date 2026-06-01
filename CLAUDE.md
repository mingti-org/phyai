# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

`phyai` is a **uv workspace monorepo** (Python ≥3.12) with five members declared in the top-level `pyproject.toml`:

| Package | Role |
| --- | --- |
| `phyai/` | Main library: engine, models, layers, runtime, parallel, vgpu, cache, weights, payload, tokenizer. Hatchling build. |
| `phyai-kernel/` | JIT-compiled CPU/CUDA kernels via `apache-tvm-ffi`, plus Triton kernels (RMSNorm/LayerNorm/AdaRMSNorm/MaskedEmbedding). Hatchling build. |
| `phyai-ext/` | CMake AOT-compiled C++ extensions (currently radix-cache prefix sharing). scikit-build-core build. |
| `phyai-model-optimizer/` | Placeholder, no source yet. |
| `phyai-utils-tools/` | Placeholder, no source yet. |

Hard pins: `torch==2.11`, `flashinfer-python==v0.6.11.post3`, `transformers==5.8.1`. Don't bump these casually — green-context and flashinfer behaviour is tied to these versions.

## Common commands

Everything goes through `uv`.

```bash
# Install / sync the workspace (creates .venv at repo root)
# uv sync will install all phyai related editable.
uv sync

# Run the full test suite (pyproject.toml testpaths covers all 5 packages)
uv run pytest

# Run a single test module / test
uv run pytest phyai/tests/layers/attention/test_static_cached_attention.py
uv run pytest phyai/tests/parallel/test_collectives.py::test_all_reduce_basic

# End-to-end pi0.5 demo (edit PI05_BASE_WEIGHTS in the script first)
uv run python examples/run_pi05.py

# Pre-commit: clang-format (C/C++, excludes third_party/), codespell, ruff-format
scripts/setup_dev_env.sh            # one-time: install pre-commit hooks
scripts/run_pre_commit.sh           # `pre-commit run --all-files`

# Kernel env probe
uv run phyai-kernel show-env
```

`phyai-ext` is built automatically by `uv sync` via scikit-build-core; the `tool.uv.cache-keys` block in `phyai-ext/pyproject.toml` invalidates the build when CMake / C++ / CUDA sources change.

CPU is the default for tests — `phyai/tests/conftest.py` autouses a fixture that overrides `EngineConfig.device.target = "cpu"` for every test. CUDA tests opt back in explicitly (passing `device="cuda"` or `.cuda()`-ing the module).

## Conventions

- C/C++: clang-format from `.clang-format` (column 128, 2-space indent, `PointerAlignment: Left`). clang-tidy from `.clang-tidy` (google + modernize + performance, `WarningsAsErrors: '*'`, identifier naming enforced: classes `CamelCase`, variables `lower_case`, globals `UPPER_CASE`). C++20 (`add_compile_options(-std=c++20)` in `phyai-ext/CMakeLists.txt`).
- Python: `ruff-format` via pre-commit. Public-by-default — `_` prefix only for genuine implementation details. Singletons exposed via `get_*()` getters, not module-level instances.

## More Conventions provide by human

- all log function in phyai package should use phyai.utils' logging api. U judge using `this_rank_log` or `all_rank_log`
- using `flashinfer` by default if CP is not set. When CP is set, pls using `MagiAttention` whose github repo is https://github.com/SandAI-org/MagiAttention.

## SKILLS! Use SKILLS if needed!

- use .claude/skills/profile_model when user want to profile a model, or want to know the roofline of one model on one hardware setting.
- use .claude/skills/solve_pr_comments when user want u to solve some PR comments.
- use .claude/skills/phyai-model-arch-research when user provides a paper, article, link, model name, checkpoint, or code repo and wants research on the related model architecture.
- use .claude/skills/phyai-local-env-report when user wants a local environment report or diagnostics for system info, CUDA/GPU state, Torch/dependency versions, phyai package versions, git state, or `PHYAI_*` env vars. Prefer its bundled `scripts/collect_env_report.py` over ad hoc env commands.
- use .claude/skills/phyai-communicate-with-memory when user provides a phyai `.memory` file, directory, pasted memory content, or memory artifact path and wants to know what it did. Parse the memory, locate the referenced code repo if it exists, verify claims against code/git/tests, and clearly separate confirmed facts from memory claims and unknowns.

## Agent memory log

All coding agents working in this repo should keep a concise work log under `.memory/` so other agents can understand what happened without re-discovering context.

- Record every meaningful action: files changed, commands run, decisions made, blockers found, validation results, and follow-up items.
- Prefer one Markdown file per task/session, named like `.memory/YYYYMMDD-HHMMSS-brief-task-name.md`.
- Keep entries concise and factual. Link to repo files when useful.
- Do not paste secrets, credentials, huge command outputs, or generated build artifacts. Summarize long outputs and point to files if needed.
- Update the memory file as work progresses, especially before handing off, pausing, or finishing.
- Other agents should read relevant `.memory/` notes before continuing related work.
- If user want to add documents, pls use mintlify skills set. It those skills are not installed. pls use `npx skills add https://mintlify.com/docs` to install it first.
