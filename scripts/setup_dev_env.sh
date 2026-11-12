#!/usr/bin/env bash
# Bootstrap the phyai development environment. Idempotent — safe to re-run.
set -euo pipefail

# Run from the repo root regardless of the caller's CWD, so the scratch dirs
# below land at the repo root and `uv` / `pre-commit` see the workspace.
cd "$(dirname "$0")/.."

# Workspace Python deps: creates the repo-root .venv and installs all five
# workspace members editable (phyai, phyai-kernel, phyai-ext, ...).
git submodule update --init third_party/cutlass
uv sync

# DIR for Agent: .memory/ (agent work logs) and .profile/ (profiling output).
# Both are gitignored; -p keeps a re-run from failing when they already exist.
mkdir -p .memory .profile

# For CI/CD: pre-commit hooks (clang-format, codespell, ruff-format).
uv tool install pre-commit
pre-commit install
pre-commit install-hooks

# Document: the docs site under docs/ is built with Mintlify (config in
# docs/docs.json). Its dev dependency is the Mintlify CLI, published on npm as
# `mint` (formerly `mintlify`); it needs Node >= 19.
#   cd docs && mint dev   # preview at http://localhost:3000
#   cd docs && mint broken-links
# For Mintlify authoring help inside Claude Code, also install the skill:
#   npx skills add https://mintlify.com/docs
if command -v npm >/dev/null 2>&1; then
  npm install -g mint
else
  echo "warn: npm not found; skipping Mintlify CLI install (see docs/AGENTS.md)" >&2
fi
