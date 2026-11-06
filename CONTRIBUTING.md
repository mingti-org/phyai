# Contributing to PhyAI

PhyAI accepts bug reports, documentation fixes, model support, runtime
improvements, and kernel work.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) in project spaces and
review discussions.

## Before you start

Search the [issue tracker](https://github.com/mingti-org/phyai/issues) before
opening a new issue or starting a large change. For changes that affect public
APIs, shared layers, runtime behavior, checkpoint formats, or hard dependency
pins, open an issue first and describe the proposed contract. A short design
discussion can prevent incompatible work in these shared areas.

Keep each pull request focused on one problem. If a change needs unrelated
cleanup, submit that cleanup separately unless it is required for the fix.

## Development setup

PhyAI is a `uv` workspace and requires Python 3.12 or newer. A working C/C++
toolchain is needed to build `phyai-ext`. The test suite also requires a
CUDA-capable NVIDIA GPU.

Fork the repository, clone your fork, and add the main repository as an
upstream remote:

```bash
git clone https://github.com/<your-user>/phyai.git
cd phyai
git remote add upstream https://github.com/mingti-org/phyai.git
git switch -c fix/short-description
```

The setup script installs the editable workspace and pre-commit hooks. It also
installs the Mintlify CLI when `npm` is available.

```bash
scripts/setup_dev_env.sh
```

To install only the Python workspace, run:

```bash
git submodule update --init third_party/cutlass
uv sync
```

`uv sync` builds `phyai-ext` through scikit-build-core. Re-run it after changing
CMake files, extension sources, or workspace dependencies.

You can inspect the active CUDA and kernel environment with:

```bash
uv run phyai-kernel show-env
```

## Repository layout

| Path | Contents |
| --- | --- |
| `phyai/` | Engine, models, shared layers, runtime, parallelism, kernel selection, and weight loading |
| `phyai-kernel/` | Triton and JIT kernel implementations, tests, and benchmarks |
| `phyai-ext/` | CMake-built C++ extensions with Python bindings |
| `phyai-model-optimizer/` | Model optimization package |
| `phyai-utils-tools/` | Processing and utility package that does not depend on `phyai` |
| `docs/` | Mintlify documentation site |
| `examples/` | Runnable examples and kernel policy files |
| `scripts/` | Development, build, and release helpers |

Package code belongs under `src/<package>`, and tests belong in that package's
`tests/` directory.

## Code guidelines

Follow the style of the code around your change. Prefer existing shared layers,
helpers, and kernel interfaces over model-specific copies.

Python code is formatted with `ruff-format`. Add type annotations where they
make interfaces and tensor contracts clearer. C and C++ code follows
`.clang-format`, uses C++20, and is checked with the repository's clang-tidy
rules.

Write code comments in English. Comments should explain a constraint or a
non-obvious decision rather than narrate the code.

Inside the `phyai` package, use the project logging API:

```python
from phyai.utils import get_logger

logger = get_logger(__name__)
```

Use `logger.info_rank0(...)` for messages that should appear once,
`logger.info(...)` for messages from every rank, and `logger.warning_once(...)`
on per-request or per-layer paths. Do not use `print` or Python's standard
`logging.getLogger` in library code.

Declare new `PHYAI_*` environment variables in `phyai/src/phyai/env.py`. Do not
read environment variables ad hoc. Dependency changes must be deliberate,
especially changes to Torch, FlashInfer, Transformers, or CUDA-related
packages. Include the corresponding `uv.lock` change and explain why it is
needed in the pull request.

Do not commit checkpoints, generated traces, profiler output, credentials,
personal directory paths, or machine-specific hardware reports.

## Model contributions

Model implementations use three separate layers:

```text
modeling_<model>.py       architecture and one forward pass
model_runner_<model>.py  runtime state, caches, and CUDA Graph capture
scheduler_*_<model>.py   sampling, denoising, and request orchestration
```

Keep KV cache state and graph capture out of modeling classes. Sampling and
noise schedules belong in the scheduler. Use shared attention, normalization,
RoPE, linear, and quantization layers instead of implementing them in a model
directory.

New model support must be checked against its reference implementation with the
same checkpoint, processed inputs, dtype policy, schedule, and random inputs.
Compare intermediate tensors before relying on an end result. Record the exact
commands and precision metrics in the pull request.

CI does not have the resources for full-checkpoint model tests. Keep those
scripts and artifacts under `.cache/`, which is not committed. Commit focused
layer tests only when the change adds or changes a reusable layer.

## Kernel contributions

Kernel changes need a correct reference path and tests for the dtype, shape,
layout, and hardware predicates they claim to support. Keep capability checks
in the kernel catalog or policy system rather than adding device branches at
model call sites.

Test CUDA Graph capture when a kernel is used in captured execution. A fused
kernel must preserve the documented rounding and output-dtype contract of the
path it replaces.

For performance claims, report the GPU, software versions, tensor shapes,
dtypes, warmup, repeat count, and before-and-after measurements. Check GPU use
with `nvidia-smi` before benchmarking and avoid collecting latency numbers on a
busy device. Include correctness results with every benchmark.

## Tests and checks

Start with the smallest test that covers the change, then broaden the test set
according to its impact.

```bash
# One module
uv run pytest phyai/tests/weights/test_loader.py

# One test
uv run pytest phyai/tests/weights/test_loader.py::test_name

# Full workspace
uv run pytest
```

The root test bootstrap stops collection when CUDA is unavailable. Before
running CUDA tests or benchmarks, check device availability:

```bash
nvidia-smi
```

Use this table as a minimum validation guide:

| Change | Minimum validation |
| --- | --- |
| Shared Python layer or runtime code | Targeted tests plus tests for direct consumers |
| Kernel implementation or dispatch | Correctness tests, selector coverage, CUDA Graph coverage when applicable, and a benchmark for performance claims |
| C++ extension | Rebuild with `uv sync` and run the relevant `phyai-ext/tests` modules |
| Model implementation | Layer tests in the repository and full-checkpoint parity outside the committed test suite |
| Documentation | Check commands and links; run the Mintlify link checker for `docs/` changes |

If your environment cannot run a required check, state which command was not
run and why. Do not describe an unrun test as passing.

Run all formatting and spelling hooks before opening a pull request:

```bash
scripts/run_pre_commit.sh
```

Some hooks rewrite files. Inspect the resulting diff and run the command again
until it passes.

## Documentation

Update documentation and examples when behavior, configuration, or public APIs
change. The documentation site uses Mintlify:

```bash
cd docs
mint dev
mint broken-links
```

Use sentence-case headings, language tags on code blocks, relative links for
repository content, and tested commands. Keep examples runnable and avoid
embedding local checkpoint paths.

## Commits and pull requests

Commit subjects and pull request titles should follow Conventional Commits when
possible:

```text
feat(phyai): add model support
fix(phyai-kernel): correct rms norm dispatch
docs: clarify source installation
```

Use a concise scope that matches the package or subsystem. Keep generated files
and unrelated formatting out of the commit.

A pull request description should include:

- The problem and the reason for the chosen approach.
- User-visible, API, precision, and performance effects.
- The exact validation commands and their results.
- Relevant issues, reference implementations, or design discussions.
- Logs, metrics, or screenshots when they help reviewers verify the result.

Call out compatibility risks and intentional numerical differences. For kernel
or model changes, include enough dtype, shape, and backend information for
another contributor to reproduce the comparison.

Before requesting review, update your branch from `main`, review the full diff,
and confirm that no local paths, credentials, checkpoints, or machine-specific
output are present.

## Reporting bugs

A useful bug report contains:

- A small reproduction and the command that runs it.
- Expected and actual behavior.
- The full traceback or relevant log lines.
- The commit or package version.
- Python, Torch, CUDA, and dependency versions.
- The output of `uv run phyai-kernel show-env` when kernel selection is involved.

Remove credentials, private paths, and unrelated environment data before
posting logs. If the problem appears only after local modifications, reproduce
it on an unmodified checkout when possible.

## License

Contributions are made under the repository's [MIT License](LICENSE). Community
participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
