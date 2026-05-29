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

Everything goes through `uv`. The lockfile (`uv.lock`) is committed.

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

---

!!! Claude code or other coding agent should update the sections below when some modules/files are modified !!!

!!! Claude code or other coding agent should update the sections below when some modules/files are modified !!!

!!! Claude code or other coding agent should update the sections below when some modules/files are modified !!!

!!! Things below is auto generated by claude, it maybe wrong !!!

!!! Things below is auto generated by claude, it maybe wrong !!!

!!! Things below is auto generated by claude, it maybe wrong !!!

## Architecture: the engine + plugin contract

The runtime entry point is `phyai.engine.Engine`. Three concepts to know:

- **`EntryArgs`** (empty base dataclass) — each model declares a typed subclass (`PI05Args(weights_paths=..., batch_size=..., ...)`). No argparse, no env-var parsing — args are programmatic.
- **`Entry` (ABC)** — every plugin implements `setup(args)` / `step(request)` / `close()`, with class attributes `name: ClassVar[str]` and `args_cls: ClassVar[type[EntryArgs]]`.
- **`Engine.__init__`** runs a **fixed orchestrated sequence**, each step independently callable so tests / advanced users can opt in:
  1. `init_engine_config(...)` — install process-level `EngineConfig` singleton.
  2. `init_cuda(target, params_dtype)` — pin device and `torch.set_default_dtype`.
  3. `init_cublas()`.
  4. `init_dist(tp_size=..., device_type=...)` — bring up `torch.distributed` (no-op at `tp_size=1`).
  5. `phyai.parallel.init(layout, mesh_dim_names, ...)` — build `Mesh` + collective dispatcher.
  6. `phyai.layers.linear.init()` — materialise the linear `KernelDispatcher` from registered backends.
  7. Resolve plugin class, validate args, instantiate, call `setup`.

**Plugin discovery is explicit**: each `phyai/src/phyai/models/<model>/main_<model>.py` decorates its Entry class with `@Engine.register`. Then `phyai/src/phyai/engine.py` imports the plugin module **at the bottom of the file** to trigger registration. Adding a new model = one import line at the bottom of `engine.py` plus a `main_<model>.py`.

`EngineConfig` is a frozen dataclass with four sub-configs:
- `BackendConfig(attn, norm, linear, vgpu)` — string-name backend selection. Each name is validated against the corresponding registry in `__post_init__`, so a typo fails at config construction, not deep in tensor allocation.
- `DeviceConfig(target, params_dtype)`.
- `ParallelConfig(world_size, dp_size, ep_size, sp_size, cp_size, tp_size)` — every field defaults to 1. `world_size` is an explicit user input (set it to whatever the launcher's `WORLD_SIZE` is), **not** a product of the per-axis sizes — real strategies overlap (sglang's `tp_size % attn_cp_size == 0`, `ep_size * moe_dp_size <= tp_size`, etc.), so the product is wrong in the general case. The engine builds a 5-axis mesh `("dp", "ep", "sp", "cp", "tp")` so model code addressing `axis="tp"` keeps working at any topology.
- `RuntimeConfig(use_cuda_graph, flashinfer_workspace_bytes, force_linear_kernel)` — runtime tunables. `flashinfer_workspace_bytes` (default 128 MiB) is consulted by `phyai.layers.attention.utils.resolve_workspace_bytes`; `force_linear_kernel` is consulted by `phyai.layers.linear.dispatch.KernelDispatcher`. Layers read these from the `EngineConfig` singleton, not from envs directly — env-var overlays happen once in `from_env`.

Construct via `EngineConfig.auto()` (host-derived defaults), `EngineConfig.from_env()` (auto + `PHYAI_*` overlays), or kwargs. Singleton accessors live in `phyai.engine_config`: `get_engine_config()` / `set_engine_config(cfg)` / `init_engine_config(cfg)`. Mutating after model construction does **not** retroactively fix already-allocated tensors — install the config first, then build models.

## Backend-registry pattern (recurring)

Every pluggable subsystem follows the same shape: a registry module with `register_*` decorators, concrete backend modules under `<pkg>/backends/` that decorate themselves at import time, a `<pkg>/backends/__init__.py` that imports each module for its side effect, and a top-level `init()` that materialises a dispatcher from the gathered registrations. Live instances:

- `phyai.layers.attention.registry` — `register_no_cache_backend("name")` / `register_cached_backend("name")` (sdpa / eager / flashinfer / flashinfer-paged). `BackendConfig.attn` selects the no-cache backend; `cached_attn_backend_for(...)` derives the cached backend from the same choice.
- `phyai.layers.linear.registry` — `register_linear_kernel(prefer_for={(spec_id, regime)})` declares lazily; `phyai.layers.linear.init()` walks the declarations.
- `phyai.layers.layer_norm` — flashinfer / phyai-kernel (Triton) norms; `list_norm_backends()` exposed.
- `phyai.parallel.registry` — `Registry.register(backend, prefer_for={Op,...})`; `phyai.parallel.init()` builds it from the available backends (NcclBackend / GlooBackend / PyNCCLBackend).
- `phyai.vgpu.backend` — flashinfer (default if importable) and torch always registered.

When adding a backend: subclass the right ABC/Protocol (`NoCacheAttentionBackend`, `LinearKernel`, `Backend`, `GreenCtxBackend`, `LayerNormBackend`, …), decorate it, add the side-effect import to the package's `backends/__init__.py`. **Do not** instantiate at module import — the registry stores classes; instantiation happens at `init()` time so importing a backend is CUDA/driver-free.

## Subsystems worth knowing about

- **`phyai.runtime`** — `CudaGraph` (one captured graph + static input buffers, copy-and-replay) and `CudaGraphRegistry` (key->graph map for shape buckets). `ModelRunner` ABC and `Scheduler` ABC. `CudaIpcBuffer` (GPU buffer, cudaMalloc) and `HostShmBuffer` (POSIX shm, optional `cudaHostRegister` for pinned/portable host pages) — symmetric inter-process transport substrates with `create()` / `attach()` classmethods and serializable `CudaIpcHandle` / `HostShmHandle` dataclasses. The application ships handles between unrelated OS processes (e.g. preprocessor → model worker) over any transport (zmq `send_pyobj`, `subprocess.Popen` argv, pipe, file). Both creator and attacher have full read/write access; only `close()` semantics differ (creator: `cudaFree` / `unlink`; attacher: `cudaIpcCloseMemHandle` / `shm.close`). Tensor views from `tensor()` keep a strong ref to the buffer via `__cuda_array_interface__` keepalive — `close()` is the explicit invalidation point. No `torch.distributed` dependency. Cross-device cudaIpc attach (producer cuda:0, attacher cuda:1) supported via the `cudaIpcMemLazyEnablePeerAccess` flag. Concrete runners and schedulers live alongside the model under `phyai.models.<model>/` (e.g. `model_runner_pi05.py`, `scheduler_ws1_pi05.py`).
- **`phyai.payload`** — `VisionForwardBatch`, `LLMForwardBatch`, `ExpertForwardBatch`. The scheduler builds these and hands them to runners; runners return tensors that get stuffed into the next stage's payload.
- **`phyai.cache`** — `KVCachePool` is a passive `(num_slots, page_size, num_kv_heads, head_dim)` K/V container in flashinfer's paged layout. Allocation policy lives in `StaticCache` (one-shot, no eviction) or the `radix_cache` facades (`MultimodalCache`, `PairedCache`, `HybridCache`, `MultiPatternCache`, `NestedCache`) which front the C++ implementation in `phyai-ext/csrc/radix_cache/`.
- **`phyai.parallel`** — collectives are addressed by axis name: `P.all_reduce(x, axis="tp")`. `init(layout=(8,), mesh_dim_names=("tp",))` builds the mesh; at `world_size=1` it short-circuits with a degenerate mesh (no `init_process_group` needed). PyNCCL is registered preferentially over `NcclBackend` for ops that benefit under graph capture. Multi-rank tests use `phyai/tests/parallel/multiprocess.py::run_distributed` to spawn N NCCL workers.
- **`phyai.vgpu`** — single-card SM partitioning via CUDA Green Context. Default backend is **flashinfer**, deliberately, because torch's multi-create does not return disjoint partitions on torch 2.11. vGPUs leak ~16 MiB per `split_device_green_ctx` call at the driver level; long-lived setup, never hot-path create.
- **`phyai.weights`** — param-attached loaders. Layers populate `param.hf_keys: list[(hf_key, shard_id)]` and `param.weight_loader: WeightLoader` during `__init__`; `load_pretrained(model, paths, remap=..., strict=...)` walks safetensors and dispatches per-param. Optional `param.optional` skips the missing-source raise (used for quant scales). Factories live in `phyai.weights.shards`: `replicated`, `sharded`, `fused`, `vocab`.
- **`phyai.layers.linear`** — `ReplicatedLinear` / `ColumnParallelLinear` / `RowParallelLinear` / `MergedColumnParallelLinear` / `QKVParallelLinear` with declarative `WeightSpec` (`Bf16Spec`, `Fp8Spec(granularity=Granularity.PER_CHANNEL | …)`).
- **`phyai.layers.attention`** — `NoCacheAttention` (prefill-only, no cache) + `StaticCachedAttention` (varlen + paged KV, designed to sit inside a captured CUDA graph — the runner plans the wrapper outside capture and replay reads pre-allocated metadata buffers updated via `.copy_()`).
- **`phyai-kernel`** — Triton kernels in `phyai_kernel/triton/` are the primary path; `jit_utils/` JIT-compiles CPU/CUDA C++ via tvm-ffi when needed.

## Env vars

Every `PHYAI_*` env var is declared once in `phyai/src/phyai/env.py::envs` with a typed parser — don't sprinkle `os.environ.get` in random modules. Currently registered:

- Backends: `PHYAI_ATTN_BACKEND`, `PHYAI_NORM_BACKEND`, `PHYAI_LINEAR_BACKEND`, `PHYAI_VGPU_BACKEND`.
- Device/dtype: `PHYAI_DEVICE`, `PHYAI_PARAMS_DTYPE` (accepts `bf16`/`fp16`/`fp32`/...).
- Runtime: `PHYAI_USE_CUDA_GRAPH`, `PHYAI_WORLD_SIZE`, `PHYAI_TP_SIZE`, `PHYAI_DP_SIZE`, `PHYAI_EP_SIZE`, `PHYAI_SP_SIZE`, `PHYAI_CP_SIZE`.
- Tuning: `PHYAI_FLASHINFER_WORKSPACE_BYTES`, `PHYAI_FORCE_LINEAR_KERNEL`.
