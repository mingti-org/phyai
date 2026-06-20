---
name: phyai-model-implement
description: >-
  Use this skill when implementing, porting, integrating, reproducing, or debugging support
  for a model in PHYAI. This includes translating architecture research into PHYAI code,
  adding model configuration, modeling modules, runners, schedulers, weight loading,
  layer reuse, focused tests, validation scripts, and implementation plans while respecting
  PHYAI model-development constraints.
---

# PHYAI Model Implementation Guide

## Before You Implement

First, carefully study the existing repositories and articles. If the user has not provided
references, ask them for the relevant references before proceeding.

## Core Three-Layer Architecture

PHYAI uses a strict three-layer separation for every model:

```text
modeling_xxx.py     - Pure architecture definition: stateless, no KV cache, no runtime state
model_runner_xxx.py - Runtime state management: KV cache pool, condition cache, prefill/decode switching
scheduler_xxx.py    - Orchestration loop: denoise loop, sampler, CFG, multi-step orchestration,
                      multi-GPU and multi-stage coordination
vae.py              - Extra models or processing files, such as VAE, and auxiliary models
```

**Never:**
- **Never** put KV cache inside modeling classes. A modeling class performs one forward pass only.
- **Never** introduce external modeling files from libraries such as diffusers, transformers, or wan_x.
- **Never** put sampling or noise-schedule logic in the modeling layer. That is the scheduler's responsibility.
- **Never** put matrix multiplication or attention in the scheduler layer unless there is no viable alternative.
  Those belong in the modeling layer.

**Call relationship between layers:**

```text
Engine Plugin (main_xxx.py)
  +-- Scheduler (scheduler_ws1_xxx.py)
        +-- ModelRunner (model_runner_xxx.py)
              +-- Model (modeling_xxx.py)
```

Each layer calls only the layer immediately below it. Do not skip layers. A scheduler must not call
an `nn.Module.forward` directly; it must go through the runner.

## File Naming Conventions

```text
phyai/src/phyai/models/<model_name>/
+-- __init__.py                      # Export all public symbols
+-- configuration_<model>.py         # Frozen dataclass configuration
+-- modeling_<model>.py              # Pure network architecture
+-- model_runner_<model>.py          # Runtime state wrapper
+-- scheduler_ws1_<model>.py         # Single-GPU scheduler; ws = world_size
+-- main_<model>.py                  # Engine plugin entry point; Entry subclass
+-- (optional) sampler_*.py          # Diffusion or ODE sampler
     NOTE: name it XxxSampler, not XxxScheduler,
           to avoid conflict with phyai.runtime.schedule.Scheduler.
```

**Naming rules:**
- Use CamelCase for class names and snake_case for file names.
- Weight remap function: `<model>_weight_remap`.
- Engine plugin name: a short lowercase name, such as `"pi05"` or `"cosmos3_policy"`.

## Implementation Steps

Follow these steps in order.

### 1. Configuration (`configuration_<model>.py`)

- Define the model configuration with `@dataclass(frozen=True)`.
- Support loading from checkpoint `config.json` with `load_config(path, XxxConfig)`.
- If checkpoint JSON keys do not match dataclass fields, use the `nested_sources` class variable
  to define the mapping.
- Validate the configuration in `__post_init__`, such as GQA divisibility and even `head_dim`.
- Give every numeric parameter a reasonable default so `XxxConfig()` can be constructed without
  arguments for CPU tests.

### 2. Modeling (`modeling_<model>.py`)

**Principles:**
- Keep modeling stateless. Each forward pass depends only on its input arguments, not on internal
  mutable state.
- Use shared layers from `phyai.layers`, such as `Linear`, `RMSNorm`, `Attention`, and
  `RotaryEmbedding`.
- Do not implement a custom attention kernel. Use `phyai.layers.attention`.
- Do not implement RoPE yourself. Use `phyai.layers.rotary_embedding`.
- Do not implement normalization yourself. Use `RMSNorm` or `LayerNorm` from `phyai.layers`.
- The weight mapping function `xxx_weight_remap(name)` returns the checkpoint key to PHYAI key
  mapping.
- If a new general-purpose layer is needed, add it to `phyai.layers` rather than to the model
  directory.

**Typical forward signature:**

```python
class XxxModel(nn.Module):
    def __init__(self, config: XxxConfig, *, params_dtype=torch.bfloat16, device="cpu"):
        ...

    def forward(self, inputs, ...) -> output:
        """Run one forward pass without storing intermediate state."""
```

### 3. Weight Loading

- Use `phyai.weights.loader.load_pretrained(module, path, remap=xxx_weight_remap, strict=False)`.
- Define the remap function as `def xxx_weight_remap(name: str) -> str | None`; returning `None`
  means the weight is skipped.
- Support both single-file safetensors and sharded indexes; automatic detection is already
  available.
- Validate with `assert len(report.missing) == 0`.

### 4. Model Runner (`model_runner_<model>.py`)

- Inherit from `phyai.runtime.model_runner.ModelRunner`, whose abstract methods are `setup()` and
  `forward()`.
- Responsibilities:
  - Condition caching: run `encode_condition` once and reuse the result in later steps.
  - CUDA graph capture, optionally controlled by a `use_cuda_graph` flag.
  - Optional `torch.compile`, applied to submodules in `setup()`.
  - `reset()`, which clears per-request caches and is called at the start of each new request.
- The runner holds a reference to the model but does **not** own the weights. Weights are loaded at
  the plugin layer and then passed in.

### 5. Scheduler (`scheduler_ws1_<model>.py`)

- Inherit from `phyai.runtime.schedule.Scheduler`, whose abstract methods are `setup()` and
  `step(request)`.
- Responsibilities:
  - Inference loop, such as repeatedly calling `runner.forward` for a diffusion denoise loop or an
    autoregressive decode loop.
  - Multi-branch forward passes, such as CFG dual branches plus guidance interpolation.
  - Sampler control, including timestep schedules and ODE solvers.
  - Condition re-imposition, such as writing conditioned regions back at every step.
  - Noise initialization: seed to generator to `randn`.
- Define requests with `@dataclass`; include all inference parameters in the request.
- `step()` should behave functionally: given a request, return a result, and do not keep state
  across requests.

### 6. Engine Plugin (`main_<model>.py`)

- Register the plugin with the `@Engine.register` decorator.
- Implement `setup(args)`, `step(request)`, and `close()`.
- `setup` is responsible for loading the configuration, building the model, loading weights,
  building the runner and scheduler, and warming up.
- `step` delegates directly to the scheduler.
- `close` releases all GPU resources.

### 7. Processor (`phyai-utils-tools` package)

- Inherit from `BaseModelProcessor` and implement `build_preprocessor()` and
  `build_postprocessor()`.
- Preprocessing: raw image/text/state to model-ready tensors.
- Postprocessing: model output to a user-friendly format.
- **Do not import `phyai`**. `phyai-utils-tools` is an independent leaf package and must not depend
  on the main library.
- If preprocessing requires a model forward, such as VAE encode, keep it in the scheduler or plugin
  layer rather than in the processor.

## Critical Constraints

### Code Organization

1. **Do not modify general components** such as `phyai.layers` or `phyai.runtime` unless a required
   layer or system component is missing. If a change is needed, tell the user first.
2. **Use one directory per model**. Put all model-specific code under `phyai/models/<model>/`; do not
   scatter it elsewhere.
3. **Do not import across models**. Model A must not import code from Model B. Promote shared logic
   to `phyai.layers`.

### Attention and Computation

4. **Use flashinfer or SDPA for attention by default**. Do not use eager attention except for debug.
5. **Precision:** default to bf16. If a submodule needs fp32, such as a timestep MLP or ViT, control
   it through `params_dtype` during construction.
6. **Do not hardcode devices in model code**. Pass the device as an argument, or infer it from
   existing tensors.

### Testing

7. **Layer-level tests** belong in `phyai/tests/`; they can run in CI in CPU mode.
8. **Model-level tests** that require full weights or GPU belong in `.cache/`, because CI does not
   have enough resources for them.
9. `conftest.py` automatically overrides `device.target = "cpu"`. CUDA tests must opt in explicitly.

### Style

10. **Logging:** use `this_rank_log` or `all_rank_log` from `phyai.utils`; do not call `print`
    directly.
11. **Comments:** write all comments in English.
12. **Naming:** public by default. Do not add leading underscores casually. Expose singletons through
    `get_*()` getters.
13. **Do not add `type: ignore`**. Fix the type instead of suppressing the warning.
14. **Import order:** stdlib, third-party, phyai, then local imports. Ruff will sort imports
    automatically.

## Validation Workflow

Validation must compare PHYAI against the original reference implementation, not only against
shape checks or smoke tests. Keep the reference repository available locally when possible, run the
same checkpoint and deterministic inputs through both implementations, and save enough intermediate
tensors to identify the first layer that diverges.

Before judging final quality, align the basics with the reference repository:

1. Use the same checkpoint, tokenizer or processor, preprocessing rules, dtype policy, random seed,
   timestep schedule, sampler settings, guidance settings, and device placement.
2. Verify config parity: every architecture field that affects tensor shapes, attention layout,
   normalization, MLP width, RoPE, patching, channel order, or action/video dimensions must match
   the reference.
3. Verify weight parity: map each checkpoint tensor to the intended PHYAI parameter, check missing
   and unexpected keys, and spot-check representative tensor values after loading.
4. Verify input parity: feed identical model-ready tensors to both implementations. If processors
   differ, dump the processed tensors and compare them before running the model.
5. Compare progressively: embedding/projection outputs, each block or major submodule, final model
   outputs, and finally the scheduler or end-to-end result.

### Single-Step Velocity Parity

```python
# Same input: PHYAI forward vs. reference forward.
# Expected cosine > 0.99, allowing for bf16 accumulation error.
cosine = F.cosine_similarity(phyai_out.flatten(), ref_out.flatten(), dim=0)
```

For a diffusion or flow model, compare the predicted velocity/noise/action for one fixed step
against the reference repository first. The final single-step output cosine similarity must be
greater than `0.99`. If it is below `0.99`, do not treat the port as validated; find the earliest
diverging intermediate tensor and fix the corresponding config, weight mapping, layout, precision,
or preprocessing issue.

### End-to-End Inference Validation

```python
# Determinism: same seed -> exactly identical output, cosine = 1.0.
# Convergence: after the denoise loop, output std should be far below noise std.
```

After single-step parity passes, run the full PHYAI scheduler against the reference repository with
the same request and deterministic seed. The final end-to-end result should also reach cosine
similarity greater than `0.99` against the reference output, unless the reference uses a documented
non-deterministic kernel. If it cannot meet this threshold, document the exact source of drift and
whether it comes from precision, sampler implementation, preprocessing, or an intentional algorithmic
deviation.

### Weight Loading Validation

```python
report = load_pretrained(model, path, remap=remap, strict=False)
assert len(report.missing) == 0  # All weights have been loaded.
```

## References

Existing model implementations live under `phyai/src/phyai/models/`. Before implementing a new
model, read one complete existing model implementation as a reference.

## Basic Performance Considerations

Write code that remains friendly to later performance optimization, including but not limited to
`torch.compile` and CUDA graphs.

### `torch.compile` Friendly

1. **Avoid data-dependent control flow.** Do not branch on runtime tensor values, such as
   `if tensor.item() > 0`, because it causes graph breaks unless the algorithm truly requires it.
2. **Avoid dynamic shapes.** Keep tensor shapes inferable at compile time whenever possible. If a
   shape must be dynamic, mark it with `torch._dynamo.mark_dynamic()`.
3. **Do not create a tensor in `forward` and immediately call `.to(device)`.** Preallocate buffers in
   `__init__` or `setup()`.
4. **Avoid mixing Python list/dict operations with tensors.** `[t1, t2, t3]` followed by
   `torch.stack()` is fine, but appending tensors to a list in a loop and then calling `cat` may
   trigger a graph break.
5. **Prefer `torch.nn.functional` over custom Python loops.** For example, use
   `F.scaled_dot_product_attention` rather than a hand-written softmax plus matmul.

### CUDA Graph Friendly

1. **Use fixed tensor shapes.** All input and output shapes must be fixed during CUDA graph capture.
   Shape changes require re-capture.
2. **Do not cause CPU-GPU synchronization in `forward`.** `tensor.item()`, `tensor.cpu()`, and
   `print(tensor)` all break graph execution.
3. **Do not allocate memory in `forward`.** Preallocate `torch.empty`, `torch.zeros`, and
   `torch.randn` buffers, then fill them in-place in `forward`.
4. **Avoid Python side effects.** CUDA graph replay does not re-run Python code; it replays only the
   captured CUDA kernel sequence.
5. **Use static flags for conditional branches instead of runtime tensors.** `if self.use_xxx:`
   with a Python bool is acceptable; `if tensor > 0:` is not.

### General Principles

1. **Reduce kernel launches.** Fuse consecutive small operations where possible, such as RMSNorm plus
   Linear.
2. **Avoid unnecessary `contiguous()` calls.** Call it only when a kernel truly requires contiguous
   memory.
3. **Do not write attention yourself.** Use the unified interface in `phyai.layers.attention`; it
   chooses flashinfer, flash_attn, or SDPA automatically.
4. **Prefer reshape/view over permute plus contiguous for large tensors.** The latter triggers an
   extra memory copy.
5. **Do not call `torch.cuda.synchronize()` in `forward`.** It serializes all streams unless needed
   for debugging.
