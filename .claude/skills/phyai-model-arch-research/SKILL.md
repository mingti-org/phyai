---
name: phyai-model-arch-research
description: Use this skill when the user provides a paper, arXiv link, technical report, model card, checkpoint name, GitHub repository, or local codebase and asks to research, explain, compare, or implement a model architecture. This skill guides architecture research for PHYAI: source collection, paper/code tracing, module decomposition, tensor shape analysis, dependency mapping, and implementation-oriented reporting.
---

# PHYAI Model Architecture Research

Use this skill to study a model architecture from a paper, repository, or both, then produce an implementation-oriented architecture report. Optimize for facts that help PHYAI model support, kernel work, quantization, profiling, and deployment.

## Research Goals

Answer these questions before proposing implementation work:

- What model family is this, and what prior architectures does it inherit from?
- What are the major modules, data flow, tensor shapes, and configurable hyperparameters?
- Which parts are standard, and which parts are novel or likely to need custom code?
- Where is the authoritative implementation: paper equations, official repo, model config, or checkpoint metadata?
- What are the likely integration risks for PHYAI: custom ops, attention variants, positional encoding, MoE routing, normalization, multimodal preprocessing, cache layout, dtype, quantization, and generation behavior?

## Source Priority

Prefer sources in this order:

1. Official code repository, release branch, or tagged commit.
2. Paper or technical report, especially architecture figures, tables, appendices, and ablations.
3. Model card, config files, tokenizer/preprocessor files, checkpoint metadata.
4. Reputable secondary explanations only when primary sources are missing or unclear.

When sources disagree, state the conflict and prefer executable/configured behavior over prose unless the user asks for a paper-only summary.

## Workflow

### 1. Clarify the Target

Identify:

- model name and version;
- paper URL, repo URL, local repo path, checkpoint path, or package name;
- desired output depth: quick summary, implementation plan, code mapping, comparison, or full report.

If the user's paper, article, link, repository, checkpoint, or model name is missing, inaccessible, ambiguous, or could refer to multiple targets, ask the user a concise follow-up question before doing architecture research. Do not guess the target architecture when the source is unclear.

If the user gives a recognizable but incomplete name, first state the likely interpretation and ask for confirmation or a source link. Only search for official sources after the target is clear enough to avoid researching the wrong model.

### 2. Collect Architecture Evidence

For papers, inspect:

- abstract and introduction for the claimed contribution;
- architecture section, equations, diagrams, and algorithm blocks;
- model size/config tables;
- training/inference details that affect architecture behavior;
- appendix for hidden implementation details.

For code repositories, inspect:

- `README`, model cards, docs, and config examples;
- model definition files such as `modeling_*.py`, `configuration_*.py`, `model.py`, `modules.py`, `layers.py`;
- preprocessing/tokenizer/processor code for multimodal models;
- generation, cache, sampling, and inference utilities;
- custom kernels, CUDA/Triton ops, fused layers, or extension bindings;
- tests and examples that show expected shapes and behavior.

Use `rg` first for local code search. Useful patterns:

```text
class .*Model|class .*For|forward\(|attention|rotary|rope|norm|mlp|moe|expert|cache|kv|vision|encoder|decoder|embed|projector
```

### 3. Build the Architecture Map

Document the model in layers:

- input pipeline: tokenizer, image/audio/video preprocessing, patching, feature extraction;
- embedding path: token/patch embeddings, special tokens, position encoding;
- backbone blocks: attention, MLP, normalization, residual layout, routing, recurrence/state-space components;
- cross-modal or adapter components: projector, connector, resampler, cross-attention;
- output heads: LM head, classification head, diffusion head, value head, decoder head;
- inference state: KV cache, recurrent state, streaming buffers, masks, packed sequences;
- configuration knobs: depth, hidden size, heads, head dim, intermediate size, experts, vocab, context length, precision.

Track tensor shapes at module boundaries when possible. Use symbolic names like:

```text
B = batch size
T = sequence length
H = hidden size
N = number of heads
D = head dim
V = vocab size
```

### 4. Compare Against Known Patterns

Call out whether the model resembles:

- LLaMA/Qwen/Mistral/GPT-style decoder-only transformers;
- encoder-decoder transformers;
- ViT/SigLIP/CLIP-style vision encoders;
- multimodal LLMs such as LLaVA/Qwen-VL/InternVL-style projector stacks;
- MoE models with router/top-k experts;
- diffusion/DiT/U-Net architectures;
- state-space or hybrid attention models;
- custom research architecture not covered by common model families.

The comparison should explain implementation consequences, not just naming similarity.

### 5. Produce a Report

Use this structure unless the user asks for a different format:

```markdown
## Executive Summary
- model family:
- core idea:
- implementation difficulty:
- main integration risks:

## Sources Checked
- paper:
- repo/code:
- configs/checkpoints:

## Architecture
Describe the full data flow from input to output.

## Module Breakdown
| Module | Purpose | Key config | Shape notes | Code location |
| --- | --- | --- | --- | --- |

## Novel or Nonstandard Parts
List components that may require custom implementation, kernels, or careful validation.

## PHYAI Integration Notes
Discuss model loading, config mapping, tokenizer/processor, kernels, cache, quantization, tests, and performance risks.

## Open Questions
List missing facts, ambiguous source conflicts, or items needing a run/checkpoint.
```

## Code Mapping Rules

When analyzing a repository:

- link architecture claims to exact file paths and functions/classes;
- distinguish public API wrappers from the actual implementation;
- check config defaults instead of assuming paper hyperparameters;
- trace `forward()` through helper modules until the real tensor operations are clear;
- inspect tests/examples to confirm expected behavior;
- avoid large refactors or code changes unless the user explicitly asks for implementation.

## Paper Reading Rules

When analyzing a paper:

- do not summarize every section equally; focus on architecture and implementation facts;
- extract equations only when they define behavior needed for implementation;
- note missing details that must be recovered from code or configs;
- separate the authors' claims from confirmed architecture mechanics.

## Validation Checklist

Before finalizing, check whether the report covers:

- source authority and version;
- full input-output path;
- module list and config knobs;
- tensor shape assumptions;
- attention/cache behavior;
- normalization, activation, positional encoding;
- custom ops/kernels or dependencies;
- checkpoint/config compatibility;
- minimal tests needed for PHYAI support.

If a fact is inferred rather than directly sourced, label it as an inference.
