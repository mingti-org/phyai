# Enactive pi0.5 checkpoint

`run_enactive.py` runs a converted Enactive pi0.5 export through PhyAI's existing
`pi05` engine and `PI05Processor`. The checkpoint's serialized processor
pipelines select an instruction-only prompt, PIL resizing and state-anchored
action decoding. There is no separate Enactive processor, runtime dependency or
model change. The robot deployment in `examples/deployment` uses the same
processing on its model server (`configs/pi05_agilex.yaml`).

## Checkpoint contents

The converted directory contains the BF16 inference weights, `config.json`, a
fast `tokenizer.json` exported from the original SentencePiece vocabulary,
`policy_preprocessor.json` / `policy_postprocessor.json` with their
normalization sidecars, and `conversion_report.json` with source/output SHA-256
values and tensor mappings. All 811 inference tensors load with no missing or
unexpected keys; eight training-only tensors are omitted.

| Setting | Value |
| --- | --- |
| Vision / text / expert widths | 1152 / 2048 / 1024 |
| Vision / text / expert layers | 27 / 18 / 18 |
| Precision | BF16 parameters and compute |
| Actions | 50 steps, 32 model dimensions, 14 physical dimensions |
| Sampling | Gaussian initialization, 10 Euler steps |

## Input and output contract

Provide RGB uint8 HWC images named `front`, `left` and `right` (front, left
wrist, right wrist), an instruction string, and the current raw 14-dimensional
joint state. Images are resized with PIL bilinear filtering, aspect ratio
preserved and black padding, to 224 by 224, then scaled to [-1, 1].

Text follows the training transform: strip surrounding whitespace, replace
underscores and newlines with spaces, tokenize with BOS, append a separately
tokenized newline, truncate to 200 tokens and pad with token 0. The prompt
contains no discretized state.

The engine returns `[B, 50, 32]` normalized actions. The processor keeps the
first 14 dimensions and applies the checkpoint quantiles:

```text
physical = (normalized + 1) / 2 * (q99 - q01 + 1e-6) + q01
absolute_joint_target = physical + current_raw_state
```

The state is added to joints 0-5 and 7-12 only; grippers 6 and 13 are absolute.
All 50 steps use the same observation anchor and are not clipped. Always pass
the raw state of the observation that produced the request.

## Run

```bash
uv run --no-sync python examples/pi05/run_enactive.py \
  --checkpoint /path/to/enactive-pi05-checkpoint \
  --input observation.npz \
  --output actions.npz
```

`observation.npz` contains `front`, `left`, `right`, `state` and a scalar string
`task`. An optional float32 `noise` array of shape `[1, 50, 32]` supplies the
initial noise; otherwise `--seed` (default 42) generates CPU noise. The output
holds `actions` `[1, 50, 14]` and `normalized_actions` `[1, 50, 32]`. CUDA
Graph is on by default; `--no-cuda-graph` disables it for this one-shot run.

## Validation summary

Twelve synthetic observations (random RGB of several aspect ratios, English and
Chinese instructions, prompts longer than 200 tokens, states within the saved
quantiles) were run through PhyAI and the Enactive reference implementation
with identical checkpoint and initial noise.

| Check | Result |
| --- | --- |
| Pixel values, language tokens/lengths, normalized states | Exact agreement |
| Postprocessing of common normalized actions | Exact agreement |
| Tokenizer (fastokens and HF Rust) against SentencePiece, 1,931 texts | Exact agreement |
| First-velocity cosine, worst of 12 | 0.999897385 |
| Valid 14D final-action cosine, worst of 12 | 0.998697594 |
| Decoded action RMSE, worst sample | 0.003382853 |
| Decoded maximum absolute error | 0.014956951 |
| CUDA Graph versus eager; repeated identical requests | Bitwise equal |
| Batched versus single-request action cosine | At least 0.999992080 |

Model outputs are not bitwise identical. The implementations differ in GELU
(PhyAI's pi0.5 uses exact GELU; the Enactive reference used its tanh
approximation), BF16 intermediate rounding, attention kernels and action-head
precision. Bitwise agreement would require model changes beyond this adapter.
Robot task success was not measured by these tests.
