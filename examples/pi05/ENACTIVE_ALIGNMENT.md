# Enactive pi0.5 checkpoint adapter and validation

The adapter runs the selected `inference-500` export through Phyai's existing
`pi05` engine and `PI05Processor`. Checkpoint sidecars configure the existing
pipeline with an instruction-only prompt, PIL resizing and state-anchored action
decoding. There is no separate Enactive processor or new runtime dependency.
The model, scheduler, layers and kernels are unchanged. The conversion scripts
are retained locally as untracked files.

Validation on 2026-09-23 passed the numerical comparison gates: cosine similarity
above 0.99 for the first flow velocity and the final actions. Preprocessing,
postprocessing and converted tensor bytes match their references exactly. BF16
model outputs are approximate: the worst normalized action difference across the
single and batched runs was 0.123322, and the worst decoded action difference was
0.014957 in the checkpoint's native units. These tests do not establish robot task
success or bitwise equivalence between engines.

## Checkpoint and reference

The source run is `pi05_cube_pnp_merged_v3_h264_h50_g4_e5_20260818_v5`, step 500.
The selected artifact is the complete `inference-500` export. Training shards from
`checkpoint-1000` were not used. The converted artifact is stored locally at
`.cache/checkpoints/enactive-pi05-step500`; weights and test artifacts are ignored
by Git.

The reference is the Enactive `gzq/cobot-pi05` branch snapshot at
`5c4e8b51f08d6b26caf8e4518e3676f10debe181`. The training commit was not recorded
in the supplied metadata, so this SHA identifies the code used for comparison,
not a proven training revision.

The later inference export changed `model_type` from `pi` to `layerwise` and added
`mlp_activation_implementation=fused_tanh`. The historical branch uses its own
BF16 tanh GELU implementation. Reference loading therefore uses the original
step-500 configuration backup
`config.json.bak-before-model-type-20260922T080457`, with a copied manifest whose
family is `pi`. Its weights come from the selected inference export. All source
files remain unchanged. Both configurations and their complete difference list
are preserved in the converted directory and `conversion_report.json`.

The resolved training configurations establish:

| Setting | Value |
| --- | --- |
| Model config | `enactive_pi__paligemma_native_und_state_pi05_h50_sdpa.yaml` |
| Data config | `joint_position__cube_pnp_merged_v3_h264__h50.yaml` |
| Runtime config | `vla__pi05_joint_position.yaml` |
| Vision / text / expert widths | 1152 / 2048 / 1024 |
| Vision / text / expert layers | 27 / 18 / 18 |
| State mode / attention | `und` / SDPA mixture of transformers |
| Parameter / compute precision | BF16 / BF16 |
| Actions | 50 steps, 32 model dimensions, 14 physical dimensions |
| Sampling | Gaussian initialization, 10 Euler steps |
| RTC / classifier-free guidance | Disabled |
| Training | 4 FSDP processes, batch 16 each, global batch 64, 5 epochs |
| Optimizer | AdamW, learning rate 5e-5, betas 0.9/0.95, weight decay 1e-10 |
| EMA configuration | Decay 0.999; converter uses the selected export as supplied |

The compatible training entry point is
`scripts/entrypoints/train/train_enactive.py`, which wires the model through
`enactive/trainer/model_wiring.py`. The policy is `enactive.models.pi.model.EnactivePi`
and its continuous action generator is `enactive/models/pi/flow_matching_generator.py`.
The saved configurations support this reconstruction; they do not preserve the
exact original shell invocation. Accelerate's `mixed_precision: no` coexists with
the explicit BF16 parameter/compute contract in the runtime configuration.

## Input and output contract

Provide RGB uint8 HWC images named `front`, `left` and `right`, an instruction
string, and the current raw 14-dimensional joint state. Camera order is front,
left wrist, right wrist. The processor uses PIL bilinear resizing with preserved
aspect ratio and black padding to 224 by 224, then float32 scaling to [-1, 1].

Text follows `openpi_pi05_instruction_3view.py`: strip surrounding whitespace,
replace underscores and newlines with spaces, tokenize using the checkpoint's
SentencePiece model with BOS, append the separately tokenized newline, truncate
to 200 tokens and pad with token 0. The instruction contains no discretized state.
The normalized state remains in the pipeline transition for diagnostics; it is
not an engine input for this checkpoint's instruction-only contract.

The engine returns `[B, 50, 32]` normalized actions. The processor keeps the first
14 dimensions and applies checkpoint quantiles:

```text
physical = (normalized + 1) / 2 * (q99 - q01 + 1e-6) + q01
absolute_joint_target = physical + current_raw_state
```

The state addition applies only to joints 0-5 and 7-12. Grippers 6 and 13 are
already absolute and must not receive the state anchor. All 50 steps use the same
observation anchor; deltas are not accumulated. Values are not clipped.
Postprocessing returns CPU float32. Always supply the raw state from the matching
observation; the processor stores no previous observation.

## Conversion and inference

Run from the repository root with a working Phyai environment. The checkpoint
includes a fast `tokenizer.json` exported from its exact original vocabulary.
The existing tokenizer loader uses fastokens when installed, or the HuggingFace
Rust tokenizer otherwise. Both were verified against the original SentencePiece
IDs. No SentencePiece package is required for inference; it was uninstalled
from the Phyai test environment. The existing `fastokenizer` extra remains
optional.

The following conversion command uses local, untracked scripts. Conversion of
an original `.model` file requires SentencePiece once in the export environment;
that dependency is not declared by Phyai. The already converted checkpoint is
ready for inference.

```bash
uv run --no-sync python examples/pi05/convert_enactive.py \
  "$ENACTIVE_RUN/inference-500" \
  .cache/checkpoints/enactive-pi05-step500 \
  --reference-config "$ENACTIVE_RUN/checkpoint-500/config.json.bak-before-model-type-20260922T080457"

uv run --no-sync python examples/pi05/run_enactive.py \
  --checkpoint .cache/checkpoints/enactive-pi05-step500 \
  --input observation.npz \
  --output actions.npz
```

Set `ENACTIVE_RUN` to the source run directory. Conversion requires a new output
directory and rejects an existing destination. It streams and verifies every
mapped tensor, preserves source sidecars and writes an explicit Phyai geometry
configuration, a fast tokenizer, and `policy_preprocessor.json` /
`policy_postprocessor.json` with normalization sidecars. There are 819 source tensors: 811 BF16 inference tensors are
mapped without changing any bytes; eight unused state-projection or training
statistics tensors are omitted with individual reasons. Phyai's strict loader
loaded all 811 tensors with zero missing or unexpected keys. Its 110 normalization
parameter casts from BF16 to FP32 preserve the stored values exactly.

`observation.npz` contains `front`, `left`, `right`, `state` and scalar string
`task`. An optional float32 `noise` array with shape `[1, 50, 32]` supplies the
initial noise. Otherwise `--seed`, default 42, generates CPU noise. The output
contains `actions` with shape `[1, 50, 14]` and `normalized_actions` with shape
`[1, 50, 32]`. Camera input is RGB, not OpenCV's default BGR.

CUDA Graph is enabled by default. `--no-cuda-graph` is available for this one-shot
example. For a persistent engine, use the default: the existing eager runner can
reject a later, longer prompt if its first attention wrapper was initialized
with a shorter capacity. The local eager comparison initializes the longest
prompt first. No scheduler workaround is installed by this adapter.

The example loads `PI05Processor.from_pretrained` with the checkpoint itself as
`tokenizer_name`, `image_resize_backend="pil"`, `normalize_pixels=True`,
`action_dim=14` and float32 preprocessing. It converts raw HWC arrays to the
existing processor's BCHW camera-tensor interface.

For batches, pass a list of instructions, `[B,14]` states and one `[B,3,H,W]`
uint8 tensor per camera. Images within each camera batch must share dimensions;
observations of different sizes can be processed separately and concatenated.
Construct the existing `PI05Args` with the required `max_batch_size` and decode
using the matching raw states. The serialized postprocessor's delta mask keeps
gripper predictions absolute.

## Validation results

After the processor reuse revision, both fastokens 0.3.2 and the existing
HuggingFace Rust backend passed 1,956 checks each with SentencePiece absent.
The 1,931 text cases cover multilingual and long prompts, control-token strings,
Unicode, empty input, and repeated or mixed whitespace. Token IDs and lengths
match SentencePiece exactly. The remaining checks compare five independent
Enactive preprocessing/postprocessing fixtures, all 12 saved model input/output
fixtures, serialization, batching and invalid action anchors. Forty existing
processing regression tests also passed.

The fast tokenizer export keeps whitespace symbols in its BPE vocabulary rather
than registering them as AddedTokens. This avoids matching literal whitespace
markers before normalization and splitting mixed spaces into different tokens.
Both backends passed the full corpus with this artifact. Prompt cleanup and the
separately encoded newline are configured through shared pipeline steps.

The 12 single and batched GPU comparisons and the runnable example were repeated
using the reused processor. The example and all single-request outputs remain
bitwise identical to the earlier Phyai results reported below.

All model comparisons used the same exported checkpoint and explicit initial
noise, on one H800 held by `salloc` through the end of GPU testing. Reference:
Python 3.10, Torch 2.8.0+cu128, Transformers 4.57.1. Phyai: Python 3.12,
Torch 2.11.0+cu130, Transformers 5.8.1, FlashInfer 0.6.17. The isolated Phyai
environment used apache-tvm-ffi 0.1.14.post0 with its installed CUTLASS DSL;
the older 0.1.9 package failed FlashInfer kernel compilation. Project runtime
pins were not changed.

The 12 generated observations use independent random RGB arrays with seeds
1701-1712, CPU Torch noise seeds 42-53, landscape/portrait/square inputs, English
instructions with whitespace, Chinese instructions, and text exceeding 200
tokens. State values vary within saved quantiles. These are synthetic numerical
fixtures, not frames from the training dataset.

| Check | Result |
| --- | --- |
| Original processor characterization against Enactive runtime | 44 passed |
| Original weight converter tests, before offline processor export was added | 24 passed |
| Current shared processing regression tests, including PI0 / PI05 | 40 passed |
| Full checkpoint load, both engines | No missing or unexpected tensors |
| Pixel values, language tokens/lengths, normalized states | Exact agreement |
| Postprocessing with common normalized actions | Exact agreement |
| First-velocity cosine, worst of 12 | 0.999897385 |
| Full 32D final-action cosine, worst of 12 | 0.998695840 |
| Valid 14D final-action cosine, worst of 12 | 0.998697594 |
| Valid 14D final-action RMSE, worst sample | 0.023440104 |
| Decoded action RMSE, worst sample | 0.003382853 |
| Normalized maximum absolute error, single requests | 0.121368647 |
| Decoded maximum absolute error, single requests | 0.014956951 |
| CUDA Graph versus eager, 12 single requests | Bitwise equal |
| Repeated identical requests | Bitwise equal |
| Delivered inference example versus test harness | Bitwise equal |
| Batch sizes 2, 1, 3, 2, 3, 1; all 12 samples | Passed; worst valid-action cosine 0.998694977 |
| Batched versus single-request action cosine | At least 0.999992080 |
| Normalized maximum absolute error, batched requests | 0.123321772 |
| Decoded maximum absolute error, batched requests | 0.014956951 |

The cosine gate is above 0.99 on both the first velocity and final action,
including the valid 14 dimensions alone. All outputs were finite. The largest
single-request normalized deviation occurs in sample 6, gripper dimension 13;
its decoded deviation is 0.006487217. The largest decoded deviation occurs in
sample 9, joint dimension 8. The per-dimension single-request maxima are:

```text
[0.001533829, 0.000605948, 0.000203868, 0.006423771, 0.003468722,
 0.011439562, 0.000752971, 0.002411485, 0.014956951, 0.012910247,
 0.002970695, 0.003070772, 0.003145784, 0.006487217]
```

Differences begin inside the vision stack. The existing implementations differ
in GELU, BF16 intermediate rounding, attention kernels and action-head compute
precision. Phyai uses exact GELU and BF16 heads; the selected Enactive reference
uses its tanh GELU path and FP32 action projections. This comparison measures the
combined numerical effect; it does not assign each error to one operation.
Bitwise agreement with Enactive would require changes beyond this input/output
adapter. Actual robot tolerances and task success remain unmeasured.

Repository instructions keep model tests outside the committed tree. The local
`.cache/enactive-align/reuse` directory contains the current `test_reuse.py`,
`build_references.py`, independent tokenizer/processor fixtures, JUnit files and
logs for both backends. `.cache/enactive-align` also contains the 12 fixed model
inputs, Enactive outputs, `phyai_probe.py`, `batch_probe.py` and
`verify_results.py`. Older characterization tests remain as historical artifacts;
the current processor suite is `reuse/test_reuse.py`.

`conversion_report.json` in the checkpoint directory records source/output file
SHA256 values and all tensor mappings. The tokenizer and processor artifacts
remain local with the weights. Conversion scripts at
`examples/pi05/convert_enactive.py` and
`phyai-utils-tools/src/phyai_utils_tools/models/pi05/convert_enactive_pi05.py`
are untracked and are not part of the branch's final source tree.

The reference fixtures were generated in the Enactive environment. Tests and
inference run in Phyai's Python 3.12 environment. The local `phyai-python`
launcher supplies the cluster's CUDA compatibility library and environment
paths. Commands for the current validation are:

```bash
.venv/bin/python -m pytest -c /dev/null --confcutdir=.cache/enactive-align/reuse \
  -p no:cacheprovider .cache/enactive-align/reuse/test_reuse.py -q
PHYAI_REUSE_BACKEND=hf .venv/bin/python -m pytest -c /dev/null \
  --confcutdir=.cache/enactive-align/reuse -p no:cacheprovider \
  .cache/enactive-align/reuse/test_reuse.py -q
# Run GPU commands inside the retained Slurm allocation.
.cache/enactive-align/phyai-python .cache/enactive-align/phyai_probe.py --count 12 --graphs
.cache/enactive-align/phyai-python .cache/enactive-align/batch_probe.py
.cache/enactive-align/phyai-python examples/pi05/run_enactive.py \
  --checkpoint .cache/checkpoints/enactive-pi05-step500 \
  --input .cache/enactive-align/case0.npz --output .cache/enactive-align/reuse/example-result.npz
.venv/bin/python .cache/enactive-align/verify_results.py
scripts/run_pre_commit.sh
```
