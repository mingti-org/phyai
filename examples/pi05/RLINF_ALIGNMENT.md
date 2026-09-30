# RLinf Tianji pi0.5 inference

`run_rlinf.py` runs a converted `openpi_rlinf` Tianji checkpoint through the
existing `PI05Processor`, PI05 engine, and `kernel_policy_rlinf_bf16.yaml`.
It accepts raw camera images, a task, and joint state, and returns absolute
joint and gripper targets. This change adds the example and this report;
model, processor, runtime, and dependency files are unchanged.

## Run the example

The converted checkpoint directory must contain:

- `config.json` with `chunk_size=50` and `num_inference_steps=5`;
- the safetensors weights and index;
- a local fast tokenizer (`tokenizer.json` and its configuration files);
- `norm_stats.json` and `rlinf_metadata.json` from the conversion.

Create an observation NPZ from the robot's RGB cameras and state:

```python
import numpy as np

np.savez(
    "observation.npz",
    head_left=head_left_rgb,      # uint8 HWC
    left_wrist=left_wrist_rgb,    # uint8 HWC
    right_wrist=right_wrist_rgb,  # uint8 HWC
    state=joint_state,            # float32, shape (16,)
    task="Plug in the Ethernet cable",
)
```

Run on an allocated GPU with the Phyai environment:

```bash
uv run --no-sync python examples/pi05/run_rlinf.py \
  --checkpoint /path/to/rlinf-pi05-tianji-step3810 \
  --input observation.npz \
  --output actions.npz
```

The output has `actions` of shape `(1, 50, 16)` and `normalized_actions` of
shape `(1, 50, 32)`. Use `--no-cuda-graph` for eager execution. `--seed`
controls CPU noise generation; supplying a float32 `noise` array of shape
`(1, 50, 32)` in the input NPZ overrides it for comparisons with another engine.
The example writes predicted targets to disk; it does not execute them on a robot.

## Input and output contract

The camera order is `head_left`, `left_wrist`, `right_wrist`, corresponding to
OpenPI's `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`. PIL bilinear
resize preserves aspect ratio, centers each image in a black 224 × 224 square,
and retains uint8 rounding. The existing processor maps pixels to `[-1, 1]`.

State uses the Tianji `plug_in_ethernet_h50_s2` quantile statistics, NumPy's
float64 statistic precision, and epsilon `1e-6`. The existing PI05 prompt step
discretizes the 16 real state coordinates before any model padding. The local
fast tokenizer adds BOS, pads on the right, and truncates at 200 tokens.
Phyai inference does not require SentencePiece or an RLinf import.

Output processing slices to 16 coordinates, applies inverse quantile
normalization, and adds the observed state to arm coordinates `0:7` and `8:15`.
Grippers at indices 7 and 15 remain absolute. The checkpoint was trained with
an action stride of 2 at 50 Hz, so successive chunk targets are 40 ms apart.

## Conversion and reference

The source is the final production checkpoint `global_step_3810`, trained for
10 epochs with a 50-action horizon. The conversion used RLinf's existing
`new_to_old_state_dict` mapping. All 667 source tensors were consumed; splitting
vision QKV and expert MLP tensors produced 811 tensors in three safetensors
shards. Every saved tensor was read back and verified exactly. Native storage
precision was preserved: the source has 545 BF16 and 122 FP32 tensors; the
converted layout has 689 BF16 and 122 FP32 tensors. Phyai's strict loader
reported **811 loaded, zero missing, zero unexpected**.

Source `full_weights.pt` SHA-256:
`7b8570c93724035adbd02abd4b170e5de7a7969f7b5d2db3a38f01833ec7945a`.

The reference used RLinf revision
`805858b84a33e74b6c8e3a05b5c96379a61224cf` with the existing local Tianji SFT
changes, its installed environment, the saved training configuration, and
`get_model(...).sample_actions(..., num_steps=5, noise=...)`. The exact local
source snapshot was recorded in a content manifest because the revision alone
does not describe those local changes. Reference source contents were unchanged;
no packages were installed, removed, or reconfigured. Importing `numpydantic`
refreshed a generated type stub, so its original timestamp was restored and
a write guard was added to the local reference scripts. The final environment
check compares file sizes and modification times against the initial snapshot.
Conversion helpers and model comparison scripts remain local.

## Validation results

Twelve observations cover the start, 40%, and 80% points of episodes
0, 9, 18, and 27 in `plug_in_ethernet`. Both implementations used the same
checkpoint, observation, and saved noise for each case, with five Euler steps.

- All real-observation token IDs and masks matched. Another 200 prompts,
  including whitespace, Chinese, special-token text, and truncation cases,
  matched the reference SentencePiece tokenizer exactly using fastokens.
- Resized uint8 pixels matched exactly. CPU versus GPU pixel normalization
  differed by at most `1.1921e-7`.
- Given identical normalized actions, output processing differed by at most
  `4.4409e-16`.
- The first predicted velocity had minimum cosine similarity **0.999963**.
- Eager and CUDA Graph outputs were bitwise identical for batch size 1.
  Every repeated request was deterministic. The public example reproduced
  the comparison harness output exactly.

The table reports the worst case across all 12 observations in each mode.
Joint and gripper errors are separated because they have different units.

| Execution | Min normalized-action cosine | Min physical-action cosine | Max arm-joint error (rad) | Max gripper error |
| --- | ---: | ---: | ---: | ---: |
| Eager, batch 1 | 0.999087 | 0.999980 | 0.019100 | 0.136225 |
| CUDA Graph, batch 1 | 0.999087 | 0.999980 | 0.019100 | 0.136225 |
| CUDA Graph, batch 3 | 0.999104 | 0.999980 | 0.019498 | 0.133833 |

These results pass the repository's cosine threshold of 0.99 for both the
first velocity and complete inference. They are numerical alignment results,
not bitwise equality with RLinf. In particular, a gripper coordinate in one
sample differs by 0.136225; the high aggregate cosine does not remove that
absolute difference.

The existing BF16 engine uses paged attention and BF16 action/time projections.
RLinf uses FP32 attention scores and FP32 action/time projections. Its current
loader also rounds selected FP32 backbone tensors through BF16 before restoring
FP32 storage; 112 tensors changed value through that loading path. The
conversion preserves original checkpoint values on disk, while each engine
uses its existing loading and compute precision. These differences, together
with kernel rounding, accumulate through the Euler loop. No model or kernel
changes were made to pursue bitwise equality.
