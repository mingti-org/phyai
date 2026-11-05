# HumanWAM-Goal

Model weights: [HumanWAM-Goal on Hugging Face](https://huggingface.co/collections/DAGroup-PKU/open-humanscale).

Predict a 48-step action chunk from an instruction, goal image, camera views, and robot state.

HumanWAM-Goal uses the `humanwam_goal` plugin and the `examples/HumanWAM-Goal` entry point.
See the inference guide in [English](../../docs/models/humanwam-goal/inference.mdx)
or [中文](../../docs/zh/models/humanwam-goal/inference.mdx).

Run from the repository root with this PhyAI checkout installed. The tested environment
uses Hopper GPUs (SM90), Torch 2.10 or 2.11, CUDA 12.8, and GLIBC 2.34 or newer.
TP=2/4 requires GPUs connected by NVLink on one host and `nvcc`.
Each GPU retains full projection weights for the token-partitioned Generator
blocks. Budget roughly the single-GPU weight memory per GPU when using TP.

Install the additional attention dependency in the same environment:

```bash
uv pip install --no-deps \
  --index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0 \
  'flash-attn-3-nv==1.0.3+cu128.torch210'

export HUMANWAM_GOAL_CHECKPOINT=/path/to/humanwam-goal
export HUMANWAM_GOAL_SAMPLE=/path/to/observation.json
```

The inference checkpoint must contain `transformer/`, `vae/`, `text_tokenizer/`,
`sampling.json`, and `processor.json`.
Sampling coefficients are computed at setup; `corrector_weights` is optional.
For parity with the released HumanNet native backend, use its Torch 2.10 and
cuBLASLt 12.8.3.14 [reproduction environment](https://github.com/DAGroup-PKU/HumanNet/tree/open-humanscale/humanwam_goal#reproducing-published-results).

Run on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python examples/HumanWAM-Goal/run_humanwam_goal.py \
  --checkpoint "$HUMANWAM_GOAL_CHECKPOINT" \
  --sample "$HUMANWAM_GOAL_SAMPLE" \
  --output .cache/humanwam-goal/actions.npy
```

For two GPUs, use `CUDA_VISIBLE_DEVICES=0,1` and add `--tp 2`; for four, use
`CUDA_VISIBLE_DEVICES=0,1,2,3` and `--tp 4`. The Engine starts the workers itself.
CUDA Graph is enabled by default; add `--no-cuda-graph` for eager execution.

## Input and output

Create `observation.json` using the following layout. Image paths are relative to
the JSON file. Replace the example state values and domain ID with your observation.

```json
{
  "instruction": "Move the object to the position shown in the goal image.",
  "images": {"head": "head.png", "left": "left.png", "right": "right.png"},
  "goal_images": {"head": "goal.png"},
  "observation_timestamp": 0.0,
  "history": [],
  "normalized_state": [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ],
  "domain_id": 0,
  "seed": 0
}
```

For visual history, add oldest-first `{"timestamp": ..., "images": {...}}` entries
to `history`. For RTC, add a `normalized_prefix` matrix within the checkpoint's
prefix-length limit. State and prefix values use the checkpoint's normalization
and model coordinates.

The output is a NumPy array of normalized actions, shaped `[1, 48, 49]` with the
default configuration. Convert it to your robot's coordinates before execution.
