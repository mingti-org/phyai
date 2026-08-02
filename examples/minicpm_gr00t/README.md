# MiniCPM-GR00T local deployment

Use this example to run the MiniCPM-V 4.6 + GR00T unified-80D policy with the
PhyAI Engine. It runs local inference only and does not start a network service.

## Requirements

- Linux with a CUDA-capable NVIDIA GPU
- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- A compatible MiniCPM-GR00T checkpoint (`.pth` or safetensors)
- The MiniCPM-V 4.6 processor/tokenizer directory

PhyAI loads the original `.pth` checkpoint directly, so no conversion is needed.

## Install

Run the following command from the repository root:

```bash
uv sync
```

`uv sync` creates `.venv` and installs the workspace packages and their locked
dependencies. Run the commands below with `uv run` so they use that environment.

## Run one inference

```bash
export CKPT=/path/to/rank_0_jobid_552915_iter_40000.pth
export PROCESSOR=/path/to/MiniCPM-V-4.6

PHYAI_USE_CUDA_GRAPH=1 \
uv run python examples/minicpm_gr00t/run_minicpm_gr00t.py \
  --checkpoint "${CKPT}" \
  --vlm-path "${PROCESSOR}" \
  --instruction "open the middle drawer of the cabinet" \
  --seed 123
```

When you omit `--image`, the example uses two deterministic blank 224×224 RGB
frames. For real inputs, pass exactly two images, with the base-camera image
first and the wrist-camera image second:

```bash
PHYAI_USE_CUDA_GRAPH=1 \
uv run python examples/minicpm_gr00t/run_minicpm_gr00t.py \
  --checkpoint "${CKPT}" \
  --vlm-path "${PROCESSOR}" \
  --image /path/to/base_camera.png \
  --image /path/to/wrist_camera.png \
  --instruction "open the middle drawer of the cabinet" \
  --seed 123 \
  --save-actions /tmp/minicpm_gr00t_actions.pt
```

The script prints an action tensor with shape `(1, 30, 80)`, dtype
`torch.float32`, and `finite=True`. The first run can take longer while the
kernels and CUDA Graph are initialized.

## Verify

```bash
uv run pytest -q \
  phyai-utils-tools/tests/test_minicpm_gr00t_processor.py

uv run ruff check \
  examples/minicpm_gr00t \
  phyai-utils-tools/src/phyai_utils_tools/models/minicpm_gr00t \
  phyai-utils-tools/tests/test_minicpm_gr00t_processor.py
```

Fixed-input numerical checks are not a measure of task-level accuracy. Evaluate
closed-loop policy quality separately on a benchmark such as LIBERO.
