# Qwen-Image 2.1

Use [run_qwen_image_21.py](run_qwen_image_21.py) to generate or edit an image with
Qwen-Image 2.1 and save the result as a PNG.

## Setup

You need Python 3.12 or newer, `uv`, CUDA-capable NVIDIA GPUs, and a local
Qwen-Image-2.1 checkpoint. Download the weights from
[Qwen/Qwen-Image-2.1 on Hugging Face](https://huggingface.co/Qwen/Qwen-Image-2.1).

Run all commands below from the repository root. Install the workspace and set
the path to your checkpoint:

```bash
uv sync
export QWEN_IMAGE_CHECKPOINT=/path/to/Qwen-Image-2.1
```

Check GPU availability with `nvidia-smi` before running inference. The examples
use GPUs 0 and 1; change `CUDA_VISIBLE_DEVICES` to select available GPUs.

## Generate an image

This command generates a 512 x 512 image using two GPUs and the included
[kernel policy](kernel_policy.yaml):

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python examples/qwen_image_21/run_qwen_image_21.py \
  --checkpoint "$QWEN_IMAGE_CHECKPOINT" \
  --kernel-policy examples/qwen_image_21/kernel_policy.yaml \
  --prompt "An orange cat sitting beside a window in warm sunlight" \
  --tp 2 \
  --height 512 \
  --width 512 \
  --steps 40 \
  --seed 42 \
  --out .cache/qwen_image_21.png
```

The Engine starts the worker processes automatically. Launch the script with
`python`; an additional `torchrun` launcher is not needed. The output directory
is created if necessary, and the image is saved to `.cache/qwen_image_21.png`.

## Edit an image

Pass an input image with `--image` and describe the edit in `--prompt`:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python examples/qwen_image_21/run_qwen_image_21.py \
  --checkpoint "$QWEN_IMAGE_CHECKPOINT" \
  --kernel-policy examples/qwen_image_21/kernel_policy.yaml \
  --image /path/to/input.png \
  --prompt "Change the background to a snowy forest while keeping the subject unchanged" \
  --tp 2 \
  --height 512 \
  --width 512 \
  --steps 40 \
  --seed 42 \
  --out .cache/qwen_image_21_edited.png
```

Repeat `--image` to supply multiple reference images, up to 10.

Without `--height` and `--width`, generation defaults to 1024 x 1024. Editing
chooses a size based on the last reference image. Set both options to choose
your output size, using multiples of 32.

## Parallelism and guidance

The required GPU count is `--tp` multiplied by `--cfg`. For one GPU, use
`CUDA_VISIBLE_DEVICES=0` and `--tp 1`. The defaults are `--tp 1 --cfg 1`.

Classifier-free guidance requires both `--guidance-scale` greater than 1 and
`--negative-prompt`. An empty negative prompt is accepted. For example, add
`--guidance-scale 4.0 --negative-prompt ""` to either command above. With the
default `--cfg 1`, the two guidance branches run sequentially.

To run the guidance branches in parallel, also add `--cfg 2`. With `--tp 2`,
this requires four visible GPUs, for example `CUDA_VISIBLE_DEVICES=0,1,2,3`.
Setting `--cfg 2` alone does not enable guidance.

## Common options

| Option | Behavior |
| --- | --- |
| `--kernel-policy PATH` | Load a kernel policy YAML file. Omit it to use the engine's configured policy or defaults. |
| `--steps N` | Set the number of inference steps. Default: `40`. |
| `--seed N` | Set the noise seed. Default: `42`. |
| `--out PATH` | Choose the output image path. Default: `.cache/qwen_image_21.png`. |

View all arguments with:

```bash
uv run --no-sync python examples/qwen_image_21/run_qwen_image_21.py --help
```
