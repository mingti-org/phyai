# MOSS-Image

Model: [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT).

Run from the repository root. Set the checkpoint path and choose available GPUs.

```bash
uv sync
export MOSS_CHECKPOINT=/path/to/MOSS-Image-SFT
```

Start the server and **Gradio UI** in one terminal. Open http://127.0.0.1:7860; Ctrl+C stops both.

```bash
CUDA_VISIBLE_DEVICES=0 python examples/moss_image/gradio_moss_image.py launch \
  --checkpoint "$MOSS_CHECKPOINT" \
  --devices 0 --cpu-offload --torch-compile \
  --graph-bucket 512 512 128 \
  --graph-bucket 1024 1024 256 \
  --graph-cache-size 4
```

`--graph-bucket HEIGHT WIDTH TEXT_TOKENS` prewarms a bucket. Height and width are pixels, divisible by 16; text capacity rounds up to 64, 128, ..., 4096. Each unique bucket needs two cache entries. Omit these flags to capture on demand. Other request shapes can reuse cached graphs or trigger a new capture.

Use two GPUs for concurrent requests, with one model replica per GPU. `--devices` uses logical indices within `CUDA_VISIBLE_DEVICES`.

```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/moss_image/gradio_moss_image.py launch \
  --checkpoint "$MOSS_CHECKPOINT" \
  --devices 0 1 --concurrency 2 --cpu-offload --torch-compile \
  --graph-bucket 1024 1024 256
```

Run the server and UI separately:

```bash
# Terminal 1: server
CUDA_VISIBLE_DEVICES=0 python examples/moss_image/gradio_moss_image.py serve \
  --checkpoint "$MOSS_CHECKPOINT" \
  --devices 0 --cpu-offload --torch-compile \
  --host 127.0.0.1 --port 50063
```

```bash
# Terminal 2: UI
python examples/moss_image/gradio_moss_image.py ui \
  --server-address 127.0.0.1:50063 \
  --host 127.0.0.1 --port 7860
```

Generate an image from the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/moss_image/run_moss_image.py \
  --checkpoint "$MOSS_CHECKPOINT" \
  --prompt 'A red apple on a wooden table.' \
  --height 1024 --width 1024 --steps 50 --seed 42 \
  --cpu-offload --torch-compile \
  --out .cache/moss_image.png
```
