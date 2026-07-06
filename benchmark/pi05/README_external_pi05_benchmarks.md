# External PI0.5 benchmark wrappers

This directory contains wrappers for benchmarking external PI0.5 inference runtimes with the same high-level settings and common `bench_n_batch.py` runner used by the PhyAI PI0.5 benchmarks.

These wrappers do **not** call PhyAI's engine. Each wrapper calls the target runtime directly, and all machine-local paths must be passed by command-line flags or environment variables.

| File | Runtime measured | Main timing scope |
| --- | --- | --- |
| `bench_flashrt_pi05.py` | FlashRT | direct `Pi05TorchFrontendRtx.infer(...)` hot path after `set_prompt`, calibration, and the first graph-building infer |
| `bench_realtime_vla_pi05.py` | realtime-vla | `Pi05Inference.forward()` hot path |
| `bench_vlacpp_pi05_client.py` | vla.cpp | ZMQ client request wall time; server phase timings are stored in JSONL `extras` |

## Common settings

Use the same settings across machines when possible:

| Setting | Recommended value |
| --- | --- |
| batch size | 1 via `--batch-sizes 1` |
| views | 2 synthetic views / camera streams for latency-only runs |
| chunk size | 50 for strict comparison rows |
| warmup | 100 via `--n-warmup` |
| timed iterations | 100 via `--n-timed` |
| result format | JSONL via `--result-file` |
| prompt | fixed prompt or fixed prompt length, recorded in JSONL `extras` |
| precision | BF16 fair row; optimized rows must be labeled separately |

Before running, verify the GPU is idle:

```bash
nvidia-smi
uptime
```

The examples below use placeholders rather than host-specific paths:

| Placeholder | Meaning |
| --- | --- |
| `<PHYAI_REPO>` | This PhyAI checkout containing `benchmark/pi05/` |
| `<FLASHRT_REPO>` | FlashRT repository clone |
| `<REALTIME_VLA_REPO>` | realtime-vla repository clone |
| `<VLACPP_REPO>` | vla.cpp repository clone |
| `<PI05_CHECKPOINT>` | PI0.5 checkpoint directory or file for the selected runtime |
| `<TOKENIZER_DIR_OR_ID>` | Local tokenizer directory or HF id used by vla.cpp client |
| `<LIBERO_STATS_JSON>` | Local LIBERO `meta/stats.json` for PI0.5 state tokenization |

## Environment setup

These scripts assume the target runtime can already be imported or executed. Keep each official repo at a known commit and record it with your results.

FlashRT:

```bash
git clone https://github.com/flashrt-project/FlashRT <FLASHRT_REPO>
cd <FLASHRT_REPO>
# Install/build FlashRT following its official README for your GPU/CUDA stack.
export FLASHRT_ROOT=<FLASHRT_REPO>
```

Notes: on RTX 5090/SM120, make sure the CUDA toolkit used to build extensions supports the GPU. For the BF16 fair row, the wrapper sets `FVK_PI05_RTX_FORCE_BF16=1`. Do not use `load_model(..., num_steps=50)` for chunk size; FlashRT `num_steps` means denoise steps.

realtime-vla:

```bash
git clone https://github.com/Dexmal/realtime-vla <REALTIME_VLA_REPO>
cd <REALTIME_VLA_REPO>
# Install realtime-vla following its official README.
export REALTIME_VLA_ROOT=<REALTIME_VLA_REPO>
```

If you pass a PI0.5 `model.safetensors`, also set `FLASHRT_ROOT` because the wrapper reuses FlashRT's checkpoint conversion helper. Loading `.pkl/.pickle` checkpoints requires `--trust-pickle-checkpoint` and should only be used for trusted files.

vla.cpp:

```bash
git clone https://github.com/VinRobotics/vla.cpp <VLACPP_REPO>
cd <VLACPP_REPO>
# Build vla-server following its official README, for example into build_sm120/.
export VLACPP_ROOT=<VLACPP_REPO>
```

Prepare the PI0.5 GGUF model, `mmproj` GGUF, tokenizer, and LIBERO `stats.json` before running. Prefer local tokenizer and stats paths to avoid network/auth issues.

Quick smoke test after setup: use the runtime-specific command below, replace the real output path with a scratch file, and set `--n-warmup 1 --n-timed 1`.

## FlashRT

`--flashrt-root` can be omitted if `FLASHRT_ROOT=<FLASHRT_REPO>` is set.

```bash
cd <PHYAI_REPO>
python benchmark/pi05/bench_flashrt_pi05.py \
  --flashrt-root <FLASHRT_REPO> \
  --checkpoint <PI05_CHECKPOINT> \
  --precision bf16 \
  --num-views 2 \
  --chunk-size 50 \
  --batch-sizes 1 \
  --n-warmup 100 \
  --n-timed 100 \
  --result-file results/flashrt_pi05_external.jsonl
```

For FlashRT optimized precision, use `--precision fp8_bf16` and keep it in a separate table row.

## realtime-vla

`--realtime-vla-root` can be omitted if `REALTIME_VLA_ROOT=<REALTIME_VLA_REPO>` is set. If `--checkpoint` points to a PI0.5 `model.safetensors` file or a directory containing `model.safetensors`, pass `--flashrt-root <FLASHRT_REPO>` as well because the wrapper reuses FlashRT's PI0.5 safetensors conversion helper.

```bash
cd <PHYAI_REPO>
python benchmark/pi05/bench_realtime_vla_pi05.py \
  --realtime-vla-root <REALTIME_VLA_REPO> \
  --flashrt-root <FLASHRT_REPO> \
  --checkpoint <PI05_CHECKPOINT> \
  --num-views 2 \
  --chunk-size 50 \
  --prompt-len 16 \
  --batch-sizes 1 \
  --n-warmup 100 \
  --n-timed 100 \
  --result-file results/realtime_vla_pi05_external.jsonl
```

If you already have a realtime-vla `.pkl`, `.pt`, or `.pth` checkpoint, `--flashrt-root` is not needed.

## vla.cpp

Start `vla-server` first. The PI0.5 GGUF server reports server phase timing when started with `--timing-detail phase`.

Example server:

```bash
<VLACPP_REPO>/build_sm120/vla-server \
  --bind tcp://127.0.0.1:5555 \
  --timing-detail phase \
  <MM_PROJ_GGUF> \
  <PI05_GGUF>
```

Then run the client benchmark. `--vlacpp-root` can be omitted if `VLACPP_ROOT=<VLACPP_REPO>` is set.

```bash
cd <PHYAI_REPO>
python benchmark/pi05/bench_vlacpp_pi05_client.py \
  --vlacpp-root <VLACPP_REPO> \
  --addr tcp://127.0.0.1:5555 \
  --arch pi05 \
  --tokenizer <TOKENIZER_DIR_OR_ID> \
  --stats-json <LIBERO_STATS_JSON> \
  --num-views 2 \
  --chunk-size 50 \
  --batch-sizes 1 \
  --n-warmup 100 \
  --n-timed 100 \
  --result-file results/vlacpp_pi05_external.jsonl
```

For reproducibility, prefer a local tokenizer path for `--tokenizer`. The default PI0.5 tokenizer id, `google/paligemma-3b-pt-224`, may require HuggingFace access and authentication. Passing `--stats-json` avoids implicit network fetches for LIBERO state quantile statistics.

## Notes

- `bench_flashrt_pi05.py` and `bench_realtime_vla_pi05.py` reuse the PhyAI `NBatchBenchRunner` JSONL schema.
- `bench_flashrt_pi05.py` records wall latency with the runner's perf-counter path because FlashRT may run work on an internal CUDA stream; the step synchronizes CUDA before returning.
- `bench_realtime_vla_pi05.py` uses the runner's CUDA event timing around `Pi05Inference.forward()`.
- `bench_vlacpp_pi05_client.py` requires a running `vla-server`; the runner records client wall latency, and server phase latency is recorded in JSONL `extras.server_phase_latency_ms`.
- vla.cpp phase timing may expose `vision` and combined `inference`, not necessarily separate prefix/expert timing.
- These wrappers are intended for latency reproduction and support only `--batch-sizes 1` for now. Component MFU tables still require the shared PI0.5 FLOP model and component timings, as described in `thor_pi05_benchmark_plan.md`.
