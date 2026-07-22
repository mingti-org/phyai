# External PI0.5 runtime latency wrappers

This document explains how to set up and run the three external PI0.5 latency
wrappers under `benchmark/pi05/`.

These scripts do not use the PhyAI engine for inference. Each script calls the
target runtime directly, while reusing PhyAI's common benchmark runner for
warmup, timing, and JSONL output.

| Script | Runtime measured | Timed call |
| --- | --- | --- |
| `bench_flashrt_pi05.py` | FlashRT | `Pi05TorchFrontendRtx.infer(obs)` |
| `bench_realtime_vla_pi05.py` | realtime-vla | `Pi05Inference.forward(...)` |
| `bench_vlacpp_pi05_client.py` | vla.cpp | one ZMQ request to a running `vla-server` |

## Common setup

Use one Python environment that can import PhyAI, PyTorch, and
`benchmark/bench_n_batch.py`.

```bash
cd <PHYAI_REPO>
python -c "import torch; import phyai; import benchmark.bench_n_batch"
nvidia-smi
```

Use the same benchmark settings when comparing runtimes:

```text
batch size: 1
views / camera streams: 2
chunk size: 50
prompt: keep the same text across runs
warmup / timed iterations: use the same values across runs
precision: label each row by the runtime's real precision path
```

The wrappers generate synthetic image/state inputs. They are for latency-only
measurements, not LIBERO accuracy evaluation.

## Placeholders

Use your own paths for these placeholders:

| Placeholder | Meaning |
| --- | --- |
| `<PHYAI_REPO>` | PhyAI checkout containing `benchmark/pi05/` |
| `<FLASHRT_REPO>` | FlashRT checkout |
| `<REALTIME_VLA_REPO>` | realtime-vla checkout |
| `<VLACPP_REPO>` | vla.cpp checkout |
| `<VLA_SERVER>` | compiled vla.cpp `vla-server` binary |
| `<PI05_CHECKPOINT>` | PI0.5 safetensors checkpoint directory or file |
| `<PI05_GGUF>` | PI0.5 GGUF file for vla.cpp |
| `<MM_PROJ_GGUF>` | vla.cpp multimodal projector GGUF |
| `<TOKENIZER_DIR>` | local tokenizer directory |
| `<LIBERO_STATS_JSON>` | local LIBERO `meta/stats.json` with `observation.state.q01/q99` |

## FlashRT

Install FlashRT following its official README, then make the checkout visible:

```bash
export FLASHRT_ROOT=<FLASHRT_REPO>
cd <PHYAI_REPO>
python -c "import sys; sys.path.insert(0, '$FLASHRT_ROOT'); from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx; print(Pi05TorchFrontendRtx)"
```

Run latency:

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
  --result-file results/flashrt_pi05.jsonl
```

Notes:

- The script uses FlashRT's direct `Pi05TorchFrontendRtx` API because `chunk_size`
  is a frontend constructor argument.
- Do not use `load_model(..., num_steps=50)` to set action chunk size. In
  FlashRT, `num_steps` means denoise steps.
- `--precision bf16` sets FlashRT's forced-BF16 PI0.5 RTX path. Use
  `--precision fp8_bf16` for FlashRT's optimized FP8/BF16 path and label that
  result separately.

## realtime-vla

Install realtime-vla following its official README. The wrapper can read a
converted `.pt` / `.pth` checkpoint directly. If you pass a PI0.5 safetensors
checkpoint, also provide FlashRT so the wrapper can reuse FlashRT's PI0.5
conversion helper.

```bash
export REALTIME_VLA_ROOT=<REALTIME_VLA_REPO>
export FLASHRT_ROOT=<FLASHRT_REPO>
cd <PHYAI_REPO>
python -c "import sys; sys.path.insert(0, '$REALTIME_VLA_ROOT'); from pi05_infer import Pi05Inference; print(Pi05Inference)"
```

Run latency:

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
  --result-file results/realtime_vla_pi05.jsonl
```

Notes:

- The wrapper uses BF16 synthetic inputs.
- For `.pkl` / `.pickle` checkpoints, add `--trust-pickle-checkpoint` only when
  the file is trusted.
- If the checkpoint does not contain `language_embeds`, the wrapper creates a
  synthetic prompt embedding for latency-only runs.

## vla.cpp

vla.cpp uses a server/client flow. Build `vla-server` with CUDA enabled, then
start the server in one shell and run the Python benchmark client in another.

Basic checks:

```bash
test -x <VLA_SERVER>
test -f <PI05_GGUF>
test -f <MM_PROJ_GGUF>
test -f <TOKENIZER_DIR>/tokenizer.json
test -f <LIBERO_STATS_JSON>
```

Start server:

```bash
<VLA_SERVER> \
  --bind tcp://127.0.0.1:5555 \
  --timing-detail phase \
  <MM_PROJ_GGUF> \
  <PI05_GGUF>
```

Run latency client:

```bash
cd <PHYAI_REPO>
python benchmark/pi05/bench_vlacpp_pi05_client.py \
  --vlacpp-root <VLACPP_REPO> \
  --addr tcp://127.0.0.1:5555 \
  --arch pi05 \
  --tokenizer <TOKENIZER_DIR> \
  --stats-json <LIBERO_STATS_JSON> \
  --num-views 2 \
  --chunk-size 50 \
  --batch-sizes 1 \
  --n-warmup 100 \
  --n-timed 100 \
  --result-file results/vlacpp_pi05.jsonl
```

Notes:

- vla.cpp needs GGUF files; a safetensors checkpoint is not enough.
- Prefer local tokenizer and stats files to avoid network or HuggingFace auth
  issues during benchmarking. For PI0.5, vla.cpp expects a lerobot-style
  `meta/stats.json`; OpenPI-style `norm_stats.json` is not the same format.
- The wrapper records client wall latency. If the server returns phase timing,
  it is written under `extras.server_phase_latency_ms`.

## Quick validation

After each runtime is installed, reduce iterations to check that the wrapper can
start and write JSONL:

```bash
--n-warmup 1 --n-timed 1 --result-file results/pi05_smoke.jsonl
```

For vla.cpp, keep `vla-server` running before starting the client smoke test.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named flash_rt` | Pass `--flashrt-root` or set `FLASHRT_ROOT`. |
| `No module named pi05_infer` | Pass `--realtime-vla-root` or set `REALTIME_VLA_ROOT`. |
| realtime-vla safetensors conversion fails | Also pass `--flashrt-root`; verify FlashRT import works. |
| vla.cpp client cannot connect | Confirm the server printed that it is ready and the `--addr` matches `--bind`. |
| vla.cpp tokenizer downloads or asks for auth | Use a local tokenizer directory. |
| GPU architecture build error | Check CUDA, PyTorch CUDA, driver, and build flags for the target GPU. |
| Latency is much slower than expected | Check `nvidia-smi`, rerun after warmup/JIT, and make sure no other process is using the GPU. |

## Timing scope

- FlashRT: wall time around steady-state `Pi05TorchFrontendRtx.infer(obs)`,
  after prompt setup, calibration, and first graph-building call.
- realtime-vla: CUDA-event time around one `Pi05Inference.forward(...)` call.
- vla.cpp: client wall time for one ZMQ request; server phase timing is copied
  from the response when available.
