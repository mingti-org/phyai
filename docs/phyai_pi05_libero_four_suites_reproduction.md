---
title: PhyAI pi0.5 LIBERO four-suite reproduction
description: Run the pi0.5 LIBERO policy with PhyAI on all four LIBERO benchmark suites.
---

# PhyAI pi0.5 LIBERO four-suite reproduction

This guide shows how to run the pi0.5 LIBERO policy with PhyAI and evaluate it with `vla-evaluation-harness` on all four LIBERO suites.
It is written for a fresh machine and avoids local machine-specific paths by using environment variables.

The four suites are:

```text
libero_spatial -> configs/benchmarks/libero/spatial.yaml
libero_object  -> configs/benchmarks/libero/object.yaml
libero_goal    -> configs/benchmarks/libero/goal.yaml
libero_10      -> configs/benchmarks/libero/10.yaml
```

The benchmark setup in this guide uses:

```text
Mode: sync
Chunk size: 10
Episodes per suite: 10 tasks x 50 episodes = 500 episodes
Total episodes: 4 suites x 500 episodes = 2000 episodes
Model: PhyAI pi0.5 LIBERO converted checkpoint
Simulator: vla-evaluation-harness LIBERO Docker container
Output: one JSON result file per suite with success, steps, timing, and chunk-size fields
```

## 1. Prerequisites

Use a Linux machine with:

```text
GPU: at least 1 CUDA GPU, 48 GB or more GPU memory recommended
Container runtime: Docker and NVIDIA Container Toolkit
Python environment manager: uv
Utility tools: tmux, nvidia-smi, ss
```

Prepare these model resources before you start:

```text
PhyAI converted checkpoint: pi05_libero_phyai_converted
PaLI-Gemma tokenizer / processor: paligemma-3b-pt-224
```

`paligemma-3b-pt-224` is a gated resource. Prefer syncing it from a machine that already has access instead of downloading it during reproduction.

## 2. Set environment variables

Set paths for the target machine:

```bash
export PHYAI_ROOT=$HOME/phyai
export VLA_ROOT=$HOME/vla-evaluation-harness
export MODEL_ROOT=$HOME/phyai_models
export PHYAI_CONTAINER=phyai_libero_eval

export PHYAI_CKPT_HOST=$MODEL_ROOT/pi05_libero_phyai_converted
export TOKENIZER_HOST=$MODEL_ROOT/paligemma-3b-pt-224

export PHYAI_CKPT_IN_CONTAINER=/data/share/pi05_libero_phyai_converted
export TOKENIZER_IN_CONTAINER=/data/share/paligemma-3b-pt-224

export LIBERO_IMAGE=ghcr.io/allenai/vla-evaluation-harness/libero:latest
```

Check that the model directories exist:

```bash
test -d "$PHYAI_CKPT_HOST"
test -d "$TOKENIZER_HOST"
```

If the models are on another machine, sync them to the target machine:

```bash
rsync -azP \
  /path/to/pi05_libero_phyai_converted \
  /path/to/paligemma-3b-pt-224 \
  user@target-host:$MODEL_ROOT/
```

## 3. Clone source code and create environments

Clone PhyAI:

```bash
git clone https://github.com/MEmbodied/phyai.git "$PHYAI_ROOT"
cd "$PHYAI_ROOT"
uv sync
```

Clone `vla-evaluation-harness`:

```bash
git clone https://github.com/allenai/vla-evaluation-harness.git "$VLA_ROOT"
cd "$VLA_ROOT"
uv sync
./.venv/bin/vla-eval --help >/tmp/vla_eval_help.log
```

If the target machine has no network access, clone both repositories on a networked machine and sync them with `rsync`.
After syncing, still run `uv sync` on the target machine so editable paths, Python versions, CUDA libraries, and local dependencies are resolved correctly.

## 4. Prepare the LIBERO Docker image

`vla-evaluation-harness` runs LIBERO inside a benchmark container.
On an `x86_64` machine, pull the official image:

```bash
docker pull "$LIBERO_IMAGE"
```

If the target machine is ARM64 and the official image is only available for `amd64`, build an ARM64 LIBERO image locally:

```bash
cd "$VLA_ROOT"
export DOCKER_DEFAULT_PLATFORM=linux/arm64
docker/build.sh libero

docker image inspect "$LIBERO_IMAGE" \
  --format '{{.Architecture}} {{.Os}}'
```

Expected output on ARM64:

```text
arm64 linux
```

Expected output on `x86_64`:

```text
amd64 linux
```

## 5. Create the PhyAI Docker container

Run the PhyAI server inside a Docker container.
Mount the PhyAI source tree, the `vla-evaluation-harness` source tree, and the model directory:

```bash
docker run -dit --gpus all \
  -v "$PHYAI_ROOT":/phyai_workspace \
  -v "$VLA_ROOT":/vla-evaluation-harness \
  -v "$MODEL_ROOT":/data/share \
  -w /phyai_workspace \
  --cap-add=SYS_ADMIN \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --shm-size=4G \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --name "$PHYAI_CONTAINER" \
  nvcr.io/nvidia/pytorch:25.12-py3 bash
```

Install the PhyAI environment inside the container:

```bash
docker exec "$PHYAI_CONTAINER" bash -lc '
cd /phyai_workspace
python3 -m pip install -U uv
uv sync
'
```

If `uv sync` produces editable paths that point to the host path instead of the container path, create a compatibility symlink.
Only run this if imports fail because a stale host path is referenced:

```bash
export HOST_USER=$(id -un)
export COMPAT_PARENT=/compat_mount

docker exec "$PHYAI_CONTAINER" bash -lc "
mkdir -p $COMPAT_PARENT/$HOST_USER
ln -sfn /phyai_workspace $COMPAT_PARENT/$HOST_USER/phyai
"
```

Verify imports inside the container:

```bash
docker exec "$PHYAI_CONTAINER" bash -lc '
cd /phyai_workspace
export PYTHONPATH=/phyai_workspace/phyai/src:/phyai_workspace/phyai-kernel:/phyai_workspace/phyai-utils-tools/src:/vla-evaluation-harness/src
/phyai_workspace/.venv/bin/python - <<PY
import phyai
import vla_eval
print("phyai", phyai.__file__)
print("vla_eval", vla_eval.__file__)
PY
'
```

## 6. Check the machine before running

Check GPU usage, ports, and containers before starting the benchmark:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
ss -ltnp | grep -E ':8000|:8001' || true
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

If the GPU is under heavy load, benchmark timing can become unstable.
Use an idle GPU or wait for other jobs to finish.

## 7. Start the PhyAI pi0.5 server

Get the PhyAI container IP and construct the WebSocket URL:

```bash
export PHYAI_CONTAINER_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$PHYAI_CONTAINER")
export PHYAI_SERVER_URL=ws://$PHYAI_CONTAINER_IP:8000
echo "$PHYAI_SERVER_URL"
```

Start the server in `tmux`:

```bash
mkdir -p "$VLA_ROOT/results"

tmux new-session -d -s phyai_pi05_libero_server "
docker exec $PHYAI_CONTAINER bash -lc '
cd /vla-evaluation-harness
export PYTHONPATH=/phyai_workspace/phyai/src:/phyai_workspace/phyai-kernel:/phyai_workspace/phyai-utils-tools/src:/vla-evaluation-harness/src
export PHYAI_TOKENIZER_PATH=$TOKENIZER_IN_CONTAINER
export PHYAI_CAMERA_MODE=two_camera
/phyai_workspace/.venv/bin/python -m vla_eval.model_servers.phyai \
  --checkpoint_path $PHYAI_CKPT_IN_CONTAINER \
  --device cuda:0 \
  --params_dtype bfloat16 \
  --attn_backend flashinfer \
  --norm_backend phyai-kernel \
  --linear_backend flashinfer \
  --flashinfer_workspace_bytes 536870912 \
  --chunk_size 10 \
  --host 0.0.0.0 \
  --port 8000
' 2>&1 | tee $VLA_ROOT/results/phyai_pi05_libero_server.log
"
```

Key settings:

| Setting | Value | Purpose |
| --- | --- | --- |
| `--checkpoint_path` | `/data/share/pi05_libero_phyai_converted` | PhyAI converted pi0.5 LIBERO checkpoint |
| `PHYAI_TOKENIZER_PATH` | `/data/share/paligemma-3b-pt-224` | Tokenizer and processor directory |
| `PHYAI_CAMERA_MODE` | `two_camera` | LIBERO sends both agent-view and wrist-camera images |
| `--params_dtype` | `bfloat16` | Parameter dtype |
| `--attn_backend` | `flashinfer` | Attention backend |
| `--norm_backend` | `phyai-kernel` | Normalization backend |
| `--linear_backend` | `flashinfer` | Linear backend |
| `--flashinfer_workspace_bytes` | `536870912` | 512 MiB FlashInfer workspace |
| `--chunk_size` | `10` | The policy returns 10 actions per inference call |
| CUDA graph | Enabled by default | Do not pass `--no-use_cuda_graph` |

Follow the server log:

```bash
tail -f "$VLA_ROOT/results/phyai_pi05_libero_server.log"
```

Wait until the log contains:

```text
capturing vision-tower CUDA graph
capturing 4 prefix-forward CUDA graph(s)
capturing the full 10-step Euler loop as one CUDA graph
Starting server on ws://0.0.0.0:8000
```

## 8. Run a smoke test

Run a minimal smoke test before launching the full benchmark.
This checks the model, LIBERO Docker image, WebSocket connection, timing fields, and chunk-size fields.

```bash
cat > "$VLA_ROOT/configs/benchmarks/libero/smoke_test_phyai_local.yaml" <<YAML
server:
  url: "$PHYAI_SERVER_URL"

docker:
  image: $LIBERO_IMAGE

output_dir: "$VLA_ROOT/results/phyai_pi05_libero_smoke"

benchmarks:
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
    name: "phyai_pi05_libero_smoke"
    episodes_per_task: 1
    max_tasks: 1
    mode: "sync"
    params:
      suite: libero_spatial
      seed: 7
      num_steps_wait: 10
YAML

cd "$VLA_ROOT"
env NO_PROXY='*' no_proxy='*' ./.venv/bin/vla-eval run \
  --config configs/benchmarks/libero/smoke_test_phyai_local.yaml \
  --server-url "$PHYAI_SERVER_URL" \
  --dev \
  --yes
```

Summarize the smoke-test timing:

```bash
cd "$VLA_ROOT"
json=$(ls -t results/phyai_pi05_libero_smoke/*.json | head -1)
./.venv/bin/python scripts/summarize_timing.py "$json"
```

The output must include:

```text
raw_chunk_size_max=10 served_chunk_size_max=10
```

If timing or chunk fields are missing, the usual cause is running `vla-eval run` without `--dev`.
The `--dev` flag mounts the host `src` tree into the LIBERO container so the benchmark uses the local runner implementation.

## 9. Run all four LIBERO suites

Create a script that runs the four suites sequentially.
The script creates one runtime YAML file per suite and writes all outputs under a single run directory.

```bash
cat > "$VLA_ROOT/run_phyai_pi05_libero_four_suites.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

: "${PHYAI_SERVER_URL:?must set PHYAI_SERVER_URL}"
: "${PHYAI_CKPT_IN_CONTAINER:?must set PHYAI_CKPT_IN_CONTAINER}"
: "${LIBERO_IMAGE:=ghcr.io/allenai/vla-evaluation-harness/libero:latest}"

cd "$(dirname "$0")"

RUN_ID="phyai_pi05_libero_four_$(date +%Y%m%d_%H%M%S)"
OUT="results/${RUN_ID}"
mkdir -p "$OUT/configs"

{
  echo "RUN_ID=${RUN_ID}"
  echo "START=$(date -Is)"
  echo "MODEL=phyai_pi05"
  echo "SERVER_URL=${PHYAI_SERVER_URL}"
  echo "CHECKPOINT=${PHYAI_CKPT_IN_CONTAINER}"
  echo "PHYAI_CAMERA_MODE=two_camera"
  echo "MODE=sync"
  echo "CHUNK_SIZE=10"
  echo "SERVER_CONFIG=use_cuda_graph=True attn=flashinfer norm=phyai-kernel linear=flashinfer workspace=536870912 params_dtype=bfloat16"
} | tee "$OUT/run_summary.log"

make_cfg() {
  local suite_name="$1"
  local suite="$2"
  local cfg="$3"
  cat > "$cfg" <<YAML
server:
  url: "${PHYAI_SERVER_URL}"

docker:
  image: ${LIBERO_IMAGE}

output_dir: "${OUT}"

benchmarks:
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
    subname: ${suite}
    episodes_per_task: 50
    mode: "sync"
    params:
      suite: ${suite}
      seed: 7
      num_steps_wait: 10
YAML
}

for item in \
  spatial:libero_spatial \
  object:libero_object \
  goal:libero_goal \
  libero10:libero_10
do
  name="${item%%:*}"
  suite="${item#*:}"
  cfg="$OUT/configs/${name}.yaml"
  log="$OUT/phyai_${name}.log"
  make_cfg "$name" "$suite" "$cfg"

  echo "SUITE_START model=phyai suite=${name} cfg=${cfg} server=${PHYAI_SERVER_URL} time=$(date -Is)" | tee -a "$OUT/run_summary.log"

  {
    echo "MODEL=phyai"
    echo "SUITE=${name}"
    echo "LIBERO_SUITE=${suite}"
    echo "CONFIG=${cfg}"
    echo "START_ISO=$(date -Is)"
    /usr/bin/time -p env NO_PROXY='*' no_proxy='*' ./.venv/bin/vla-eval run \
      --config "$cfg" \
      --server-url "$PHYAI_SERVER_URL" \
      --dev \
      --yes
    echo "END_ISO=$(date -Is)"
  } 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}

  echo "SUITE_END model=phyai suite=${name} status=${status} log=${log} time=$(date -Is)" | tee -a "$OUT/run_summary.log"

  result_json=$(ls -t "$OUT/${suite}_sync_"*.json "$OUT"/*"${suite}"*_sync_*.json 2>/dev/null | head -1 || true)
  if [ -n "$result_json" ]; then
    ./.venv/bin/python scripts/summarize_timing.py "$result_json" | sed "s/^/TIMING phyai_${name} /" | tee -a "$OUT/run_summary.log"
  else
    echo "WARN no result json found for ${name}" | tee -a "$OUT/run_summary.log"
  fi

  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done

echo "ALL_DONE $(date -Is)" | tee -a "$OUT/run_summary.log"
SH
chmod +x "$VLA_ROOT/run_phyai_pi05_libero_four_suites.sh"
```

Start the full run:

```bash
cd "$VLA_ROOT"
tmux new-session -d -s phyai_pi05_libero_four \
  "PHYAI_SERVER_URL=$PHYAI_SERVER_URL PHYAI_CKPT_IN_CONTAINER=$PHYAI_CKPT_IN_CONTAINER LIBERO_IMAGE=$LIBERO_IMAGE ./run_phyai_pi05_libero_four_suites.sh"
```

Check progress:

```bash
tmux ls
cd "$VLA_ROOT"
latest=$(ls -td results/phyai_pi05_libero_four_* | head -1)
sed -n '1,220p' "$latest/run_summary.log"
tail -80 "$latest"/phyai_spatial.log
```

## 10. Summarize success rate and timing

After the run finishes, print the run summary:

```bash
cd "$VLA_ROOT"
latest=$(ls -td results/phyai_pi05_libero_four_* | head -1)
cat "$latest/run_summary.log"
```

Summarize timing from all result JSON files:

```bash
cd "$VLA_ROOT"
latest=$(ls -td results/phyai_pi05_libero_four_* | head -1)
./.venv/bin/python scripts/summarize_timing.py "$latest"/*.json
```

Summarize success rate:

```bash
cd "$VLA_ROOT"
latest=$(ls -td results/phyai_pi05_libero_four_* | head -1)
./.venv/bin/python - <<'PY' "$latest"/*.json
import json
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    p = Path(arg)
    data = json.loads(p.read_text())
    eps = [ep for task in data.get("tasks", []) for ep in task.get("episodes", [])]
    succ = sum(1 for ep in eps if ep.get("metrics", {}).get("success"))
    total = len(eps)
    rate = succ / total * 100.0 if total else 0.0
    steps = sum(int(ep.get("steps", 0)) for ep in eps)
    print(f"{p.name}: success={succ}/{total} rate={rate:.1f}% steps={steps} mean_success={data.get('mean_success')}")
PY
```

Record these fields for each suite:

```text
RUN_ID
Result directory
Checkpoint
Server URL
Suite
Success / total
Success rate
Steps
/usr/bin/time -p real
model_wait_sec
model_inference_sec
env_step_sec
obs_sec
avg_model_inference_ms
model_inference_calls
model_buffer_hits
raw_chunk_size_max
served_chunk_size_max
```

`raw_chunk_size_max=10` and `served_chunk_size_max=10` are key checks for this four-suite reproduction.

## 11. Reference results

Timing depends on the GPU, driver, machine load, and container environment.
Success rate can also vary slightly across runs.
The following results are from a previous run with the same evaluation setup:

| Suite | Success rate | Success | Steps | Wall time | Model inference time | Env step time | Benchmark action-wait time | Average model inference | Inference calls | Buffer hits | Chunk check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `libero_spatial` | 97.8% | 489/500 | 52,926 | 4,198.73s | 200.56s | 1,239.61s | 2,269.16s | 36.33ms | 5,520 | 47,406 | raw=10, served=10 |
| `libero_object` | 99.8% | 499/500 | 68,712 | 5,199.23s | 256.92s | 1,169.84s | 3,453.50s | 36.18ms | 7,102 | 61,610 | raw=10, served=10 |
| `libero_goal` | 98.0% | 490/500 | 56,292 | 3,981.09s | 210.86s | 1,045.35s | 2,365.71s | 36.10ms | 5,841 | 50,451 | raw=10, served=10 |
| `libero_10` | 94.2% | 471/500 | 134,962 | 9,150.13s | 496.56s | 2,227.75s | 6,237.24s | 36.21ms | 13,713 | 121,249 | raw=10, served=10 |

Use these numbers as references, not strict pass/fail thresholds.
For reproduction, first confirm:

```text
All four suites finish 500 episodes each
Chunk check is raw=10 served=10
The server log confirms CUDA graph capture
The result JSON files contain timing fields
```

## 12. Troubleshooting

### 12.1 LIBERO Docker image architecture mismatch

If the target machine is ARM64 and the official image is only available for `amd64`, build the image locally:

```bash
cd "$VLA_ROOT"
export DOCKER_DEFAULT_PLATFORM=linux/arm64
docker/build.sh libero
```

### 12.2 The benchmark cannot connect to the PhyAI server

If the PhyAI server runs in a bridge Docker container, the host-side `vla-eval` process should connect to the container IP:

```bash
export PHYAI_CONTAINER_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$PHYAI_CONTAINER")
export PHYAI_SERVER_URL=ws://$PHYAI_CONTAINER_IP:8000
```

### 12.3 The result JSON has no timing fields

Run the benchmark with `--dev`.
This mounts the host `$VLA_ROOT/src` tree into the LIBERO container and ensures the benchmark uses the runner that records timing fields.

### 12.4 The PaLI-Gemma tokenizer is missing

`paligemma-3b-pt-224` is a gated resource.
Sync it from an existing machine to `$MODEL_ROOT/paligemma-3b-pt-224` instead of relying on an on-the-fly download.

### 12.5 The PhyAI server log has no CUDA graph capture messages

Check whether the server command accidentally passed `--no-use_cuda_graph` or whether it started the wrong server adapter.
The expected log must include:

```text
capturing vision-tower CUDA graph
capturing 4 prefix-forward CUDA graph(s)
capturing the full 10-step Euler loop as one CUDA graph
```

### 12.6 Release resources

```bash
tmux kill-session -t phyai_pi05_libero_four || true
tmux kill-session -t phyai_pi05_libero_server || true
docker stop "$PHYAI_CONTAINER" || true
ss -ltnp | grep ':8000' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```
