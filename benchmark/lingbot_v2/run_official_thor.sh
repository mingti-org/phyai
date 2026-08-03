#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CHECKPOINT=""
PROCESSOR=""
OFFICIAL_REPO=""
INPUT=""
OUTPUT=""
GPU=0
N_WARMUP=10
N_TIMED=50
SKIP_CHECK=0

usage() {
  cat <<'EOF'
Usage:
  benchmark/lingbot_v2/run_official_thor.sh \
    --checkpoint DIR --processor DIR --official-repo DIR --input FILE [options]

Required:
  --checkpoint DIR       LingBot V2 checkpoint directory
  --processor DIR        Qwen3-VL processor/config directory
  --official-repo DIR    source prepared by prepare_official_thor.py
  --input FILE           fixed model-ready safetensors input

Options:
  --gpu N                CUDA_VISIBLE_DEVICES index (default: 0)
  --n-warmup N           warmup forwards (default: 10)
  --n-timed N            CUDA-event timed forwards (default: 50)
  --output FILE          output JSON
  --skip-check           bypass idle GPU/CPU checks
  -h, --help             show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --processor) PROCESSOR="$2"; shift 2 ;;
    --official-repo) OFFICIAL_REPO="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --n-warmup) N_WARMUP="$2"; shift 2 ;;
    --n-timed) N_TIMED="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

cd "${REPO_ROOT}"
for directory in "${CHECKPOINT}" "${PROCESSOR}" "${OFFICIAL_REPO}"; do
  [[ -d "${directory}" ]] || {
    echo "Required directory does not exist: ${directory}" >&2
    exit 2
  }
done
[[ -f "${INPUT}" ]] || {
  echo "Fixed input artifact does not exist: ${INPUT}" >&2
  exit 2
}
if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${SCRIPT_DIR}/lingbot_v2_official_thor_bf16.json"
fi

export PYTHONUNBUFFERED=1
export MAX_JOBS="${MAX_JOBS:-1}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"

echo "============================================================"
echo " Official LingBot V2 hook-free Thor benchmark"
echo " checkpoint   : ${CHECKPOINT}"
echo " processor    : ${PROCESSOR}"
echo " official repo: ${OFFICIAL_REPO}"
echo " input        : ${INPUT}"
echo " input SHA256 : $(sha256sum "${INPUT}" | awk '{print $1}')"
echo " GPU          : ${GPU}"
echo " warmup/timed : ${N_WARMUP}/${N_TIMED}"
echo " attention    : official eager, Thor compatibility"
echo " MoE          : official Robby Triton, strict"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

if [[ "${SKIP_CHECK}" -eq 0 ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    UTIL="$(nvidia-smi --id="${GPU}" \
      --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')"
    if [[ -n "${UTIL}" && "${UTIL}" -gt 10 ]]; then
      echo "Target GPU is busy (${UTIL}% utilization); benchmark aborted." >&2
      exit 1
    fi
  fi

  LOAD1="$(awk '{print $1}' /proc/loadavg)"
  NCPU="$(getconf _NPROCESSORS_ONLN)"
  LOAD_RATIO="$(awk -v load="${LOAD1}" -v ncpu="${NCPU}" \
    'BEGIN { if (ncpu > 0) printf "%.3f", load / ncpu; else print "1.0" }')"
  echo "CPU load check: load1=${LOAD1}, logical_cpus=${NCPU}, ratio=${LOAD_RATIO}"
  if awk -v ratio="${LOAD_RATIO}" 'BEGIN { exit !(ratio > 0.75) }'; then
    echo "CPU appears busy; benchmark aborted. Use --skip-check only if justified." >&2
    exit 1
  fi
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")
x = torch.ones(16, device="cuda")
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("CUDA tensor test:", float(x.sum()))
PY

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
  benchmark/lingbot_v2/official_latency_lingbot_v2.py \
  --checkpoint "${CHECKPOINT}" \
  --processor "${PROCESSOR}" \
  --official-repo "${OFFICIAL_REPO}" \
  --input "${INPUT}" \
  --device cuda:0 \
  --n-warmup "${N_WARMUP}" \
  --n-timed "${N_TIMED}" \
  --output "${OUTPUT}"

echo "Official benchmark complete."
echo "JSON: ${OUTPUT}"
