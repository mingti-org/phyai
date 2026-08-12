#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CHECKPOINT="${CHECKPOINT:-}"
INPUT="${LINGBOT_BENCH_INPUT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU=0
VISION_DTYPE="bf16"
# Benchmark PHYAI's optimized production path by default.
PATCH_EMBED_BACKEND="gemm"
LINEAR_KERNEL="torch"
N_WARMUP=10
N_TIMED=50
N_PROF_STEPS=5
NO_ROOFLINE=0
USE_CUDA_GRAPH=1
SKIP_CHECK=0
SKIP_PLOT=0
PLOT_ONLY=0
OUT=""
FIG_DIR="${SCRIPT_DIR}/figures"

usage() {
  cat <<'EOF'
Usage:
  benchmark/lingbot_v2/run.sh --checkpoint DIR --input FILE [options]

Required:
  -c, --checkpoint DIR       LingBot V2 checkpoint directory
  -i, --input FILE           Fixed model-ready safetensors input artifact

Options:
  -g, --gpu N                CUDA_VISIBLE_DEVICES index (default: 0)
      --vision-dtype DTYPE   bf16 or fp32 (default: bf16)
      --patch-embed-backend  conv3d or gemm (default: gemm)
      --linear-kernel NAME   torch or flashinfer (default: torch)
      --n-warmup N           warmup engine steps (default: 10)
      --n-timed N            CUDA-event timed steps (default: 50)
      --n-prof-steps N       CUDA-event component passes (default: 5)
      --use-cuda-graph       capture/replay the fixed Expert Euler loop (default)
      --no-cuda-graph        disable CUDA Graph for diagnostics
      --no-roofline          skip measured BF16 peak and bandwidth
      --skip-check           bypass idle GPU/CPU checks
      --skip-plot            do not render figures after profiling
      --plot-only            render figures from --out without a GPU run
  -o, --out FILE             output JSON path
  -f, --fig-dir DIR          output figure directory
  -h, --help                 show this help

The benchmark uses batch=1, three 256x256 views, BF16 ViT, chunk=50,
ten Euler steps, Torch Linear, and torch.compile off. GEMM is PHYAI's default
optimized PatchEmbed path; select Conv3D explicitly for operator parity checks.
CUDA Graph is enabled by default and can be disabled for diagnostics.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--checkpoint) CHECKPOINT="$2"; shift 2 ;;
    -i|--input) INPUT="$2"; shift 2 ;;
    -g|--gpu) GPU="$2"; shift 2 ;;
    --vision-dtype) VISION_DTYPE="$2"; shift 2 ;;
    --patch-embed-backend) PATCH_EMBED_BACKEND="$2"; shift 2 ;;
    --linear-kernel) LINEAR_KERNEL="$2"; shift 2 ;;
    --n-warmup) N_WARMUP="$2"; shift 2 ;;
    --n-timed) N_TIMED="$2"; shift 2 ;;
    --n-prof-steps) N_PROF_STEPS="$2"; shift 2 ;;
    --use-cuda-graph) USE_CUDA_GRAPH=1; shift ;;
    --no-cuda-graph) USE_CUDA_GRAPH=0; shift ;;
    --no-roofline) NO_ROOFLINE=1; shift ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    --skip-plot) SKIP_PLOT=1; shift ;;
    --plot-only) PLOT_ONLY=1; shift ;;
    -o|--out) OUT="$2"; shift 2 ;;
    -f|--fig-dir) FIG_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/phyai/src:${REPO_ROOT}/phyai-kernel:${REPO_ROOT}/phyai-utils-tools/src:${REPO_ROOT}/phyai-ext/src:${REPO_ROOT}/phyai-model-optimizer/src:${PYTHONPATH:-}"
export MAX_JOBS="${MAX_JOBS:-1}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
export PYTHONUNBUFFERED=1

if [[ "${PLOT_ONLY}" -eq 1 ]]; then
  [[ -n "${OUT}" && -f "${OUT}" ]] || {
    echo "--plot-only requires --out pointing to an existing JSON file." >&2
    exit 2
  }
  "${PYTHON_BIN}" benchmark/lingbot_v2/plot_lingbot_v2.py \
    --in "${OUT}" \
    --out-dir "${FIG_DIR}"
  exit 0
fi

[[ -d "${CHECKPOINT}" ]] || {
  echo "Checkpoint directory does not exist: ${CHECKPOINT}" >&2
  exit 2
}
[[ -f "${INPUT}" ]] || {
  echo "Fixed input artifact does not exist: ${INPUT}" >&2
  exit 2
}
[[ "${VISION_DTYPE}" == "bf16" || "${VISION_DTYPE}" == "fp32" ]] || {
  echo "--vision-dtype must be bf16 or fp32." >&2
  exit 2
}
[[ "${PATCH_EMBED_BACKEND}" == "conv3d" || "${PATCH_EMBED_BACKEND}" == "gemm" ]] || {
  echo "--patch-embed-backend must be conv3d or gemm." >&2
  exit 2
}
[[ "${LINEAR_KERNEL}" == "torch" || "${LINEAR_KERNEL}" == "flashinfer" ]] || {
  echo "--linear-kernel must be torch or flashinfer." >&2
  exit 2
}

echo "============================================================"
echo " LingBot V2 fixed-contract benchmark"
echo " checkpoint    : ${CHECKPOINT}"
echo " input         : ${INPUT}"
echo " input SHA256  : $(sha256sum "${INPUT}" | awk '{print $1}')"
echo " GPU           : ${GPU}"
echo " vision dtype  : ${VISION_DTYPE}"
echo " PatchEmbed    : ${PATCH_EMBED_BACKEND}"
echo " linear kernel : ${LINEAR_KERNEL}"
echo " warmup/timed  : ${N_WARMUP}/${N_TIMED}"
echo " profile steps : ${N_PROF_STEPS}"
echo " CUDA Graph    : ${USE_CUDA_GRAPH}"
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

if [[ -z "${OUT}" ]]; then
  GPU_SLUG="$(CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -c \
    'import re,torch; print(re.sub(r"[^a-z0-9]+","_",torch.cuda.get_device_name(0).lower()).strip("_"))')"
  if [[ "${USE_CUDA_GRAPH}" -eq 1 ]]; then
    GRAPH_SLUG="graph_on"
  else
    GRAPH_SLUG="graph_off"
  fi
  OUT="${SCRIPT_DIR}/lingbot_v2_profile_${GPU_SLUG}_${VISION_DTYPE}_${PATCH_EMBED_BACKEND}_${GRAPH_SLUG}.json"
fi

ROOFLINE_ARGS=()
if [[ "${NO_ROOFLINE}" -eq 1 ]]; then
  ROOFLINE_ARGS+=(--no-roofline)
fi

CUDA_GRAPH_ARGS=()
if [[ "${USE_CUDA_GRAPH}" -eq 1 ]]; then
  CUDA_GRAPH_ARGS+=(--use-cuda-graph)
else
  CUDA_GRAPH_ARGS+=(--no-cuda-graph)
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
  benchmark/lingbot_v2/profile_lingbot_v2.py \
  --checkpoint "${CHECKPOINT}" \
  --input "${INPUT}" \
  --device cuda:0 \
  --vision-dtype "${VISION_DTYPE}" \
  --patch-embed-backend "${PATCH_EMBED_BACKEND}" \
  --linear-kernel "${LINEAR_KERNEL}" \
  --n-warmup "${N_WARMUP}" \
  --n-timed "${N_TIMED}" \
  --n-prof-steps "${N_PROF_STEPS}" \
  --trace-dir "${SCRIPT_DIR}/traces" \
  --out "${OUT}" \
  "${CUDA_GRAPH_ARGS[@]}" \
  "${ROOFLINE_ARGS[@]}"

if [[ "${SKIP_PLOT}" -eq 0 ]]; then
  if "${PYTHON_BIN}" -c "import matplotlib" >/dev/null 2>&1; then
    "${PYTHON_BIN}" benchmark/lingbot_v2/plot_lingbot_v2.py \
      --in "${OUT}" \
      --out-dir "${FIG_DIR}"
  else
    echo "matplotlib is unavailable; JSON is complete and plotting was skipped."
    echo "Render later with: $0 --plot-only --out ${OUT}"
  fi
fi

echo "Benchmark complete."
echo "JSON: ${OUT}"
echo "Figures: ${FIG_DIR}"
