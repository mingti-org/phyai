#!/usr/bin/env bash
# Run the full pi0 profiling pipeline: probe the card + measure roofline, sweep
# batch sizes, then render figures. Outputs are gitignored.
#
# Usage:
#   ./run.sh --checkpoint /path/to/pi0_pytorch [options]
#
# Options:
#   -c, --checkpoint DIR    pi0 checkpoint folder (optional; omit for random weights)
#   -g, --gpu N             GPU to pin via CUDA_VISIBLE_DEVICES (default 0)
#   -b, --batch-sizes "..." space-separated batch list (default "1 2 4 8")
#   -l, --lang-len N        prompt token count (default 1)
#       --num-images N      override camera count, 2 or 3 (default checkpoint config)
#       --vision-dtype DT   float32 or bfloat16 (default float32)
#       --fixed-noise       use a seed-0 initial action noise tensor
#   -o, --out FILE          profile JSON path (default pi0_profile_<gpu>.json)
#   -f, --fig-dir DIR       figure output dir (default figures/)
#       --n-warmup N        warmup steps   (default 10)
#       --n-timed N         timed steps    (default 50)
#       --workspace-mib N   flashinfer workspace MiB (default 512)
#       --no-roofline       skip peak/bandwidth microbench
#       --skip-check        don't abort when target GPU looks busy
#       --plot-only         render figures from an existing JSON
#   -h, --help              show this help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT="${CKPT:-}"
GPU=0
BATCH_SIZES="1 2 4 8"
LANG_LEN=1
NUM_IMAGES=""
VISION_DTYPE="float32"
FIXED_NOISE=0
OUT=""
FIG_DIR="${SCRIPT_DIR}/figures"
N_WARMUP=10
N_TIMED=50
WORKSPACE_MIB=512
NO_ROOFLINE=0
SKIP_CHECK=0
PLOT_ONLY=0

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--checkpoint) CKPT="$2"; shift 2 ;;
    -g|--gpu) GPU="$2"; shift 2 ;;
    -b|--batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    -l|--lang-len) LANG_LEN="$2"; shift 2 ;;
    --num-images) NUM_IMAGES="$2"; shift 2 ;;
    --vision-dtype) VISION_DTYPE="$2"; shift 2 ;;
    --fixed-noise) FIXED_NOISE=1; shift ;;
    -o|--out) OUT="$2"; shift 2 ;;
    -f|--fig-dir) FIG_DIR="$2"; shift 2 ;;
    --n-warmup) N_WARMUP="$2"; shift 2 ;;
    --n-timed) N_TIMED="$2"; shift 2 ;;
    --workspace-mib) WORKSPACE_MIB="$2"; shift 2 ;;
    --no-roofline) NO_ROOFLINE=1; shift ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    --plot-only) PLOT_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

cd "${REPO_ROOT}"

if [[ "${PLOT_ONLY}" -eq 1 ]]; then
  [[ -n "${OUT}" ]] || { echo "--plot-only needs --out <existing JSON>" >&2; exit 2; }
  [[ -f "${OUT}" ]] || { echo "JSON not found: ${OUT}" >&2; exit 1; }
  echo "[plot-only] rendering figures from ${OUT} -> ${FIG_DIR}"
  uv run --with matplotlib python benchmark/pi0/plot_pi0.py \
    --in "${OUT}" --out-dir "${FIG_DIR}"
  exit 0
fi

if [[ -n "${CKPT}" && ! -d "${CKPT}" ]]; then
  echo "ERROR: --checkpoint must be an existing directory (got: '${CKPT}')." >&2
  exit 1
fi

echo "============================================================"
echo " pi0 profiling pipeline"
echo "   checkpoint : ${CKPT:-<random weights>}"
echo "   GPU        : ${GPU}   (CUDA_VISIBLE_DEVICES)"
echo "   batch sizes: ${BATCH_SIZES}"
echo "   lang_len   : ${LANG_LEN}    warmup=${N_WARMUP} timed=${N_TIMED}"
echo "   vision     : ${VISION_DTYPE}   workspace=${WORKSPACE_MIB} MiB"
echo "============================================================"

if [[ "${SKIP_CHECK}" -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  N_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
  [[ "${N_GPUS}" -eq 1 ]] && GPU_WORD="GPU" || GPU_WORD="GPUs"
  read -r USED UTIL < <(nvidia-smi --id="${GPU}" \
    --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ',' | awk '{print $1, $2}')
  [[ "${USED:-}" =~ ^[0-9]+$ ]] || USED=""
  [[ "${UTIL:-}" =~ ^[0-9]+$ ]] || UTIL=""
  echo "[check] GPU ${GPU}: ${USED:-?} MiB used, ${UTIL:-?}% util (${N_GPUS} ${GPU_WORD} visible)"
  if [[ "${N_GPUS}" -le 1 ]]; then
    if [[ "${UTIL:-0}" -gt 10 ]]; then
      echo "ERROR: the only GPU looks busy (${UTIL}% util). Pass --skip-check to override." >&2
      exit 1
    fi
  elif [[ "${USED:-0}" -gt 1024 || "${UTIL:-0}" -gt 10 ]]; then
    echo "ERROR: GPU ${GPU} looks busy (${USED} MiB / ${UTIL}%). Pass --skip-check to override." >&2
    exit 1
  fi
fi

if [[ -z "${OUT}" ]]; then
  GPU_SLUG="$(CUDA_VISIBLE_DEVICES="${GPU}" uv run python -c \
    'import torch,re;print(re.sub(r"[^a-z0-9]+","_",torch.cuda.get_device_name(0).lower()).strip("_"))' \
    2>/dev/null || echo gpu)"
  OUT="${SCRIPT_DIR}/pi0_profile_${GPU_SLUG}.json"
fi
echo "[out] profile JSON -> ${OUT}"

EXTRA=()
[[ -n "${CKPT}" ]] && EXTRA+=(--checkpoint "${CKPT}")
[[ -n "${NUM_IMAGES}" ]] && EXTRA+=(--num-images "${NUM_IMAGES}")
[[ "${FIXED_NOISE}" -eq 1 ]] && EXTRA+=(--fixed-noise)
[[ "${NO_ROOFLINE}" -eq 1 ]] && EXTRA+=(--no-roofline)

CUDA_VISIBLE_DEVICES="${GPU}" uv run python benchmark/pi0/profile_pi0.py \
  "${EXTRA[@]}" \
  --batch-sizes ${BATCH_SIZES} \
  --lang-len "${LANG_LEN}" \
  --vision-dtype "${VISION_DTYPE}" \
  --workspace-bytes "$((WORKSPACE_MIB * 1024 * 1024))" \
  --n-warmup "${N_WARMUP}" \
  --n-timed "${N_TIMED}" \
  --out "${OUT}"

echo "[plot] rendering figures -> ${FIG_DIR}"
uv run --with matplotlib python benchmark/pi0/plot_pi0.py \
  --in "${OUT}" --out-dir "${FIG_DIR}"

echo ""
echo "done."
echo "  JSON    : ${OUT}"
echo "  figures : ${FIG_DIR}/fig1..fig7 .svg"
