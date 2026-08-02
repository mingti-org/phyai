#!/usr/bin/env bash
# Run the full pi0.5 profiling pipeline: probe the card + measure its roofline,
# sweep batch sizes, then render the figures. Outputs (JSON, figures, traces)
# are regenerated here and are gitignored — only this script and the .py/.md
# files are committed.
#
# Usage:
#   ./run.sh --checkpoint /path/to/pi05_base [options]
#
# Options:
#   -c, --checkpoint DIR    pi05_base checkpoint folder (REQUIRED; or set $CKPT)
#   -g, --gpu N             GPU to pin via CUDA_VISIBLE_DEVICES (default 0)
#   -b, --batch-sizes "..." space-separated batch list (default "1 2 4 8 16 32")
#   -l, --lang-len N        prompt token count (default 1)
#   -o, --out FILE          profile JSON path (default pi05_profile_<gpu-slug>.json)
#   -f, --fig-dir DIR       figure output dir (default figures/)
#       --n-warmup N        warmup steps   (default 10)
#       --n-timed N         timed steps    (default 50)
#       --no-roofline       skip the peak/bandwidth microbench
#       --skip-check        don't abort when the target GPU looks busy
#       --plot-only         re-render figures from an existing JSON (no GPU run)
#   -h, --help              show this help
#
# Examples:
#   ./run.sh -c $CKPT -g 7
#   ./run.sh -c $CKPT -g 7 -b "1 4 16" --n-timed 30
#   ./run.sh --plot-only -o pi05_profile_nvidia_thor.json
set -euo pipefail

# --- Repo paths. This script lives in benchmark/pi05; repo root is 2 up. ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Defaults ---
CKPT="${CKPT:-}"
GPU=0
BATCH_SIZES="1 2 4 8 16 32"
LANG_LEN=1
OUT=""
FIG_DIR="${SCRIPT_DIR}/figures"
N_WARMUP=10
N_TIMED=50
NO_ROOFLINE=0
SKIP_CHECK=0
PLOT_ONLY=0

usage() { sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--checkpoint) CKPT="$2"; shift 2 ;;
    -g|--gpu)        GPU="$2"; shift 2 ;;
    -b|--batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    -l|--lang-len)   LANG_LEN="$2"; shift 2 ;;
    -o|--out)        OUT="$2"; shift 2 ;;
    -f|--fig-dir)    FIG_DIR="$2"; shift 2 ;;
    --n-warmup)      N_WARMUP="$2"; shift 2 ;;
    --n-timed)       N_TIMED="$2"; shift 2 ;;
    --no-roofline)   NO_ROOFLINE=1; shift ;;
    --skip-check)    SKIP_CHECK=1; shift ;;
    --plot-only)     PLOT_ONLY=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Plot-only fast path: re-render figures from an existing JSON, no GPU needed.
# ---------------------------------------------------------------------------
if [[ "${PLOT_ONLY}" -eq 1 ]]; then
  [[ -n "${OUT}" ]] || { echo "--plot-only needs --out <existing JSON>" >&2; exit 2; }
  [[ -f "${OUT}" ]] || { echo "JSON not found: ${OUT}" >&2; exit 1; }
  echo "[plot-only] rendering figures from ${OUT} -> ${FIG_DIR}"
  uv run --with matplotlib python benchmark/pi05/plot_pi05.py \
    --in "${OUT}" --out-dir "${FIG_DIR}"
  exit 0
fi

# --- Validate checkpoint ---
if [[ -z "${CKPT}" || ! -d "${CKPT}" ]]; then
  echo "ERROR: --checkpoint must be an existing directory (got: '${CKPT}')." >&2
  echo "       Pass -c /path/to/pi05_base or export CKPT=..." >&2
  exit 1
fi

echo "============================================================"
echo " pi0.5 profiling pipeline"
echo "   checkpoint : ${CKPT}"
echo "   GPU        : ${GPU}   (CUDA_VISIBLE_DEVICES)"
echo "   batch sizes: ${BATCH_SIZES}"
echo "   lang_len   : ${LANG_LEN}    warmup=${N_WARMUP} timed=${N_TIMED}"
echo "   LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "============================================================"

# --- GPU busy check. A contended card skews the roofline microbench and the
# latencies, so refuse by default (override with --skip-check).
#
# Single-card boxes need different handling: there is no "other GPU" to move to,
# and integrated / unified-memory cards (e.g. Jetson Thor) report the whole
# system's memory as "used" and may report utilization as [N/A]. So on one card
# we drop the memory gate entirely and judge contention by live utilization
# alone; on a multi-card box a loaded target just means "pick another one". ---
if [[ "${SKIP_CHECK}" -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  N_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
  [[ "${N_GPUS}" -eq 1 ]] && GPU_WORD="GPU" || GPU_WORD="GPUs"
  read -r USED UTIL < <(nvidia-smi --id="${GPU}" \
    --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ',' | awk '{print $1, $2}')
  # Non-numeric readings (e.g. utilization "[N/A]" on integrated GPUs) -> unknown.
  [[ "${USED:-}" =~ ^[0-9]+$ ]] || USED=""
  [[ "${UTIL:-}" =~ ^[0-9]+$ ]] || UTIL=""
  echo "[check] GPU ${GPU}: ${USED:-?} MiB used, ${UTIL:-?}% util (${N_GPUS} ${GPU_WORD} visible)"
  if [[ "${N_GPUS}" -le 1 ]]; then
    # Single card: only live utilization signals contention; ignore baseline memory.
    if [[ "${UTIL:-0}" -gt 10 ]]; then
      echo "ERROR: the only GPU looks busy (${UTIL}% util). Benchmark numbers" >&2
      echo "       would be skewed. Wait for it to free up, or pass --skip-check." >&2
      exit 1
    fi
  elif [[ "${USED:-0}" -gt 1024 || "${UTIL:-0}" -gt 10 ]]; then
    echo "ERROR: GPU ${GPU} looks busy (${USED} MiB / ${UTIL}%). Benchmark numbers" >&2
    echo "       would be skewed. Pick an idle GPU or pass --skip-check." >&2
    exit 1
  fi
fi

# --- Default output name carries the detected card slug. ---
if [[ -z "${OUT}" ]]; then
  GPU_SLUG="$(CUDA_VISIBLE_DEVICES="${GPU}" uv run python -c \
    'import torch,re;print(re.sub(r"[^a-z0-9]+","_",torch.cuda.get_device_name(0).lower()).strip("_"))' \
    2>/dev/null || echo gpu)"
  OUT="${SCRIPT_DIR}/pi05_profile_${GPU_SLUG}.json"
fi
echo "[out] profile JSON -> ${OUT}"

# --- 1. Profile sweep (probes device + measures roofline, then sweeps). ---
ROOFLINE_FLAG=()
[[ "${NO_ROOFLINE}" -eq 1 ]] && ROOFLINE_FLAG=(--no-roofline)

CUDA_VISIBLE_DEVICES="${GPU}" uv run python benchmark/pi05/profile_pi05.py \
  --checkpoint "${CKPT}" \
  --batch-sizes ${BATCH_SIZES} \
  --lang-len "${LANG_LEN}" \
  --n-warmup "${N_WARMUP}" \
  --n-timed "${N_TIMED}" \
  --out "${OUT}" \
  "${ROOFLINE_FLAG[@]}"

# --- 2. Render figures (matplotlib pulled in only for the render). ---
echo "[plot] rendering figures -> ${FIG_DIR}"
uv run --with matplotlib python benchmark/pi05/plot_pi05.py \
  --in "${OUT}" --out-dir "${FIG_DIR}"

echo ""
echo "done."
echo "  JSON    : ${OUT}"
echo "  figures : ${FIG_DIR}/fig1..fig7 .svg"
