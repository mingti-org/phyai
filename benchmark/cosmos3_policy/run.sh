#!/usr/bin/env bash
# Run the full Cosmos3 policy (action / VLA) profiling pipeline: probe the card +
# measure its roofline, sweep batch size at the recommended denoise-step count,
# then render the figures. Outputs (JSON, figures) are regenerated here and are
# gitignored — only this script and the .py/.md files are committed.
#
# The denoise-step count is FIXED at the released DROID policy's recommended N=4
# (cosmos-framework sample_args.json). The swept axis is BATCH SIZE — how many
# observations / robots are served in one forward. Action inference takes the
# action and never decodes the jointly-denoised rollout video, so VAE decode is
# off the critical path (time it separately with --decode-video).
#
# Usage:
#   ./run.sh --checkpoint /path/to/Cosmos3-Nano-Policy-DROID [options]
#
# Options:
#   -c, --checkpoint DIR    policy checkpoint (transformer/ + vae/) (REQUIRED; or $CKPT)
#   -g, --gpu N             GPU to pin via CUDA_VISIBLE_DEVICES (default 0)
#   -b, --batch-sizes "..." space-separated batch list (default "1 2 4 8 16")
#       --guidance F        CFG scale; >1 = cond+uncond per step (default 3.0)
#       --action-chunk N    action horizon (default 32, tech report §4.2.5)
#       --raw-action-dim N  embodiment action width (default 10, droid_lerobot)
#       --domain-id N       embodiment domain id (default 8, droid_lerobot)
#       --num-frames N      observation video frames (default 0 = auto = chunk+1)
#       --height N          observation height (default 480)
#       --width N           observation width (default 832)
#   -o, --out FILE          profile JSON path (default cosmos3_policy_profile_<gpu>.json)
#   -f, --fig-dir DIR       figure output dir (default figures/)
#       --n-warmup N        warmup iters    (default 2)
#       --n-timed N         timed iters      (default 5)
#       --no-vae            skip VAE encode/decode (transformer phases only)
#       --decode-video      also time VAE decode (off the action path)
#       --no-roofline       skip the peak/bandwidth microbench
#       --skip-check        don't abort when the target GPU looks busy
#       --plot-only         re-render figures from an existing JSON (no GPU run)
#   -h, --help              show this help
#
# Examples:
#   ./run.sh -c $CKPT -g 7
#   ./run.sh -c $CKPT -g 7 -b "1 4 16" --guidance 1.0
#   ./run.sh --plot-only -o cosmos3_policy_profile_nvidia_thor.json
set -euo pipefail

# --- Repo paths. This script lives in benchmark/cosmos3_policy; root is 2 up. ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Defaults ---
CKPT="${CKPT:-}"
GPU=0
BATCH_SIZES="1 2 4 8 16"
GUIDANCE=3.0
ACTION_CHUNK=32
RAW_ACTION_DIM=10
DOMAIN_ID=8
NUM_FRAMES=0
HEIGHT=480
WIDTH=832
OUT=""
FIG_DIR="${SCRIPT_DIR}/figures"
N_WARMUP=2
N_TIMED=5
NO_VAE=0
DECODE_VIDEO=0
NO_ROOFLINE=0
SKIP_CHECK=0
PLOT_ONLY=0

usage() { sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--checkpoint)   CKPT="$2"; shift 2 ;;
    -g|--gpu)          GPU="$2"; shift 2 ;;
    -b|--batch-sizes)  BATCH_SIZES="$2"; shift 2 ;;
    --guidance)        GUIDANCE="$2"; shift 2 ;;
    --action-chunk)    ACTION_CHUNK="$2"; shift 2 ;;
    --raw-action-dim)  RAW_ACTION_DIM="$2"; shift 2 ;;
    --domain-id)       DOMAIN_ID="$2"; shift 2 ;;
    --num-frames)      NUM_FRAMES="$2"; shift 2 ;;
    --height)          HEIGHT="$2"; shift 2 ;;
    --width)           WIDTH="$2"; shift 2 ;;
    -o|--out)          OUT="$2"; shift 2 ;;
    -f|--fig-dir)      FIG_DIR="$2"; shift 2 ;;
    --n-warmup)        N_WARMUP="$2"; shift 2 ;;
    --n-timed)         N_TIMED="$2"; shift 2 ;;
    --no-vae)          NO_VAE=1; shift ;;
    --decode-video)    DECODE_VIDEO=1; shift ;;
    --no-roofline)     NO_ROOFLINE=1; shift ;;
    --skip-check)      SKIP_CHECK=1; shift ;;
    --plot-only)       PLOT_ONLY=1; shift ;;
    -h|--help)         usage; exit 0 ;;
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
  uv run --with matplotlib python benchmark/cosmos3_policy/plot_cosmos3_policy.py \
    --in "${OUT}" --out-dir "${FIG_DIR}"
  exit 0
fi

# --- Validate checkpoint ---
if [[ -z "${CKPT}" || ! -d "${CKPT}" ]]; then
  echo "ERROR: --checkpoint must be an existing directory (got: '${CKPT}')." >&2
  echo "       Pass -c /path/to/Cosmos3-Nano-Policy-DROID or export CKPT=..." >&2
  exit 1
fi

echo "============================================================"
echo " Cosmos3 policy profiling pipeline (N=4 fixed, sweep batch)"
echo "   checkpoint : ${CKPT}"
echo "   GPU        : ${GPU}   (CUDA_VISIBLE_DEVICES)"
echo "   batch sizes: ${BATCH_SIZES}   guidance=${GUIDANCE}"
echo "   action     : chunk=${ACTION_CHUNK} raw_dim=${RAW_ACTION_DIM} domain=${DOMAIN_ID}"
echo "   obs        : $([ "${NUM_FRAMES}" -eq 0 ] && echo "auto(chunk+1)" || echo "${NUM_FRAMES}f") ${HEIGHT}x${WIDTH}"
echo "   warmup=${N_WARMUP} timed=${N_TIMED}  no_vae=${NO_VAE} decode_video=${DECODE_VIDEO}"
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
  OUT="${SCRIPT_DIR}/cosmos3_policy_profile_${GPU_SLUG}.json"
fi
echo "[out] profile JSON -> ${OUT}"

# --- Optional flags ---
EXTRA=()
[[ "${NO_VAE}" -eq 1 ]] && EXTRA+=(--no-vae)
[[ "${DECODE_VIDEO}" -eq 1 ]] && EXTRA+=(--decode-video)
[[ "${NO_ROOFLINE}" -eq 1 ]] && EXTRA+=(--no-roofline)

# --- 1. Profile sweep (probes device + measures roofline, then sweeps batch). ---
CUDA_VISIBLE_DEVICES="${GPU}" uv run python benchmark/cosmos3_policy/profile_cosmos3_policy.py \
  --checkpoint "${CKPT}" \
  --batch-sizes ${BATCH_SIZES} \
  --guidance "${GUIDANCE}" \
  --action-chunk "${ACTION_CHUNK}" \
  --raw-action-dim "${RAW_ACTION_DIM}" \
  --domain-id "${DOMAIN_ID}" \
  --num-frames "${NUM_FRAMES}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --n-warmup "${N_WARMUP}" \
  --n-timed "${N_TIMED}" \
  --out "${OUT}" \
  "${EXTRA[@]}"

# --- 2. Render figures (matplotlib pulled in only for the render). ---
echo "[plot] rendering figures -> ${FIG_DIR}"
uv run --with matplotlib python benchmark/cosmos3_policy/plot_cosmos3_policy.py \
  --in "${OUT}" --out-dir "${FIG_DIR}"

echo ""
echo "done."
echo "  JSON    : ${OUT}"
echo "  figures : ${FIG_DIR}/fig1..fig7 .svg"
