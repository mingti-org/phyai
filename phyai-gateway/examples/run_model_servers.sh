#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 REPLICAS [server.py] [server args...]" >&2
    echo "Required server args: --checkpoint-dir DIR --tokenizer-dir DIR" >&2
    echo "Environment: BASE_PORT=50063 GPU_START=0 LOG_DIR=./logs/model_servers PYTHON_BIN=..." >&2
    exit 2
fi
REPLICAS="$1"
shift
if ! [[ "$REPLICAS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPLICAS must be a positive integer, got: $REPLICAS" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/inference_server_pi0.5.py"
if [[ $# -gt 0 && "$1" != -* ]]; then
    SERVER_SCRIPT="$1"
    shift
    if [[ "$SERVER_SCRIPT" != */* ]]; then
        SERVER_SCRIPT="$SCRIPT_DIR/$SERVER_SCRIPT"
    fi
fi
if [[ ! -f "$SERVER_SCRIPT" ]]; then
    echo "Server script not found: $SERVER_SCRIPT" >&2
    exit 2
fi

BASE_PORT="${BASE_PORT:-50063}"
GPU_START="${GPU_START:-0}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/model_servers}"
if ! [[ "$BASE_PORT" =~ ^[1-9][0-9]*$ ]] || ((BASE_PORT + REPLICAS - 1 > 65535)); then
    echo "BASE_PORT and REPLICAS must select ports between 1 and 65535" >&2
    exit 2
fi
if ! [[ "$GPU_START" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "GPU_START must be a non-negative integer, got: $GPU_START" >&2
    exit 2
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
    # Resolve once so the tracked PIDs belong to the model processes, not uv.
    PYTHON_BIN="$(uv run --project "$SCRIPT_DIR/../.." python -c 'import sys; print(sys.executable)')"
fi
mkdir -p "$LOG_DIR"

PIDS=()
cleanup() {
    trap - EXIT
    if ((${#PIDS[@]} > 0)); then
        kill "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' SIGINT
trap 'exit 143' SIGTERM

for ((i = 0; i < REPLICAS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU=$((GPU_START + i))
    LOG_FILE="$LOG_DIR/server_$i.log"
    CUDA_VISIBLE_DEVICES="$GPU" \
        "$PYTHON_BIN" "$SERVER_SCRIPT" --port "$PORT" "$@" \
        >"$LOG_FILE" 2>&1 &
    PIDS+=("$!")
    echo "Server $i: GPU=$GPU port=$PORT log=$LOG_FILE"
done

# A replica exit stops the group; propagate failure to the caller.
wait -n "${PIDS[@]}"
