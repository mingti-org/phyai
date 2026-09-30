#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 DP [server.py] [server args...]" >&2
    echo "" >&2
    echo "Example:" >&2
    echo "  $0 4 inference_server_pi0.5.py" >&2
    echo "" >&2
    echo "Environment variables:" >&2
    echo "  BASE_PORT=30000" >&2
    echo "  GPU_START=0" >&2
    echo "  PYTHON_BIN=python" >&2
    echo "  LOG_DIR=./logs/model_servers" >&2
    exit 2
fi
DP="$1"
shift
if ! [[ "$DP" =~ ^[1-9][0-9]*$ ]]; then
    echo "DP must be a positive integer, got: $DP" >&2
    exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 0 && "$1" != -* ]]; then
    SERVER_SCRIPT="$1"
    shift
else
    SERVER_SCRIPT="$SCRIPT_DIR/inference_server_pi0.5.py"
fi
if [[ "$SERVER_SCRIPT" != */* ]]; then
    SERVER_SCRIPT="$SCRIPT_DIR/$SERVER_SCRIPT"
fi
if [[ ! -f "$SERVER_SCRIPT" ]]; then
    echo "Server script not found: $SERVER_SCRIPT" >&2
    exit 2
fi
BASE_PORT="${BASE_PORT:-30000}"
GPU_START="${GPU_START:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/model_servers}"
if ! [[ "$BASE_PORT" =~ ^[0-9]+$ ]]; then
    echo "BASE_PORT must be a non-negative integer, got: $BASE_PORT" >&2
    exit 2
fi
if ! [[ "$GPU_START" =~ ^[0-9]+$ ]]; then
    echo "GPU_START must be a non-negative integer, got: $GPU_START" >&2
    exit 2
fi
mkdir -p "$LOG_DIR"
PIDS=()
cleanup() {
    trap - SIGINT SIGTERM EXIT
    if ((${#PIDS[@]} > 0)); then
        echo
        echo "Stopping servers: ${PIDS[*]}"
        kill "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup SIGINT SIGTERM EXIT
for ((i = 0; i < DP; i++)); do
    PORT=$((BASE_PORT + i))
    GPU=$((GPU_START + i))
    LOG_FILE="$LOG_DIR/server_$i.log"
    echo "Starting server $i:"
    echo "  GPU:  $GPU"
    echo "  Port: $PORT"
    echo "  Log:  $LOG_FILE"
    CUDA_VISIBLE_DEVICES="$GPU" \
        "$PYTHON_BIN" \
        "$SERVER_SCRIPT" \
        --port "$PORT" \
        "$@" \
        >"$LOG_FILE" 2>&1 &
    PID=$!
    PIDS+=("$PID")
    echo "  PID:  $PID"
done
echo
echo "Servers started: ${PIDS[*]}"
echo "Press Ctrl-C to stop all servers."
wait

# BASE_PORT=31000 ./run_model_servers.sh 1