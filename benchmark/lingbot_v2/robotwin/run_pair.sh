#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the same current RoboTwin task queue against Official and PHYAI LingBot V2.

Required:
  --phyai-root PATH       PHYAI repository root
  --robotwin-root PATH    pinned RoboTwin tree with XPolicyLab
  --output PATH           output root for both runs and the final report

Optional:
  --official-host HOST    Official XPolicyLab bridge host; default: 127.0.0.1
  --official-port PORT    Official XPolicyLab bridge port; default: 18080
  --phyai-host HOST       PHYAI XPolicyLab bridge host; default: 127.0.0.1
  --phyai-port PORT       PHYAI XPolicyLab bridge port; default: 18081
  --num-tasks N           first N official tasks; default: 1
  --tasks LIST            comma-separated explicit task names; overrides --num-tasks
  --test-num N            accepted episodes per task; default: 1
  --seed N                RoboTwin seed group; default: 0
  --task-config NAME      default: demo_clean
  --instruction-type T    seen or unseen; default: unseen (Official ACT config)
  --execution-mode M      action execution mode; only Official chunk is supported
  --expert-check BOOL     default: true
  --resume                skip completed task logs

Both direct LingBot servers and their legacy_policy_bridge.py processes must
already be running. Use separate bridge ports so this script can evaluate the
two backends sequentially with an identical simulator contract.
EOF
}

phyai_root=""
robotwin_root=""
output=""
official_host=127.0.0.1
official_port=18080
phyai_host=127.0.0.1
phyai_port=18081
num_tasks=1
task_list=""
test_num=1
seed=0
task_config=demo_clean
instruction_type=unseen
execution_mode=chunk
expert_check=true
resume=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phyai-root) phyai_root="$2"; shift 2 ;;
    --robotwin-root) robotwin_root="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --official-host) official_host="$2"; shift 2 ;;
    --official-port) official_port="$2"; shift 2 ;;
    --phyai-host) phyai_host="$2"; shift 2 ;;
    --phyai-port) phyai_port="$2"; shift 2 ;;
    --num-tasks) num_tasks="$2"; shift 2 ;;
    --tasks) task_list="$2"; shift 2 ;;
    --test-num) test_num="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --task-config) task_config="$2"; shift 2 ;;
    --instruction-type) instruction_type="$2"; shift 2 ;;
    --execution-mode) execution_mode="$2"; shift 2 ;;
    --expert-check) expert_check="$2"; shift 2 ;;
    --resume) resume=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in phyai_root robotwin_root output; do
  if [[ -z "${!value_name}" ]]; then
    echo "Missing --${value_name//_/-}" >&2
    exit 2
  fi
done

runner="${phyai_root}/benchmark/lingbot_v2/robotwin/run_current_robotwin.sh"
summarizer="${phyai_root}/benchmark/lingbot_v2/robotwin/summarize_success.py"
sim_client="${robotwin_root}/scripts/eval_policy_xpolicylab.py"
for path in "$runner" "$summarizer" "$sim_client"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 2
  fi
done

mkdir -p "${output}/official" "${output}/phyai"
echo "RoboTwin paired evaluation"
echo "Official endpoint: ${official_host}:${official_port}"
echo "PHYAI endpoint   : ${phyai_host}:${phyai_port}"
echo "tasks            : $num_tasks"
echo "episodes/task    : $test_num"
echo "instruction type : $instruction_type"
echo "bridge mode      : $execution_mode"

common_args=(
  --robotwin-root "$robotwin_root"
  --num-tasks "$num_tasks"
  --test-num "$test_num"
  --seed "$seed"
  --task-config "$task_config"
  --instruction-type "$instruction_type"
  --execution-mode "$execution_mode"
  --expert-check "$expert_check"
)
if [[ -n "$task_list" ]]; then
  common_args+=(--tasks "$task_list")
fi
if [[ "$resume" == true ]]; then
  common_args+=(--resume)
fi

echo "===== Official backend ====="
(
  bash "$runner" \
    "${common_args[@]}" \
    --output "${output}/official" \
    --policy-name LingBotV2_Official \
    --host "$official_host" \
    --port "$official_port"
)

echo "===== PHYAI backend ====="
(
  bash "$runner" \
    "${common_args[@]}" \
    --output "${output}/phyai" \
    --policy-name LingBotV2_PHYAI \
    --host "$phyai_host" \
    --port "$phyai_port"
)

python3 "$summarizer" \
  --official-dir "${output}/official" \
  --phyai-dir "${output}/phyai" \
  --csv "${output}/lingbot-v2-robotwin-success.csv" \
  --markdown "${output}/lingbot-v2-robotwin-success.md"
