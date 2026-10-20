#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run current RoboTwin tasks against one pre-started XPolicyLab policy endpoint.

Required:
  --robotwin-root PATH   pinned RoboTwin source with initialized XPolicyLab
  --output PATH          result root for this backend
  --policy-name NAME     result label, for example LingBotV2_Official

Optional:
  --host HOST            XPolicyLab bridge host; default: 127.0.0.1
  --port PORT            XPolicyLab bridge port; default: 18080
  --num-tasks N          first N tasks from the official LingBot list; default: 1
  --tasks LIST           comma-separated explicit task names; overrides --num-tasks
  --test-num N           accepted episodes per task; default: 1
  --seed N               RoboTwin seed group; default: 0
  --task-config NAME     default: demo_clean
  --instruction-type T   seen or unseen; default: unseen (Official ACT config)
  --execution-mode M     action execution mode; only Official chunk is supported
  --expert-check BOOL    default: true
  --resume               skip logs already completed at --test-num
EOF
}

robotwin_root=""
output=""
policy_name=""
host="127.0.0.1"
port=18080
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
    --robotwin-root) robotwin_root="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --policy-name) policy_name="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
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

for name in robotwin_root output policy_name; do
  if [[ -z "${!name}" ]]; then
    echo "Missing --${name//_/-}" >&2
    exit 2
  fi
done

if ! [[ "$num_tasks" =~ ^[1-9][0-9]*$ ]] || (( num_tasks > 50 )); then
  echo "--num-tasks must be an integer from 1 through 50" >&2
  exit 2
fi
if ! [[ "$test_num" =~ ^[1-9][0-9]*$ ]]; then
  echo "--test-num must be a positive integer" >&2
  exit 2
fi
if [[ "$expert_check" != true && "$expert_check" != false ]]; then
  echo "--expert-check must be true or false" >&2
  exit 2
fi
if [[ "$instruction_type" != seen && "$instruction_type" != unseen ]]; then
  echo "--instruction-type must be seen or unseen" >&2
  exit 2
fi
if [[ "$execution_mode" != chunk ]]; then
  echo "--execution-mode must be chunk" >&2
  exit 2
fi
eval_script="${robotwin_root}/scripts/eval_policy_xpolicylab.py"
xpl_server="${robotwin_root}/XPolicyLab/client_server/ws/model_server.py"
task_config_path="${robotwin_root}/env_cfg/task_config/${task_config}.yml"
for path in "$eval_script" "$xpl_server" "$task_config_path"; do
  if [[ ! -f "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 2
  fi
done

official_tasks=(
  lift_pot hanging_mug stack_bowls_three scan_object handover_block
  click_bell put_object_cabinet open_microwave stack_blocks_three place_shoe
  adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
  click_alarmclock dump_bin_bigbin grab_roller handover_mic move_can_pot
  move_pillbottle_pad move_playingcard_away place_cans_plasticbox
  place_container_plate place_dual_shoes place_empty_cup place_fan
  place_mouse_pad place_object_basket place_object_scale place_object_stand
  place_phone_stand move_stapler_pad open_laptop pick_diverse_bottles
  pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket
  place_bread_skillet place_burger_fries place_can_basket press_stapler
  rotate_qrcode shake_bottle_horizontally shake_bottle stack_blocks_two
  stack_bowls_two stamp_seal turn_switch put_bottles_dustbin
)

if [[ -n "$task_list" ]]; then
  IFS=',' read -r -a tasks <<< "$task_list"
  if [[ ${#tasks[@]} -eq 0 ]]; then
    echo "--tasks must name at least one task" >&2
    exit 2
  fi
  for task_name in "${tasks[@]}"; do
    found=false
    for official_task in "${official_tasks[@]}"; do
      if [[ "$task_name" == "$official_task" ]]; then
        found=true
        break
      fi
    done
    if [[ "$found" != true ]]; then
      echo "Unknown RoboTwin task in --tasks: $task_name" >&2
      exit 2
    fi
  done
else
  tasks=("${official_tasks[@]:0:num_tasks}")
fi

python_bin="${PYTHON:-python3}"
export PYTHONPATH="${robotwin_root}:${robotwin_root}/XPolicyLab:${PYTHONPATH:-}"
export ROBOTWIN_SUPPRESS_EVAL_CONFIG=1

protocol_manifest="${output}/protocol.env"
eval_sha256="$(sha256sum "$eval_script" | awk '{print $1}')"
task_config_sha256="$(sha256sum "$task_config_path" | awk '{print $1}')"
protocol_content="$(cat <<EOF
schema=lingbot-v2-robotwin-current-v3
policy_name=${policy_name}
endpoint=${host}:${port}
task_config=${task_config}
task_config_sha256=${task_config_sha256}
instruction_type=${instruction_type}
execution_mode=${execution_mode}
seed_group=${seed}
expert_check=${expert_check}
action_type=joint
eval_script_sha256=${eval_sha256}
EOF
)"
if [[ -f "$protocol_manifest" ]]; then
  existing_protocol="$(cat "$protocol_manifest")"
  if [[ "$existing_protocol" != "$protocol_content" ]]; then
    echo "Result directory uses a different evaluation protocol: $protocol_manifest" >&2
    diff -u <(printf '%s\n' "$existing_protocol") <(printf '%s\n' "$protocol_content") || true
    echo "Use a fresh --output directory; do not mix incompatible success logs." >&2
    exit 1
  fi
elif compgen -G "${output}/eval_logs/*.log" >/dev/null; then
  echo "Existing logs have no protocol manifest: ${output}/eval_logs" >&2
  echo "Use a fresh --output directory instead of resuming unaudited logs." >&2
  exit 1
else
  mkdir -p "$output"
  printf '%s\n' "$protocol_content" > "$protocol_manifest"
fi
mkdir -p "${output}/eval_logs"

echo "RoboTwin current-protocol evaluation"
echo "policy       : $policy_name"
echo "endpoint     : ${host}:${port}"
echo "task config  : $task_config"
echo "instruction  : $instruction_type"
echo "bridge mode : $execution_mode"
echo "tasks        : ${tasks[*]}"
echo "episodes/task: $test_num"
echo "seed group   : $seed"
echo "expert check : $expert_check"
echo "manifest     : $protocol_manifest"

for index in "${!tasks[@]}"; do
  task_name="${tasks[$index]}"
  log_path="${output}/eval_logs/${task_name}.log"
  if [[ "$resume" == true && -f "$log_path" ]] &&
    grep -Eiq "Final success rate: [0-9]+/${test_num}[[:space:]]*=" "$log_path"; then
    echo "[skip completed] $task_name"
    continue
  fi

  echo "[run $((index + 1))/${#tasks[@]}] $task_name"
  (
    cd "$robotwin_root"
    "$python_bin" -u "$eval_script" \
      --bench_name RoboTwin \
      --task_name "$task_name" \
      --policy_name "$policy_name" \
      --host "$host" \
      --port "$port" \
      --protocol ws \
      --eval_batch false \
      --root_dir "$robotwin_root" \
      --device_id 0 \
      --additional_info "ckpt_name=${policy_name},action_type=joint" \
      --seed "$seed" \
      --task_config "$task_config" \
      --instruction_type "$instruction_type" \
      --test_num "$test_num" \
      --expert_check "$expert_check"
  ) 2>&1 | tee "$log_path"

  if grep -Fq "Policy rollout error:" "$log_path"; then
    echo "Policy rollout failed; refusing to count this episode: $log_path" >&2
    exit 1
  fi
  if ! grep -Eiq \
    "Final success rate: [0-9]+/${test_num}[[:space:]]*=" \
    "$log_path"; then
    echo "Final success marker is missing: $log_path" >&2
    exit 1
  fi
done

echo "RoboTwin backend evaluation complete: $output"
