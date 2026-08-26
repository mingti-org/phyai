# LingBot V2 RoboTwin evaluation

This directory compares closed-loop task success between the official LingBot
V2 implementation and PHYAI on a current RoboTwin/XPolicyLab checkout.

Current RoboTwin no longer uses the direct websocket protocol shipped by the
official LingBot repository. The comparison therefore has one transport-only
bridge on each side:

```text
RoboTwin evaluator
  -> XPolicyLab websocket
  -> legacy_policy_bridge.py
  -> direct LingBot websocket
  -> Official or PHYAI policy
```

The same bridge code is used for both backends. It maps protocol field names
but does not normalize values, resize images, run the model, or change actions.
Those operations remain in the official and PHYAI policy implementations.

For implementation diagnostics, both bridges may additionally receive
`--paired-noise --noise-seed 20260819`. This injects the same diffusion noise
for the same task, episode seed, and model-request index. Keep it disabled when
measuring the unmodified Official success protocol.

## Evaluation contract

- cameras: `cam_head`, `cam_left_wrist`, and `cam_right_wrist`, preserved as
  RGB and mapped to the three official LingBot camera keys;
- state: RoboTwin's interleaved 14-D layout
  `[left arm 6, left gripper, right arm 6, right gripper]`;
- normalization: the official `assets/norm_stats/robotwin.json` with
  `bounds_99_woclip` and epsilon `1e-6`;
- output: an unnormalized `50 x 14` action chunk in RoboTwin joint order;
- language split: `unseen`, matching the released legacy
  `policy/ACT/deploy_policy.yml` evaluation harness;
- execution: each model request returns the complete 50-step action chunk,
  matching the released LingBot deployment CLI and launcher defaults;
- task order: the 50-task order from the official LingBot launcher;
- simulator revision, assets, task config, prompts, seeds, episode count, and
  expert filtering must be identical for Official and PHYAI.

## Prerequisites

Keep these trees and data separate:

1. this PHYAI repository;
2. the official `lingbot-vla-v2` source;
3. the pinned RoboTwin source with its pinned XPolicyLab checkout;
4. the Qwen3-VL processor and a LingBot V2 checkpoint post-trained for
   RoboTwin.

The released pre-trained checkpoint alone is not a valid task-success
checkpoint. The official server also expects the training-run layout: its
checkpoint is below `checkpoints/global_step_*/hf_ckpt`, while
`lingbotvla_cli.yaml` is stored at the training-run root.

Before loading weights, verify the PHYAI boundary adapter against the official
statistics:

```bash
export PYTHONPATH=/workspace/phyai/src:/workspace/phyai-kernel:/workspace/phyai-utils-tools/src:/workspace/phyai-ext/src:/workspace/phyai-model-optimizer/src

python3 benchmark/lingbot_v2/robotwin/check_adapter_parity.py \
  --stats-json /models/lingbot-v2-robotwin-run/assets/norm_stats/robotwin.json
```

## Start the direct policy servers

Start the official server on one port:

```bash
cd /workspace/official-lingbot
export QWEN3VL_PATH=/models/Qwen3-VL-4B-Instruct-processor

python3 deploy/lingbot_vla_v2_policy.py \
  --model_path /models/lingbot-v2-robotwin-run/checkpoints/global_step_30000/hf_ckpt \
  --use_length 50 \
  --chunk_ret true \
  --use_compile false \
  --port 9330
```

Start PHYAI on another port. On Thor, `gemm` avoids the unsupported cuDNN BF16
Conv3D shape while preserving BF16 PatchEmbed semantics.

```bash
cd /workspace/phyai
export LINGBOT_PROCESSOR=/models/Qwen3-VL-4B-Instruct-processor
export LINGBOT_ROBOTWIN_STATS=/workspace/official-lingbot/assets/norm_stats/robotwin.json

python3 -m benchmark.lingbot_v2.robotwin.phyai_policy_server \
  --model_path /models/lingbot-v2-robotwin-run/checkpoints/global_step_30000/hf_ckpt \
  --use_length 50 \
  --port 9331 \
  --patch-embed-backend gemm \
  --linear-kernel torch \
  --use-cuda-graph true
```

Run `nvidia-smi` and a minimal CUDA tensor operation before either server. Use
the same GPU, checkpoint, processor, statistics, precision, and inference-step
settings for the comparison.

## Start the protocol bridges

Run the bridges in an environment that contains the pinned RoboTwin and
XPolicyLab sources. With host networking, the direct servers may live in a
different container on the same machine.

```bash
python3 -m benchmark.lingbot_v2.robotwin.legacy_policy_bridge \
  --robotwin-root /workspace/RoboTwin \
  --backend-host 127.0.0.1 \
  --backend-port 9330 \
  --port 18080

python3 -m benchmark.lingbot_v2.robotwin.legacy_policy_bridge \
  --robotwin-root /workspace/RoboTwin \
  --backend-host 127.0.0.1 \
  --backend-port 9331 \
  --port 18081
```

To isolate model implementation differences, add the same options to both
commands:

```bash
  --paired-noise --noise-seed 20260819
```

## Run and summarize

First run one accepted episode on one task:

```bash
bash benchmark/lingbot_v2/robotwin/run_pair.sh \
  --phyai-root /workspace/phyai \
  --robotwin-root /workspace/RoboTwin \
  --output /results/lingbot-v2-robotwin \
  --num-tasks 1 \
  --test-num 1 \
  --instruction-type unseen
```

After the smoke test succeeds, rerun with the agreed task and episode counts.
Both backends are evaluated sequentially with the same arguments. `--resume`
skips only tasks whose final success line already matches the requested episode
count.

The output contains:

- `official/eval_logs/*.log`;
- `phyai/eval_logs/*.log`;
- `lingbot-v2-robotwin-success.csv`;
- `lingbot-v2-robotwin-success.md`.

Each backend output also contains `protocol.env`. A result directory is
rejected when its recorded instruction split, simulator source hash, seed, or
task config differs from the requested run. This prevents `--resume` from
mixing results produced under different simulator contracts.

## Audit against the released table

The released table reports 100 accepted episodes per task. The released
LingBot launcher references `script/eval_polict_client_openpi.py`, which is not
present in the public RoboTwin revisions inspected here, and current RoboTwin
uses a later XPolicyLab evaluator. Consequently this harness is suitable for a
paired Official/PHYAI implementation comparison, but it is not by itself an
exact reproduction of the model-card protocol. Audit a completed Official
result directory before interpreting an absolute gap:

```bash
python3 benchmark/lingbot_v2/robotwin/audit_success_gap.py \
  --model-card /models/lingbot-vla-v2-6b-robotwin/README.md \
  --results-dir /results/official \
  --lingbot-root /workspace/official-lingbot \
  --robotwin-root /workspace/RoboTwin \
  --json /results/official-model-card-audit.json
```

The audit reports exact per-task binomial tests, Wilson intervals, observed
instruction split, episode-count mismatches, RoboTwin revision provenance, and
whether the legacy Official client/harness files are present. A low p-value
shows that sampling noise alone is not a sufficient explanation; it does not
prove that a model implementation is wrong when the published evaluator or
simulator revision is missing.

The report includes micro Avg Acc across all episodes and macro Avg Acc across
tasks. Incomplete or mismatched task sets fail by default instead of producing
a partial comparison.
