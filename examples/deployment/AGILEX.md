# AgileX pi0.5 example

This follows `tianji.py`: construct an adapter, the shared `Pi05Policy`, and a
`CompositeRobot`; discard one warmup prediction; run `RobotDeployment`.
There is no RTC, memory, separate inference server, or overlapping prediction.
No generic engine, shared policy, or deployment-loop changes are required.

## Run

Use the prepared PhyAI environment from the checkout root. The camera and Piper
drivers must already be running with ROS domain 0 and `ROS_LOCALHOST_ONLY=0`
(or change `robot.domain_id` / `robot.localhost_only` to match).

**Launching this example sends model actions.** Stop competing control clients
and check the robot workspace and hardware emergency stop before running it.
It does not enable motors, home the robot, clip actions, check controller
ownership, or add a robot-specific watchdog. Stopping only stops publication;
it does not disable motors or issue a hold command.

```bash
CKPT=/path/to/enactive-pi05-step1000
TASK='The instruction used for this checkpoint and scene.'
# The existing engine setting provides enough scratch space for FA2 planning.
export PHYAI_FLASHINFER_WORKSPACE_BYTES=268435456

# Configuration only: no model or ROS resources.
uv run --no-sync python -m examples.deployment.agilex --print-config

# Bounded live trial. max_steps counts interpolated dual-arm send calls.
uv run --no-sync python -m examples.deployment.agilex \
  "policy.checkpoint=$CKPT" "task=$TASK" max_steps=200
```

Omit `max_steps` for continuous execution until Ctrl-C. A local YAML can be
supplied with `--config .cache/agilex.local.yaml`, as in the Tianji example.
Two hundred sends correspond to about one second of active 200 Hz command
sending; loading, warmup and inference gaps add wall time.

For **inference without any ROS connection or robot commands**, use the merged
standalone example with a saved observation NPZ:

```bash
uv run --no-sync python examples/pi05/run_enactive.py \
  --checkpoint "$CKPT" --input /path/to/observation.npz \
  --output .cache/agilex-actions.npz
```

Input fields are native RGB uint8 HWC `front`, `left`, `right`; raw 14D `state`;
and scalar string `task`. Optional `noise` is `[1, 50, 32]` for paired comparisons.
Keep the checkpoint, prompt, images, state, initial noise, dtype and denoising
steps identical when comparing engine outputs.

## Data path

- Read front/left/right images from `/camera_f/color/image_raw`,
  `/camera_l/color/image_raw`, `/camera_r/color/image_raw`.
- Read dual-arm `sensor_msgs/msg/JointState` from `/puppet/joint_left` and
  `/puppet/joint_right`. Reorder by `joint0` through `joint6`.
- State/action order is left six joint radians + gripper meters, then right.
  `AgilexPi05Adapter` uses the exact processor factory from `run_enactive.py`.
  The saved processor performs PIL resizing/padding, tokenization, quantile
  reversal and joint-only chunk anchoring; grippers remain absolute positions.
- The engine emits 50 targets. The shared loop retains 25, with a 30 Hz source
  timeline, and interpolates the complete dual-arm command at nominal 200 Hz.
  Interpolation is in `RobotDeployment`, not in the ROS worker. A full prefix
  covers 25/30 = 0.833 s; observation/inference happens after it drains.
- Publish positions named `joint0` through `joint6` to `/joint_left_states` and
  `/joint_right_states`. Each backend write publishes one message per arm.

PhyAI uses Python 3.12, while ROS Humble on this machine uses Python 3.10.
`robots/agilex_worker.py` runs under the configured system interpreter, and
`robots/agilex.py` connects it to `CompositeRobot` through an inherited Unix
socket. This interpreter bridge is the only extra process; it has no model,
trajectory queue, or rate controller. The existing shared loop retains its
schema, sample-age, prediction-age and scheduling checks, unchanged from Tianji.

## Review map

- `agilex.py`, `configs/agilex.yaml`: composition and configuration.
- `adapters/agilex_pi05.py`: model/robot observation and action mapping.
- `robots/agilex.py`: schema and ROS worker lifecycle.
- `robots/agilex_worker.py`, `robots/agilex_protocol.py`: ROS I/O and wire codecs.
