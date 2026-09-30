r"""Deploy a converted Enactive pi0.5 checkpoint on a dual-arm AgileX robot.

This is a LIVE example, not a dry-run. It follows ``tianji.py``: load the policy,
discard one warmup prediction, then run the shared ``RobotDeployment`` loop.
There is no RTC, memory, separate inference server, or overlapping prediction.
The commands below describe the tested setup using existing Orbbec camera and
Piper drivers. Adapt the site-local paths and camera/CAN mapping for another
installation. Driver scripts are not shipped with PhyAI.

1. Prepare the robot host
-------------------------
Run these commands in Bash on the robot host, for example after ``ssh agilex``.
The assumed layout is ``$HOME/phyai`` (this checkout), ``$HOME/models`` (converted
checkpoints), and the existing ``$HOME/aloha_capture_app``, ``$HOME/camera_ros``,
``$HOME/agilex_ws``, and ``$HOME/piper_ros`` driver workspaces.

Use the prepared PhyAI ``.venv`` with ``uv run --no-sync``. PhyAI uses Python
3.12, while the ROS Humble worker uses ``/usr/bin/python3`` and sources
``/opt/ros/humble/setup.bash`` in its own process. Do not add ROS Python 3.10
packages to the PhyAI interpreter's ``PYTHONPATH``.

Before starting drivers, clear the robot workspace, check the physical emergency
stop, and stop other action-producing clients. Do not run an Enactive deployment
client/tunnel alongside this example. The Piper launch below defaults to
``auto_enable=true`` and can enable the motors. PhyAI itself does not enable,
home, clip trajectories, or discover control ownership.

Inspect the existing CAN setup::

    ip -brief link show type can

The driver expects ``can_left`` and ``can_right`` to be UP at 1 Mbit/s. If they
are already configured, leave them alone. If not, review the site's USB-to-CAN
mapping before running its existing setup script; do not reconfigure a live bus::

    cd "$HOME/piper_ros"
    bash can_config.sh

2. Start cameras and Piper in tmux
----------------------------------
All participating processes must use the same ROS domain and discovery settings.
These commands explicitly set ``ROS_DOMAIN_ID=0`` and ``ROS_LOCALHOST_ONLY=0`` to
match ``configs/agilex.yaml`` rather than inheriting shell defaults. If using
another domain, change both the driver environment and ``robot.domain_id`` /
``robot.localhost_only`` overrides.

Check ``tmux ls`` first. If the camera/Piper windows below already exist and are
healthy, reuse them instead of starting duplicate drivers. The following creates
a new session and writes logs under the ignored ``.cache`` directory::

    mkdir -p "$HOME/phyai/.cache/deployment/agilex-live"
    tmux new-session -d -s phyai-agilex-live -n camera '
        set -o pipefail
        export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
        bash "$HOME/aloha_capture_app/scripts/start_camera.sh" 2>&1 |
            tee "$HOME/phyai/.cache/deployment/agilex-live/camera.log"
    '
    tmux set-window-option -t phyai-agilex-live:camera remain-on-exit on

The camera wrapper sources ROS and the installed camera workspaces, then runs
``$HOME/camera_ros/scripts/start_orbbec_camera.sh``. That script launches three
Orbbec cameras with the site's configured serial numbers. Wait for the streams
to start, then launch the dual-arm Piper drivers::

    tmux new-window -d -t phyai-agilex-live -n piper '
        set -o pipefail
        export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
        bash "$HOME/aloha_capture_app/scripts/start_piper.sh" 2>&1 |
            tee "$HOME/phyai/.cache/deployment/agilex-live/piper.log"
    '
    tmux set-window-option -t phyai-agilex-live:piper remain-on-exit on

The Piper wrapper sources ROS and the installed driver workspaces, then runs
``ros2 launch piper start_two_piper.launch.py``. It uses ``can_left`` and
``can_right`` and exposes the joint-state interfaces listed below.

In a separate Bash shell, check that the sensor topics have publishers and both
command topics have driver subscribers. Commands in this block only inspect
ROS; they do not send actions::

    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
    ros2 topic list --no-daemon
    for topic in /camera_f/color/image_raw /camera_l/color/image_raw \
        /camera_r/color/image_raw /puppet/joint_left /puppet/joint_right \
        /joint_left_states /joint_right_states; do
        ros2 topic info "$topic" --verbose --no-daemon
    done

Before starting PhyAI, the two command topics should have zero publishers.
Sensor publishers alone do not prove that frames are arriving; inspect the
camera/driver logs if startup later times out waiting for an observation.

3. Inspect configuration without moving the robot
-------------------------------------------------
Use a fresh Bash shell for PhyAI, without manually sourcing the ROS workspaces.
This command prints the resolved YAML and exits before model loading, ROS
connection, or action publication::

    cd "$HOME/phyai"
    uv run --no-sync python -m examples.deployment.agilex --print-config \
        "policy.checkpoint=$HOME/models/enactive-pi05-step1000" \
        "task=Place the red cube on the green cube."

The checkpoint must be the converted PhyAI directory, including its saved
processor, rather than the original Enactive training checkpoint. Defaults are
BF16 kernels, CUDA graphs, the checkpoint's 10 denoising steps, a 30 Hz target
timeline, and a nominal 200 Hz command rate. ``max_steps: null`` means no step
limit. To use a private YAML, add ``--config .cache/agilex.local.yaml``; dotted
command-line overrides take precedence.

4. Start continuous live inference and control
----------------------------------------------
This starts sending model actions after loading, receiving an observation, and
discarding the warmup chunk. It runs until Ctrl-C or an error, with no maximum
step count. The existing FlashInfer workspace setting gives its FA2 planner
256 MiB of scratch space; it is not a GPU-memory precheck::

    tmux new-window -d -t phyai-agilex-live -n policy '
        set -o pipefail
        cd "$HOME/phyai" || exit
        export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
        export PHYAI_FLASHINFER_WORKSPACE_BYTES=268435456
        CKPT="$HOME/models/enactive-pi05-step1000"
        TASK="Place the red cube on the green cube."
        uv run --no-sync python -u -m examples.deployment.agilex \
            "policy.checkpoint=$CKPT" "task=$TASK" 2>&1 |
            tee .cache/deployment/agilex-live/policy.log
    '
    tmux set-window-option -t phyai-agilex-live:policy remain-on-exit on

No Enactive server, SSH tunnel, or second model-serving process is needed.
The only helper process is the system-Python ROS worker. It transports images,
feedback and commands; interpolation is in ``RobotDeployment``, not the worker.

The checkpoint predicts 50 targets; the loop executes the first 25, representing
25/30 seconds of the model timeline, then reads a new observation and predicts
again. The nominal 200 Hz applies while sending an interpolated prefix. The
ordinary serial loop has observation/inference gaps between prefixes; this is
not a continuous 200 Hz inference loop.

Watch startup and logs, or attach to the session::

    tail -f "$HOME/phyai/.cache/deployment/agilex-live/policy.log"
    tmux attach -t phyai-agilex-live

Successful startup prints ``Warmup: ... discarded 50 actions`` followed by
``AgileX: source=30Hz, control=200Hz, horizon=25; Ctrl-C stops publishing``.
These messages indicate control-loop startup, not task completion. In tmux,
select the policy window to inspect it; Ctrl-B then D detaches without stopping
control. Ctrl-C in ``tail -f`` only stops the log viewer, not the robot policy.

5. Stop, verify, and deliberately restart
-----------------------------------------
From any shell on the robot host, interrupt only the policy window::

    tmux send-keys -t phyai-agilex-live:policy C-c
    tmux list-panes -t phyai-agilex-live:policy \
        -F '#{window_name} dead=#{pane_dead} exit=#{pane_dead_status}'

Wait for ``dead=1`` (normally exit 130 after Ctrl-C). To inspect publisher counts
from a separate ROS shell::

    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
    ros2 topic info /joint_left_states --verbose --no-daemon
    ros2 topic info /joint_right_states --verbose --no-daemon

Both should report zero publishers once discovery catches up, assuming no other
controller is running. Ctrl-C closes the PhyAI process and its ROS worker and
stops publishing. It leaves cameras, Piper drivers, and motor enable state
unchanged: it does not send a hold, home, or motor-disable command. For an urgent
physical stop, use the robot's hardware emergency stop, not an SSH/chat request.

If the policy pane is dead and the drivers are still healthy, intentionally
restart the SAME checkpoint, task, and command with::

    tmux respawn-pane -t phyai-agilex-live:policy

This immediately starts a new live run and overwrites ``policy.log``. Save logs
first if needed. To change the task or checkpoint, replace the dead policy
window with the launch command above using new values; do not start a second
policy beside an active one.

Interfaces and environment notes
---------------------------------
Observations are RGB uint8 images, 640x480 in the supplied config, from
``/camera_f/color/image_raw``, ``/camera_l/color/image_raw``, and
``/camera_r/color/image_raw``, plus ``sensor_msgs/msg/JointState`` feedback from
``/puppet/joint_left`` and ``/puppet/joint_right``. State/action order is left
six joints in radians plus gripper opening in meters, then the same for right.
Each backend write publishes one ``JointState`` per arm, named ``joint0`` through
``joint6``, to ``/joint_left_states`` and ``/joint_right_states``. Camera rates
and feedback rates are determined by their drivers, not by ``control_hz``.

The prepared environment keeps the repository's Torch, FlashInfer and
Transformers pins. For the tested FlashInfer 0.6.17 / apache-tvm-ffi 0.1.9 pair,
CUTLASS DSL 4.8 caused a ``make_kwargs_wrapper(map_dataclass_to_tuple=...)``
compatibility error. The environment-only fix was::

    cd "$HOME/phyai"
    uv pip install --python .venv/bin/python 'nvidia-cutlass-dsl[cu13]==4.5.2'

Do not reapply this to a working environment without checking its versions.
The lockfile was not changed for this workaround, so use ``uv run --no-sync``
to preserve the prepared environment. If the FA2 planner reports insufficient
workspace, first check that the 256 MiB environment setting above is present.
See ``AGILEX.md`` for processor/action mapping and inference-only comparison
using ``examples/pi05/run_enactive.py`` with a saved observation NPZ; that
standalone path does not connect to ROS or send robot commands.
"""

from __future__ import annotations

import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.deployment"

from phyai_robot import RobotDeployment

from .configuration import parse_config, repository_path
from .options import DeploymentConfig
from .policies import Pi05Policy, Pi05PolicyConfig
from .robots.agilex import (
    ACTION_SCHEMA,
    AgilexRobotConfig,
    make_agilex_robot,
    wait_for_observation,
)

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "agilex.yaml"


@dataclass
class AgilexDeploymentConfig:
    task: str = ""
    max_steps: int | None = None
    robot: AgilexRobotConfig = field(default_factory=AgilexRobotConfig)
    policy: Pi05PolicyConfig = field(default_factory=Pi05PolicyConfig)
    deployment: DeploymentConfig = field(
        default_factory=lambda: DeploymentConfig(
            action_hz=30, execution_horizon=25, max_queued_actions=25
        )
    )

    def validate(self):
        if not self.task.strip():
            raise ValueError("task must be non-empty")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive or null")
        self.robot.validate()
        self.policy.validate()
        self.deployment.to_options()


def main(argv=None):
    config = parse_config(
        AgilexDeploymentConfig,
        default_config=DEFAULT_CONFIG,
        argv=argv,
        validate=AgilexDeploymentConfig.validate,
    )
    from .adapters.agilex_pi05 import AgilexPi05Adapter

    options = config.deployment.to_options(interpolate_keys=ACTION_SCHEMA)
    checkpoint = repository_path(config.policy.checkpoint)
    with ExitStack() as resources:
        adapter = AgilexPi05Adapter(
            checkpoint, task=config.task, action_hz=options.action_hz
        )
        if options.execution_horizon > adapter.horizon:
            raise ValueError("execution_horizon exceeds checkpoint chunk_size")
        policy = Pi05Policy(
            checkpoint,
            kernel_policy=repository_path(config.policy.kernel_policy),
            seed=config.policy.seed,
            num_inference_steps=config.policy.num_inference_steps,
            use_cuda_graph=config.policy.use_cuda_graph,
            num_threads=config.policy.num_threads,
        )
        resources.callback(policy.close)
        robot = make_agilex_robot(config.robot)
        resources.callback(robot.close)
        robot.connect()
        observation = wait_for_observation(
            robot,
            timeout_s=options.observation_timeout_s,
            max_age_s=options.max_sample_age_s,
        )
        started = time.monotonic()
        warmup = adapter.to_actions(
            policy.predict(adapter.to_request(observation)), observation
        )
        print(
            f"Warmup: {time.monotonic() - started:.3f}s; denoise_steps={policy.num_inference_steps}; discarded {len(warmup.actions)} actions",
            flush=True,
        )
        wait_for_observation(
            robot,
            timeout_s=options.observation_timeout_s,
            max_age_s=options.max_sample_age_s,
        )
        print(
            f"AgileX: source={options.action_hz:g}Hz, control={options.control_hz:g}Hz, horizon={options.execution_horizon}; Ctrl-C stops publishing",
            flush=True,
        )
        RobotDeployment(
            robot=robot, policy=policy, adapter=adapter, options=options
        ).run(max_steps=config.max_steps)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Deployment interrupted; robot cleanup requested.", file=sys.stderr)
        raise SystemExit(130) from None
