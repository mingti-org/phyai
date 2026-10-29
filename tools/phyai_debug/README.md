# Robot trace inspector

Record a Tianji client session and inspect the observations, predicted chunks,
published commands, and feedback on one timeline. The debug launcher uses the
existing PhyAI deployment lifecycle. The viewer reads saved files and has no
robot control endpoints.

## What runs where

| Command | Environment | Robot access |
| --- | --- | --- |
| `check` | Robot Python with the debug source on its import path | Configuration and compatibility checks only |
| `demo` | Python with NumPy and Pillow | Synthetic data only |
| `view` | Python with the viewer assets built | Local file reads only |
| `run` | The same sourced ROS environment as the normal Tianji client | **Enables control through the normal client lifecycle** |

`run` does not start the model server, home the robot, or change controller
thresholds. Start the model server separately, using its existing environment and
checkpoint configuration. Stop an existing robot client before launching the
debug client; two clients must not own the robot at once.

## Prepare the checkout

From the debug worktree root:

```bash
export PYTHONPATH="$PWD:$PWD/phyai-robot/src${PYTHONPATH:+:$PYTHONPATH}"
npm --prefix tools/phyai_debug/web ci
npm --prefix tools/phyai_debug/web run build
python -m tools.phyai_debug --help
```

Use the robot environment's interpreter for recording. The tool uses NumPy,
Pillow, PyYAML, OmegaConf, and pyzmq. Recording also needs the existing client's
ROS message packages and H.264 decoder dependencies. If you are preparing a
separate offline test environment, install the CPU dependencies there:

```bash
python -m pip install numpy pillow pyyaml omegaconf pyzmq pytest
```

Do not run root `uv sync` in a robot environment merely to
install this tool: the model and robot environments remain separate.

The frontend bundle is local. Playback does not load scripts or robot assets
from a CDN. `node_modules` and generated bundles are not committed.

## Try playback without a robot

```bash
python -m tools.phyai_debug demo --output ./traces/demo
python -m tools.phyai_debug view ./traces/demo --host 127.0.0.1 --port 8770
```

Open `http://127.0.0.1:8770`. Demo data is explicitly synthetic and should never
be used as evidence about controller tracking or model accuracy.

For a viewer running on a remote robot computer, keep its loopback binding and
forward the port from your laptop:

```bash
ssh -N -L 8770:127.0.0.1:8770 ROBOT_HOST
```

The viewer has no authentication. A non-loopback binding exposes recorded
images, state, configuration, and model provenance to reachable clients.

## Configure recording

Copy `tools/phyai_debug/examples/trace.yaml` to a local configuration file. Keep
the model's existing `tianji_client.yaml` as the source of robot topics,
calibration, endpoint, execution timing, and safety thresholds.

| Trace setting | Purpose |
| --- | --- |
| `output_root` | Parent directory for new runs; an existing run is never overwritten |
| `queue_capacity` | Maximum queued numeric and lifecycle events |
| `image_capacity` | Separate capacity for image work so images cannot consume the event budget |
| `flush_timeout_s` | Maximum wait for recorder shutdown after client cleanup |
| `robot_assets` | Optional directory containing `robot.urdf` and its mesh tree |
| `model_bundle` | Optional model directory; copy JSON/YAML provenance, never weights |
| `capture_video` | Save compressed H.264 packets in addition to exact request images |
| `passive_ros` | Subscribe to mux output, control modes, and DAgger events; client callbacks retain continuous feedback |

Paths in the trace YAML are relative to that YAML file's directory. `--output`
selects a new run directory explicitly. Extra client arguments use the original
client's dotted overrides; they are not trace settings.

Set `robot_assets` to your installed console's `assets` directory if you want the
same robot geometry as its live 3D view. Asset copying validates paths and keeps
a local snapshot with the trace. It does not copy model checkpoint weights.
Without a robot asset snapshot, numeric and image inspection still works.

Use a disk with sufficient free space. Request images are lossless snapshots of
the arrays sent to the policy. Continuous compressed video adds context; it is
not a substitute for those exact images.

## Check, then record

First inspect configuration and compatibility without contacting robot control:

```bash
python -m tools.phyai_debug check \
  --client-config /path/to/model/tianji_client.yaml \
  --trace-config ./debug.yaml
```

This check does not prove that a ROS topic is publishing, that the server can
load its checkpoint, or that the control loop will meet deadlines.

At the robot, source the same ROS setup and activate the same interpreter that
you use for the ordinary client. With the model server ready, run:

```bash
python -m tools.phyai_debug run \
  --client-config /path/to/model/tianji_client.yaml \
  --trace-config ./debug.yaml \
  --output ./traces/test-001
```

**This command starts real robot control after the original warmup and activation
sequence.** Stay at the robot with the emergency stop available. `Ctrl+C` follows
the original client's stop and cleanup path; the recorder then flushes pending
events. Detaching from tmux does not stop the client.

Open the resulting trace:

```bash
python -m tools.phyai_debug view ./traces/test-001 --port 8770
```

## Read the trace

1. Check recording completeness before interpreting a run. Queue overflow,
   writer failure, and unfinished shutdown are recorded explicitly. A complete
   recorder shutdown does not establish loss-free DDS delivery.
2. Play or seek the shared timeline. RGB, joint plots, error plots, and the 3D
   view follow the selected time. Use request navigation to inspect an inference
   step and the images it actually consumed.
3. Compare actual feedback with the model target and the published command.
   They are different stages. A model result can be clipped by the adapter,
   interpolated by the scheduler, or ignored by a mux during human takeover.
4. Select a request to inspect its full action horizon. Only the retained
   execution prefix is scheduled. A discarded prediction is not a tracking
   failure.
5. Inspect age curves against the recorded freshness threshold. The gate checks
   callback-relative age before inference. The observation may be older by the
   time its action is executed.

The default overlays use blue for actual state and orange for a target, with
distinct line styles. Choose joint curves rather than comparing an unlabeled
array of numbers. Grippers have separate units and plots.

In the 3D view, time mode compares feedback and the corresponding command.
Chunk inspection compares the chosen predicted action against the request
observation. URDF forward kinematics renders the selected joint positions; it
does not simulate dynamics. The current Tianji console URDF has no movable
gripper joints, so its grippers remain numeric plots.

## Timing and action semantics

- `received_at_ns` is the client ROS callback timestamp on the monotonic clock.
  It is not a hardware capture timestamp.
- `source_stamp_ns` preserves an available ROS header separately. A missing
  header remains unknown; the tool does not manufacture one for gripper feedback.
- A 16D state combines samples with different arrival times. Inspect each source
  and the oldest required sample, not a single timestamp assigned to the vector.
- Client-side action traces are policy RPC outputs, normally physical targets
  after the server's postprocessor. They are not the model's normalized internal
  tensors. A joint-delta model already restored to absolute targets must not
  receive a second delta-to-absolute conversion in the viewer.
- A successful `publish()` call is not proof of actuator receipt. Compare
  client publication, observed mux output, and feedback separately.

The [design](DESIGN.md) describes collection points, correlation, storage, and
failure handling.

## Validation boundaries

The automated checks cover serialization, query bounds, path handling,
request/chunk association, configuration, and offline integration. They must
not enable robot control. CPU fixtures and a working viewer do not prove
200 Hz timing on your hardware or model success on a task.

Before a supervised trial, confirm that the viewer can open a synthetic trace,
the debug check accepts the intended checkout, and the client configuration
still contains the expected topics, calibration, and timing. Keep existing
control deadlines unchanged while measuring the recorder's added cost.
