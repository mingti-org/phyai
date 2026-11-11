# AgileX pi0.5 deployment

`ROBOT_ENVIRONMENTS.md` covers installation, configuration ownership, the RPC
lifecycle and the two-terminal startup. This page covers the AgileX side.

## Before starting the client

The camera and Piper drivers are site-specific and are not shipped by PhyAI.
Before running the client:

- check the physical emergency stop and the workspace, and stop every other
  command-producing client;
- check that both CAN interfaces are up (`ip -brief link show type can`);
  never reconfigure a live bus;
- start the camera and Piper drivers on the same ROS domain as
  `robot.domain_id` (default 0). The Piper launch may enable motors;
- confirm the topics below exist, the command topics have Piper subscribers
  and no other publishers:

```bash
ros2 topic info /joint_left_states --verbose
ros2 topic info /puppet/joint_left --verbose
```

## Conventions

- Cameras: `/camera_f/color/image_raw`, `/camera_l/color/image_raw`,
  `/camera_r/color/image_raw`, letterboxed on the robot to the model's image
  size (224) and sent as `image.front`, `image.left`, `image.right`.
- Feedback: `/puppet/joint_left` and `/puppet/joint_right`, reordered by
  `joint0` through `joint6`.
- State and action rows: left six joint radians + gripper meters, then right.
- The checkpoint's exported processor performs normalization, tokenization and
  joint-only delta anchoring on the server. Grippers are absolute.
- The server returns 50 targets; the client executes 25 by default at 30 Hz with
  200 Hz control. One write publishes both arms to `/joint_left_states` and
  `/joint_right_states`.
- No motor enable, homing or clipping. Stop means stop publishing, not disable.

All topics and QoS profiles are `robot.*` settings in `configs/agilex.yaml`.

## Files

- `agilex.py`, `configs/agilex.yaml`: robot client entry point and defaults.
- `robots/agilex.py`: configuration, schemas, `VectorAdapter`, and the
  composition of one joint backend and one backend per camera.
- `robots/agilex_codecs.py`: JointState conversions, without ROS imports.
- `configs/pi05_agilex.yaml`: model server settings for this checkpoint.

For saved-data inference without ROS, use `examples/pi05/run_enactive.py`.
