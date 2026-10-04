# PhyAI gateway

The gateway connects existing clients to registered gRPC model servers. It handles protocol conversion and routing; backends own model preprocessing, tensor shapes, and action semantics. LeRobot and RLinf clients keep their existing request and response formats.

From the workspace root:

```bash
uv run phyai-gateway
```

The registry and inference RPCs listen on port 50111; HTTP listens on port 30000. Backends register a model name and endpoint, then send heartbeats. The gateway selects the server with the fewest in-flight requests for that model and excludes servers that miss heartbeats for 15 seconds. One endpoint can register several model names.

`ModelInference.Infer` accepts a registered `model_name`, a `request_id`, and model-defined named `inputs` tensors. Backends return the same request ID and named `outputs`. The optional `images`, `robot_state`, `instruction`, and `actions` fields support robot policies. `extensions_json` carries model metadata. See [model_inference.proto](phyai_gateway/proto/model_inference.proto) for dtypes and wire fields.

`GET /health` reports gateway liveness. `POST /v1/actions/generations` accepts MessagePack with NumPy arrays in either client format:

- Flat requests use `model_name` or `model`, `observation`, optional `metadata`, and optional `requested_action_horizon`. Sensors retain the names `main_images`, `wrist_images`, and `extra_view_images`; a stack of cameras becomes `wrist_images.0`, `wrist_images.1`, etc. Missing sensors are allowed. State width, camera count, and action width come from the model. The response contains `actions` shaped `[batch, horizon, action_dim]`.
- RLinf's SGLang clients use `input`, `parameters`, and `runtime`. NumPy observation fields become named `inputs` tensors; remaining input fields and parameters stay in `extensions_json`. The response preserves the `data[0].action.values` envelope. For a client that omits the model field, configure its backend with `--http-model MODEL` on the gateway.

For LeRobot's original async client:

```bash
uv run --package phyai-gateway --extra lerobot phyai-gateway --lerobot
```

Register the backend under the client's existing `pretrained_name_or_path` value. The adapter derives state order and camera names from `RemotePolicyConfig.lerobot_features`, applies camera `rename_map` entries, and passes policy configuration to the backend in `extensions_json.lerobot`. Set `--lerobot-fps` to the client's control rate; the default is 30. Original pickle classes and Torch action tensors remain part of this protocol, so the optional LeRobot dependency is needed when this adapter is enabled. Use it with trusted clients.

The separate RobotInference service is enabled with `--robot-model MODEL`. Its backend accepts the fixed image format in [robot.proto](phyai_gateway/proto/robot.proto). State is packed as joint angles, joint velocities, then `x, y, z, roll, pitch, yaw`; `extensions_json.robot` records `joint_angle_count` and `joint_velocity_count`. The backend returns `[1, joint_count + 1]` FLOAT32 actions: joint target angles followed by gripper opening in `[0, 1]`.

The bundled PI0.5 example uses a local LeRobot checkpoint and tokenizer:

```bash
uv run python phyai-gateway/examples/inference_server_pi0.5.py \
  --checkpoint-dir /path/to/checkpoint \
  --tokenizer-dir /path/to/tokenizer
```

The example listens on port 50063 and registers as `pi05`. For a remote gateway, set `--gateway-registry HOST:50111` and `--advertised-endpoint HOST:50063`. Its defaults match the flat HTTP demo: camera names `main_images` and `wrist_images`, 360 × 360 RGB images, eight state values, and seven action values. These are example settings, not gateway requirements. Set `--image-names` in checkpoint camera order and `--state-dim`, `--action-dim`, and `--max-batch-size` for the checkpoint. For LeRobot, also set `--model-name` to the client's pretrained identifier and use its camera names after renaming. The backend must produce actions in the units and order expected by that robot.

Run the HTTP demo with:

```bash
uv run --package phyai-gateway --extra benchmark \
  python phyai-gateway/examples/load_test.py --requests 10 --warmup 0
```

To regenerate the checked-in bindings:

```bash
uv run --package phyai-gateway --extra codegen \
  python phyai-gateway/phyai_gateway/scripts/generate_proto.py
scripts/run_pre_commit.sh
```
