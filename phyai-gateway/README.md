# PhyAI gateway

The gateway routes RLinf HTTP requests to registered gRPC model servers. It selects the server with the fewest in-flight requests and excludes servers that miss heartbeats for 15 seconds.

From the workspace root:

```bash
uv run phyai-gateway
```

The registry listens on port 50111 and HTTP on port 30000. Start the PI0.5 example in another terminal with local LeRobot checkpoint and tokenizer directories:

```bash
uv run python phyai-gateway/examples/inference_server_pi0.5.py \
  --checkpoint-dir /path/to/checkpoint \
  --tokenizer-dir /path/to/tokenizer
```

The example listens on port 50063 and registers with the local gateway. For a remote gateway, set `--gateway-registry HOST:50111` and `--advertised-endpoint HOST:50063` to addresses the two processes can reach. Use `--max-batch-size` to configure batches larger than one.

`GET /health` reports gateway liveness. `POST /v1/actions/generations` accepts MessagePack with these fields:

- `model_name`: a registered model name, `pi05` by default in the example.
- `observation.main_images` and `observation.wrist_images`: NumPy uint8 arrays shaped `[batch, height, width, 3]`.
- `observation.states`: a float32 or float64 array shaped `[batch, state_dim]`.
- `observation.task_descriptions`: one instruction per sample.
- `requested_action_horizon`: the positive number of actions to return.
- `metadata`: `mode="eval"`, `batch_size`, integer `stage_id`, and boolean `reset`.

The response contains a NumPy `actions` array shaped `[batch, horizon, action_dim]`. The example uses LIBERO camera names (`agentview`, `robot0_eye_in_hand`), 360 × 360 images, eight state values, and seven action values. Use the checkpoint's matching model server settings. The array codec and a request example are in `examples/load_test.py`:

```bash
uv run --package phyai-gateway --extra benchmark \
  python phyai-gateway/examples/load_test.py --requests 10 --warmup 0
```

For LeRobot's async client, install the `lerobot` extra and run with `--lerobot`. Its pickle protocol requires trusted clients. Set `--lerobot-fps` to the client's control rate; the default is 30. The adapter expects LIBERO state and camera fields.

The separate RobotInference service is enabled with `--robot-model MODEL`. Its backend must accept the state and image format in `proto/robot.proto` and return six joint target angles followed by gripper opening in `[0, 1]`. The bundled LIBERO PI0.5 example does not implement that joint-control contract.

To regenerate the checked-in bindings:

```bash
uv run --package phyai-gateway --extra codegen \
  python phyai-gateway/phyai_gateway/scripts/generate_proto.py
scripts/run_pre_commit.sh
```
