# PhyAI gateway

PhyAI gateway gives your inference client one address for model servers running on one or more machines. Start each server separately, give the gateway its address, and point your client at the gateway.

## Start the gateway

Run the commands below from the repository root. If you need to start a model server, follow the [server setup guide](https://phyai.mintlify.app/deployment/server-gateway) ([中文](https://phyai.mintlify.app/zh/deployment/server-gateway)) to prepare its YAML configuration, then run:

```bash
uv run --package phyai phyai server server.yaml
```

In another terminal, connect the gateway to your server. This example uses a server whose YAML sets `model_name: pi05` and `port: 50063`:

```bash
uv run --package phyai-gateway phyai-gateway \
  --backend pi05 127.0.0.1:50063
```

Use the server machine's reachable IP address when it runs elsewhere. The gateway listens on HTTP port `30000` and gRPC port `50111`. You can choose different addresses with `--http-host`, `--http-port`, and `--gateway`.

Check that the server is ready before connecting your client:

```bash
curl http://127.0.0.1:30000/v1/backends
```

Wait for its `healthy` field to become `true`. The gateway keeps checking registered servers and resumes sending requests when an unavailable server recovers.

## Connect your client

For clients using the MessagePack action API, use `http://GATEWAY_HOST:30000` as the server URL. Requests go to `/v1/actions/generations`. If your client does not send a model name, choose one when starting the gateway:

```bash
uv run --package phyai-gateway phyai-gateway \
  --backend pi05 127.0.0.1:50063 --http-model pi05
```

The bundled PI0.5 server accepts the `observation` request format. RLinf clients that send the SGLang `input` / `parameters` / `runtime` format still need a matching model server adapter; they cannot use this PI0.5 configuration directly yet.

For LeRobot's async client, enable its adapter:

```bash
uv run --package phyai-gateway --extra lerobot phyai-gateway \
  --lerobot --lerobot-fps 30 \
  --backend your-org/your-policy 127.0.0.1:50063
```

Replace `your-org/your-policy` with the client's existing `pretrained_name_or_path`, and set the server YAML's `server.model_name` to the same value. Point the client's `server_address` at `GATEWAY_HOST:50111`, and match `--lerobot-fps` to its control rate.

The [setup guide](https://phyai.mintlify.app/deployment/server-gateway#connect-a-lerobot-client) explains how to match the server's camera and action settings to your robot. The client's request and response formats stay unchanged. Enable LeRobot only for clients you trust, since its protocol uses pickle.

## Use more than one server

Repeat `--backend` for each server. Give replicas of the same model the same name:

```bash
uv run --package phyai-gateway phyai-gateway \
  --backend pi05 10.0.0.11:50063 \
  --backend pi05 10.0.0.12:50063
```

The gateway sends each request to a healthy replica with the fewest active requests. Use this setup for stateless inference: successive requests from one client can reach different servers. To serve different models, register each under its own name.

You can also add a running server without restarting the gateway:

```bash
curl -X POST http://127.0.0.1:30000/v1/backends \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"pi05","endpoint":"10.0.0.13:50063"}'
```

To take a server out of rotation, find its `server_id` with `GET /v1/backends`, then remove it:

```bash
curl -X DELETE http://127.0.0.1:30000/v1/backends/SERVER_ID
```

Requests already running on that server can finish. Registrations last until the gateway exits, so keep your initial server list in the launch command.

## Regenerate protocol bindings

The internal gRPC interface is defined in [model_inference.proto](phyai_gateway/proto/model_inference.proto). If you edit it, regenerate the Python bindings:

```bash
uv run --package phyai-gateway --extra codegen \
  python phyai-gateway/phyai_gateway/scripts/generate_proto.py
scripts/run_pre_commit.sh
```
