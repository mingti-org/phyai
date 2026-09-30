import threading

import grpc
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from phyai_gateway.adapters.rlinf import (
    RLinfAdapter,
    RLinfBackendResponseError,
    RLinfPayloadError,
    RLinfWireError,
)
from phyai_gateway.clients.model_inference import NoHealthyModelServerError

MAX_MESSAGE_BYTES = 100 * 1024 * 1024


def _error_response(status_code, code, message):
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _backend_error_response(error):
    status = error.code()
    if status in (grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.OUT_OF_RANGE):
        return _error_response(422, "backend_rejected_request", error.details())
    if status == grpc.StatusCode.DEADLINE_EXCEEDED:
        return _error_response(504, "backend_timeout", error.details())
    if status in (
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.CANCELLED,
    ):
        return _error_response(503, "backend_unavailable", error.details())
    return _error_response(502, "backend_error", error.details())


def create_http_app(model_client):
    app = FastAPI(title="Robot Gateway HTTP API")
    rlinf_adapter = RLinfAdapter(model_client)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/inference/{client_type}")
    async def receive_inference(client_type: str, request: Request):
        body = await request.body()
        print(
            f"HTTP request received: client_type={client_type}, "
            f"content_type={request.headers.get('content-type')}, "
            f"body_bytes={len(body)}, ",
            flush=True,
        )
        return JSONResponse(
            status_code=202,
            content={
                "status": "received",
                "client_type": client_type,
                "body_bytes": len(body),
            },
        )

    @app.post("/v1/actions/generations")
    async def receive_rlinf_observation(request: Request):
        content_type = request.headers.get("content-type", "").split(";", 1)[0]
        if content_type.lower() != "application/msgpack":
            return _error_response(
                415,
                "unsupported_media_type",
                "Content-Type must be application/msgpack",
            )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_MESSAGE_BYTES:
                    return _error_response(413, "payload_too_large", "request is too large")
            except ValueError:
                return _error_response(400, "invalid_content_length", "invalid Content-Length")

        body = await request.body()
        if not body:
            return _error_response(400, "invalid_msgpack", "request body is empty")
        if len(body) > MAX_MESSAGE_BYTES:
            return _error_response(413, "payload_too_large", "request is too large")
        try:
            response_body = await run_in_threadpool(rlinf_adapter.infer_msgpack, body)
        except RLinfWireError as error:
            return _error_response(400, "invalid_msgpack", str(error))
        except RLinfPayloadError as error:
            return _error_response(422, "invalid_payload", str(error))
        except NoHealthyModelServerError as error:
            return _error_response(503, "backend_unavailable", str(error))
        except grpc.RpcError as error:
            return _backend_error_response(error)
        except RLinfBackendResponseError as error:
            return _error_response(502, "invalid_backend_response", str(error))
        except Exception:
            return _error_response(500, "internal_error", "Gateway inference failed")

        return Response(content=response_body, media_type="application/msgpack")

    return app


class HTTPServer:
    def __init__(self, host: str, port: int, model_client):
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_http_app(model_client),
                host=host,
                port=port,
                log_level="info",
            )
        )
        self._thread = threading.Thread(
            target=self._server.run,
            name="gateway-http",
            daemon=True,
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.should_exit = True
        self._thread.join()
