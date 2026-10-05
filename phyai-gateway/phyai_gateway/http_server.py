import asyncio
from contextlib import suppress
import logging
import threading

import grpc
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, StrictStr

from phyai_gateway.adapters.rlinf import (
    RLinfAdapter,
    RLinfBackendResponseError,
    RLinfPayloadError,
    RLinfWireError,
)
from phyai_gateway.clients.model_inference import NoHealthyModelServerError

MAX_MESSAGE_BYTES = 100 * 1024 * 1024
logger = logging.getLogger(__name__)


class BackendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: StrictStr
    endpoint: StrictStr


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


async def _infer_until_disconnect(request, adapter, body):
    cancelled = threading.Event()

    async def watch_disconnect():
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                cancelled.set()
                return

    watcher = asyncio.create_task(watch_disconnect())
    try:
        return await run_in_threadpool(
            adapter.infer_msgpack, body, cancel_event=cancelled
        )
    finally:
        cancelled.set()
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher


def create_http_app(model_client, *, default_model=None, registry=None):
    app = FastAPI(title="PhyAI Gateway HTTP API")
    rlinf_adapter = RLinfAdapter(model_client, default_model=default_model)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    if registry is not None:

        @app.post("/v1/backends", status_code=202)
        async def add_backend(request: BackendRequest):
            try:
                return registry.add_backend(request.model_name, request.endpoint)
            except ValueError as error:
                return _error_response(422, "invalid_backend", str(error))

        @app.get("/v1/backends")
        async def list_backends():
            return {"backends": registry.list_backends()}

        @app.delete("/v1/backends/{server_id}", status_code=204)
        async def remove_backend(server_id: str):
            if not registry.remove_backend(server_id):
                return _error_response(
                    404, "backend_not_found", "backend is not registered"
                )
            return Response(status_code=204)

    @app.post("/v1/actions/generations")
    async def receive_rlinf_observation(request: Request):
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip()
        if content_type.lower() != "application/msgpack":
            return _error_response(
                415,
                "unsupported_media_type",
                "Content-Type must be application/msgpack",
            )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                length = int(content_length)
                if length < 0:
                    raise ValueError
                if length > MAX_MESSAGE_BYTES:
                    return _error_response(
                        413, "payload_too_large", "request is too large"
                    )
            except ValueError:
                return _error_response(
                    400, "invalid_content_length", "invalid Content-Length"
                )

        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_MESSAGE_BYTES:
                return _error_response(413, "payload_too_large", "request is too large")
            body.extend(chunk)
        if not body:
            return _error_response(400, "invalid_msgpack", "request body is empty")
        try:
            response_body = await _infer_until_disconnect(
                request, rlinf_adapter, bytes(body)
            )
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
            logger.exception("Gateway inference failed")
            return _error_response(500, "internal_error", "Gateway inference failed")

        return Response(content=response_body, media_type="application/msgpack")

    return app
