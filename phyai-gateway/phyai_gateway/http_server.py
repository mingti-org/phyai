import logging

import grpc
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
logger = logging.getLogger(__name__)


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


def create_http_app(model_client, *, default_model=None):
    app = FastAPI(title="PhyAI Gateway HTTP API")
    rlinf_adapter = RLinfAdapter(model_client, default_model=default_model)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

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
            response_body = await run_in_threadpool(
                rlinf_adapter.infer_msgpack, bytes(body)
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
