"""Gateway batching for backends that report a request capacity."""

from __future__ import annotations

import json
import struct
import threading

import grpc
import pytest
from phyai_gateway.bindings import model_inference_pb2 as pb
from phyai_gateway.clients.model_inference import (
    ModelInferenceClient,
    InvalidBatchedResponse,
)
from phyai_gateway.services.model_registry import SelectedModelServer


class Registry:
    def __init__(self, max_batch_size):
        self.selected = SelectedModelServer(
            "backend", "backend:50063", "pi05", max_batch_size
        )
        self.acquired = 0
        self.released = 0

    def acquire_server(self, model_name):
        assert model_name == "pi05"
        self.acquired += 1
        return self.selected

    def release_server(self, server_id):
        assert server_id == "backend"
        self.released += 1

    def endpoints(self):
        return {self.selected.endpoint}


class Future:
    def __init__(self, response):
        self.response = response

    def result(self, timeout=None):
        return self.response

    def cancel(self):
        pass


class Stub:
    def __init__(self, *, bad_response=False, cancel_event=None, failure=None):
        self.requests = []
        self.timeouts = []
        self.bad_response = bad_response
        self.cancel_event = cancel_event
        self.failure = failure
        self.Infer = self

    def future(self, request, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self.cancel_event is not None and len(self.requests) == 1:
            self.cancel_event.set()
        if self.failure and len(self.requests) == 2:
            raise self.failure
        image = request.images[0]
        batch_size = image.shape[0] if len(image.shape) == 4 else 1
        image_values = [image.data[index * 12] for index in range(batch_size)]
        values = [float(value) for value in image_values for _ in range(3 * 2)]
        response = pb.InferenceResponse(
            request_id=request.request_id,
            actions=pb.Tensor(
                data=struct.pack(f"<{len(values)}f", *values),
                shape=[batch_size, 3, 2],
                dtype=pb.DATA_TYPE_FLOAT32,
            ),
            inference_time_us=10,
        )
        if self.bad_response and len(self.requests) == 2:
            response.request_id = "wrong-id"
        return Future(response)


def request_for(batch_size, *, two_cameras=False):
    instructions = [f"task-{index}" for index in range(batch_size)]
    images = [
        pb.Image(
            name=name,
            data=b"".join(bytes([index + offset]) * 12 for index in range(batch_size)),
            shape=[batch_size, 2, 2, 3],
            encoding=pb.IMAGE_ENCODING_RAW,
            dtype=pb.DATA_TYPE_UINT8,
            layout=pb.IMAGE_LAYOUT_HWC,
        )
        for name, offset in (
            [("main_images", 0), ("wrist_images", 100)]
            if two_cameras
            else [("main_images", 0)]
        )
    ]
    return pb.InferenceRequest(
        request_id="original",
        model_name="pi05",
        images=images,
        robot_state=pb.Tensor(
            data=struct.pack(f"<{batch_size * 2}f", *range(batch_size * 2)),
            shape=[batch_size, 2],
            dtype=pb.DATA_TYPE_FLOAT32,
        ),
        instructions=instructions,
        requested_action_horizon=3,
        extensions_json=json.dumps({"batch_size": batch_size}),
    )


def client_for(cap, stub):
    registry = Registry(cap)
    client = ModelInferenceClient(registry)
    client._stubs[registry.selected.endpoint] = stub
    return client, registry


@pytest.mark.parametrize(
    ("batch_size", "capacity", "sizes"),
    [
        (64, 32, [32, 32]),
        (33, 32, [32, 1]),
        (32, 32, [32]),
        (8, 32, [8]),
        (33, None, [33]),
    ],
)
def test_backend_capacity_preserves_batch_order(batch_size, capacity, sizes):
    stub = Stub()
    client, registry = client_for(capacity, stub)
    response = client.infer(request_for(batch_size, two_cameras=True), "pi05")
    assert [request.images[0].shape[0] for request in stub.requests] == sizes
    assert response.request_id == "original"
    assert list(response.actions.shape) == [batch_size, 3, 2]
    if capacity is not None and batch_size > capacity:
        assert struct.unpack(f"<{batch_size * 6}f", response.actions.data)[
            ::6
        ] == tuple(float(index) for index in range(batch_size))
    assert registry.acquired == registry.released == 1
    assert len(stub.timeouts) == len(sizes)
    assert all(0 < timeout <= 600 for timeout in stub.timeouts)
    if capacity is not None and batch_size > capacity:
        assert response.inference_time_us == 10 * len(sizes)
        assert len({request.request_id for request in stub.requests}) == len(sizes)
        offset = 0
        for chunk, size in zip(stub.requests, sizes):
            extensions = json.loads(chunk.extensions_json)
            assert extensions["batch_size"] == size
            assert "instructions" not in extensions
            assert list(chunk.instructions) == [
                f"task-{index}" for index in range(offset, offset + size)
            ]
            assert list(chunk.robot_state.shape) == [size, 2]
            assert [image.data[0] for image in chunk.images] == [offset, offset + 100]
            assert struct.unpack(f"<{size * 2}f", chunk.robot_state.data) == tuple(
                float(index) for index in range(offset * 2, (offset + size) * 2)
            )
            offset += size


def test_invalid_second_response_does_not_return_partial_actions():
    stub = Stub(bad_response=True)
    client, registry = client_for(32, stub)
    with pytest.raises(InvalidBatchedResponse):
        client.infer(request_for(33), "pi05")
    assert registry.released == 1


def test_cancellation_skips_remaining_chunks():
    cancelled = threading.Event()
    stub = Stub(cancel_event=cancelled)
    client, registry = client_for(32, stub)
    with pytest.raises(grpc.RpcError) as error:
        client.infer(request_for(64), "pi05", cancel_event=cancelled)
    assert error.value.code() == grpc.StatusCode.CANCELLED
    assert len(stub.requests) == 1
    assert registry.released == 1


def test_single_sample_robot_request_is_not_split():
    stub = Stub()
    client, registry = client_for(1, stub)
    request = request_for(1)
    request.images[0].shape[:] = [2, 2, 3]
    request.robot_state.shape[:] = [2]
    request.extensions_json = ""
    response = client.infer(request, "pi05")
    assert len(stub.requests) == 1
    assert list(stub.requests[0].instructions) == ["task-0"]
    assert response.request_id == "original"
    assert registry.released == 1


def test_backend_failure_does_not_retry_later_chunks():
    stub = Stub(failure=RuntimeError("backend failed"))
    client, registry = client_for(32, stub)
    with pytest.raises(RuntimeError, match="backend failed"):
        client.infer(request_for(96), "pi05")
    assert len(stub.requests) == 2
    assert registry.released == 1


def test_unsupported_batch_shape_is_rejected_before_forwarding():
    stub = Stub()
    client, registry = client_for(1, stub)
    request = request_for(2)
    request.images[0].encoding = pb.IMAGE_ENCODING_PNG
    with pytest.raises(grpc.RpcError) as error:
        client.infer(request, "pi05")
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert not stub.requests
    assert registry.released == 1


@pytest.mark.parametrize("instructions", [["task-0"], ["task-0", " "]])
def test_oversized_batch_requires_one_instruction_per_sample(instructions):
    stub = Stub()
    client, registry = client_for(1, stub)
    request = request_for(2)
    request.instructions[:] = instructions
    with pytest.raises(grpc.RpcError) as error:
        client.infer(request, "pi05")
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert not stub.requests
    assert registry.released == 1


def test_declared_oversized_batch_cannot_hide_behind_small_image():
    stub = Stub()
    client, registry = client_for(1, stub)
    request = request_for(2)
    request.images[0].shape[0] = 1
    with pytest.raises(grpc.RpcError) as error:
        client.infer(request, "pi05")
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert not stub.requests
    assert registry.released == 1
