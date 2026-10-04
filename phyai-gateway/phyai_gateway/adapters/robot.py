import itertools
import math
import struct
import time

import grpc

from phyai_gateway.bindings import (
    model_inference_pb2,
    robot_pb2,
    robot_pb2_grpc,
)
from phyai_gateway.clients.model_inference import abort_for_backend_error

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_CHANNELS = 3
IMAGE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS


class RobotAdapter(robot_pb2_grpc.RobotInferenceServicer):
    """Serve a backend that returns six joint targets and one gripper opening."""

    def __init__(self, model_client, model_name):
        self._model_client = model_client
        self._model_name = model_name
        self._request_ids = itertools.count(1)

    def Communicate(self, request_iterator, context):
        for sensor_data in request_iterator:
            request_id = f"gateway-{next(self._request_ids)}"
            request = self._build_request(sensor_data, request_id, context)
            try:
                response = self._model_client.infer(request, self._model_name)
            except Exception as error:
                abort_for_backend_error(context, error)
            yield self._build_action(response, request_id, context)

    @staticmethod
    def _build_request(sensor_data, request_id, context):
        if len(sensor_data.image_rgb) != IMAGE_BYTES:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"image_rgb must contain exactly {IMAGE_BYTES} bytes",
            )

        state = sensor_data.state
        values = [
            *state.joint_angles,
            *state.joint_velocities,
            state.x,
            state.y,
            state.z,
            state.roll,
            state.pitch,
            state.yaw,
        ]
        if not all(math.isfinite(value) for value in values):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Robot state must contain finite values",
            )
        try:
            state_data = struct.pack(f"<{len(values)}f", *values)
        except OverflowError:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "Robot state values must fit float32"
            )
        return model_inference_pb2.InferenceRequest(
            request_id=request_id,
            timestamp_ns=sensor_data.timestamp_ns,
            images=[
                model_inference_pb2.Image(
                    name="front",
                    data=sensor_data.image_rgb,
                    shape=[IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
                    dtype=model_inference_pb2.DATA_TYPE_UINT8,
                    encoding=model_inference_pb2.IMAGE_ENCODING_RAW,
                    layout=model_inference_pb2.IMAGE_LAYOUT_HWC,
                )
            ],
            robot_state=model_inference_pb2.Tensor(
                data=state_data,
                shape=[len(values)],
                dtype=model_inference_pb2.DATA_TYPE_FLOAT32,
            ),
            instruction=sensor_data.language_instruction,
            requested_action_horizon=1,
        )

    @staticmethod
    def _build_action(response, request_id, context):
        if response.request_id != request_id:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server response request_id does not match request",
            )
        actions = response.actions
        if actions.dtype != model_inference_pb2.DATA_TYPE_FLOAT32:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server actions dtype must be FLOAT32",
            )
        if list(actions.shape) != [1, 7]:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server actions must have shape [1, 7]: six joint targets and a gripper opening",
            )
        value_count = actions.shape[1]
        if len(actions.data) != value_count * 4:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server actions shape does not match data length",
            )

        values = struct.unpack(f"<{value_count}f", actions.data)
        if not all(math.isfinite(value) for value in values):
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server actions contain non-finite values",
            )
        if not 0.0 <= values[6] <= 1.0:
            context.abort(
                grpc.StatusCode.DATA_LOSS,
                "Model Server gripper opening must be in [0, 1]",
            )
        return robot_pb2.ActionCmd(
            target_joint_angles=values[:6],
            gripper_opening=values[6],
            timestamp_ns=time.time_ns(),
            inference_time=response.inference_time_us,
        )
