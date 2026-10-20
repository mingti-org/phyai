"""Protocol-v2 messages: one UTF-8 JSON frame followed by raw numeric frames.

JSON describes shape, dtype (including byte order), field names and operation
metadata. Each following frame is one array in C order. Both peers validate the
protocol version and byte counts before reconstructing arrays, so they need no
matching Python object layout, NumPy ABI, Torch or ROS installation.

Requests and replies carry named model arrays, such as camera images and a
state vector in, and an action matrix out. Robot field names, units and
receive timestamps never cross the wire.
"""

from __future__ import annotations

import json
import math
from typing import Any
from dataclasses import field, dataclass
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from ..types import Schema, FeatureSpec
from ..validation import validate_schema, validate_values

PROTOCOL_VERSION = 2
MAX_BYTES = 128 * 1024 * 1024


def schema_to_json(schema: Schema) -> dict[str, Any]:
    """Normalize a schema for exact JSON comparison on either peer."""
    validate_schema(schema)
    return {
        key: dict(
            shape=list(s.shape),
            dtype=np.dtype(s.dtype).str,
            unit=s.unit,
            names=None if s.names is None else list(s.names),
            frame=s.frame,
        )
        for key, s in schema.items()
    }


def schema_from_json(value: Mapping[str, Any]) -> dict[str, FeatureSpec]:
    schema = {
        k: FeatureSpec(
            tuple(v["shape"]),
            v["dtype"],
            v["unit"],
            None if v["names"] is None else tuple(v["names"]),
            v["frame"],
        )
        for k, v in value.items()
    }
    validate_schema(schema)
    return schema


@dataclass(frozen=True)
class PolicySpec:
    """The model's input/output arrays plus descriptive metadata.

    Metadata is application-owned (for example horizon, image size and task);
    clients use it to configure themselves before opening hardware.
    """

    inputs: Schema
    outputs: Schema
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            inputs=schema_to_json(self.inputs),
            outputs=schema_to_json(self.outputs),
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicySpec:
        return cls(
            schema_from_json(value["inputs"]),
            schema_from_json(value["outputs"]),
            dict(value["metadata"]),
        )


def pack(meta: Mapping[str, Any], arrays: Any = ()) -> list[Any]:
    """Build a bounded multipart message without copying array data.

    Array frames are byte views of C-contiguous arrays; send them with
    copy=False and do not modify the arrays until the send completes.
    Object/string/complex arrays and non-finite JSON metadata are rejected.
    """
    frames, descriptors = [], []
    for value in arrays:
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
            raise ValueError("wire values must be real numeric numpy arrays")
        descriptors.append(dict(shape=list(value.shape), dtype=value.dtype.str))
        frames.append(np.ascontiguousarray(value).reshape(-1).view(np.uint8).data)
    header = json.dumps(
        dict(meta, protocol=PROTOCOL_VERSION, arrays=descriptors),
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if len(header) + sum(frame.nbytes for frame in frames) > MAX_BYTES:
        raise ValueError("policy message exceeds size limit")
    return [header, *frames]


def unpack(frames: Any) -> tuple[dict[str, Any], list[NDArray[Any]]]:
    """Validate framing and return metadata plus arrays viewing the frames.

    The shape/size check precedes reconstruction. Frames received as bytes are
    immutable, so the arrays are read-only views and need no further copy.
    """
    if not frames or sum(map(len, frames)) > MAX_BYTES:
        raise ValueError("invalid policy message size")
    header = json.loads(frames[0])
    if not isinstance(header, dict):
        raise ValueError("policy metadata must be a JSON object")
    if (
        type(header.get("protocol")) is not int
        or header["protocol"] != PROTOCOL_VERSION
    ):
        raise ValueError("policy protocol version mismatch")
    descriptors = header.pop("arrays")
    if len(descriptors) != len(frames) - 1:
        raise ValueError("array frame count mismatch")
    arrays = []
    for desc, data in zip(descriptors, frames[1:]):
        dtype, shape = np.dtype(desc["dtype"]), tuple(desc["shape"])
        if dtype.kind not in "biuf" or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid array dtype or shape")
        if math.prod(shape) * dtype.itemsize != len(data):
            raise ValueError("array frame size mismatch")
        arrays.append(np.frombuffer(data, dtype=dtype).reshape(shape))
    return header, arrays


def values_payload(
    values: Mapping[str, NDArray[Any]], schema: Schema
) -> tuple[dict[str, Any], list[NDArray[Any]]]:
    """Validate named arrays against schema and order them for pack()."""
    validate_values(values, schema)
    keys = list(schema)
    return dict(keys=keys), [values[key] for key in keys]


def read_values(
    header: Mapping[str, Any], arrays: list[NDArray[Any]], schema: Schema
) -> dict[str, NDArray[Any]]:
    """Reject duplicate, missing or malformed fields and rebuild the mapping."""
    keys = header.get("keys")
    if (
        not isinstance(keys, list)
        or len(keys) != len(set(keys))
        or len(keys) != len(arrays)
    ):
        raise ValueError("array field mismatch")
    values = dict(zip(keys, arrays))
    validate_values(values, schema)
    return values
