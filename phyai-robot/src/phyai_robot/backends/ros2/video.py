"""H.264 camera streams carried as sensor_msgs/CompressedImage; needs the 'video' extra.

Design & Problem Statement:
    Industrial robot platforms (like Tianji / Marvin Pro) often provide multi-camera arrays
    at high native resolutions (e.g. four 2560x1984 GMSL sensors).
    - Uncompressed transmission over DDS UDP sockets generates hundreds of megabytes per second
      (e.g. ~686 MB/s), fragmenting each frame into ~120 UDP packets. Under default Linux network
      socket buffers (net.core.rmem_max = 208 KB), packet loss exceeds 98%, causing delivered
      raw frame rates to plummet to ~0.5 - 1.8 fps.
    - Conversely, hardware-encoded H.264 video streams reduce bandwidth to ~1 MB/s (~30 fps).
      However, H.264 is a stateful streaming format where predicted frames (P-frames) depend
      strictly on earlier reference pictures.

The "Silent Corruption" Trap:
    When an intermediate H.264 reference packet is dropped over lossy transport, standard FFmpeg
    decoders continue decoding without raising an exception, outputting visually distorted,
    corrupted macroblock artifacts for up to several seconds until the next keyframe (IDR).
    Feeding corrupted artifacts into a Vision-Language-Action (VLA) policy causes severe command
    instability and actuator hazards.

Solution in this Module:
    1. `FrameNumTracker`: Lightweight Annex B stream parser inspecting NAL slice headers.
       It tracks H.264 `frame_num` continuity according to ITU-T H.264 specs. The instant a
       reference frame gap is detected, the stream is marked unsynchronized, and ALL decoded frames
       are withheld until the next clean IDR keyframe arrives.
    2. `H264Decoder`: Stateful PyAV/FFmpeg wrapper using SLICE multi-threading for zero frame-latency.
       Decodes every packet to preserve codec state, but withholds output during recovery periods.
       Callers see a paused observation (triggering safe snapshot aging) rather than corrupted images.
    3. `MosaicDecoder`: Decomposes multi-camera mosaic frames (e.g. 1920x1488 2x2 grid) into
       named logical camera tiles sharing a single, unified reception timestamp.
"""

from __future__ import annotations

import math
import time
from typing import Any
from collections.abc import Mapping, Callable, Iterator

import numpy as np
from numpy.typing import NDArray

# H.264 NAL Unit Types (ITU-T H.264 Table 7-1).
NAL_SLICE = 1  # Coded slice of a non-IDR picture.
NAL_IDR = 5  # Coded slice of an IDR (Instantaneous Decoding Refresh) picture.
NAL_SPS = 7  # Sequence Parameter Set.
NAL_PPS = 8  # Picture Parameter Set.

# H.264 profile_idc values that include chroma_format_idc and scaling matrices in SPS.
HIGH_PROFILES = frozenset(
    {44, 83, 86, 100, 110, 118, 122, 128, 134, 135, 138, 139, 244}
)


def nal_units(data: bytes) -> Iterator[bytes]:
    """Yield NAL units (header byte first) from an Annex B byte stream.

    Scans for 3-byte and 4-byte start codes (00 00 01 and 00 00 00 01).
    Trailing zeroes preceding subsequent start codes are stripped.
    """
    starts = []
    index = data.find(b"\x00\x00\x01")
    while index >= 0:
        starts.append(index + 3)
        index = data.find(b"\x00\x00\x01", index + 3)
    for position, start in enumerate(starts):
        end = starts[position + 1] - 3 if position + 1 < len(starts) else len(data)
        while end > start and data[end - 1] == 0:
            end -= 1  # Leading zeros of 4-byte start code belong to next boundary.
        if end > start:
            yield data[start:end]


def _rbsp(payload: bytes) -> bytes:
    """Remove emulation prevention bytes (0x000003 -> 0x0000) to recover raw byte sequence payload."""
    if b"\x00\x00\x03" not in payload:
        return payload
    out = bytearray()
    zeros = 0
    for byte in payload:
        if zeros >= 2 and byte == 3:
            zeros = 0
            continue
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


class _Bits:
    """Big-endian bit-level reader with Exponential-Golomb decoding routines."""

    def __init__(self, data: bytes) -> None:
        self._data, self._position = data, 0

    def bit(self) -> int:
        """Read a single bit."""
        if self._position >= 8 * len(self._data):
            raise ValueError("truncated H.264 header")
        value = (self._data[self._position >> 3] >> (7 - (self._position & 7))) & 1
        self._position += 1
        return value

    def bits(self, count: int) -> int:
        """Read `count` bits as an unsigned integer."""
        value = 0
        for _ in range(count):
            value = (value << 1) | self.bit()
        return value

    def unsigned_exp_golomb(self) -> int:
        """Decode an unsigned Exponential-Golomb integer."""
        zeros = 0
        while not self.bit():
            zeros += 1
            if zeros > 31:
                raise ValueError("invalid Exp-Golomb code")
        return (1 << zeros) - 1 + self.bits(zeros)

    def signed_exp_golomb(self) -> int:
        """Decode a signed Exponential-Golomb integer."""
        value = self.unsigned_exp_golomb()
        return (value + 1) // 2 if value & 1 else -(value // 2)

    # Shorthand aliases matching H.264 specification syntax notation.


def _parse_sps(nal: bytes) -> tuple[int, int, bool]:
    """Parse Sequence Parameter Set (SPS) NAL unit.

    Returns:
        tuple of (sps_id, log2_max_frame_num, separate_colour_plane_flag).
    """
    reader = _Bits(_rbsp(nal[1:]))
    profile = reader.bits(8)
    reader.bits(16)  # Constraint flags and level_idc.
    sps_id = reader.unsigned_exp_golomb()
    separate = False
    if profile in HIGH_PROFILES:
        chroma_format = reader.unsigned_exp_golomb()
        if chroma_format == 3:
            separate = bool(reader.bit())
        reader.unsigned_exp_golomb()  # bit_depth_luma_minus8
        reader.unsigned_exp_golomb()  # bit_depth_chroma_minus8
        reader.bit()  # qpprime_y_zero_transform_bypass_flag
        if reader.bit():  # seq_scaling_matrix_present_flag
            for list_index in range(12 if chroma_format == 3 else 8):
                if reader.bit():
                    last = following = 8
                    for _ in range(16 if list_index < 6 else 64):
                        if following:
                            following = (last + reader.signed_exp_golomb() + 256) % 256
                        last = following or last
    return sps_id, reader.unsigned_exp_golomb() + 4, separate


class FrameNumTracker:
    """Tracks H.264 slice frame_num continuity to detect lost reference pictures.

    Specification Invariant (H.264 Clause 7.4.3):
        When gaps_in_frame_num_value_allowed_flag is 0 (standard for low-latency live streams),
        each reference picture must have:
            frame_num == (prev_ref_frame_num + 1) % MaxFrameNum
        If frame_num jumps by more than 1, one or more reference pictures were lost in transport.
        Subsequent P-frames are corrupted until the next IDR keyframe arrives.
    """

    def __init__(self) -> None:
        self._sps: dict[
            int, tuple[int, bool]
        ] = {}  # sps_id -> (log2_max_frame_num, separate_plane)
        self._pps: dict[int, int] = {}  # pps_id -> sps_id
        self._previous_reference: int | None = None

    def update(self, data: bytes) -> bool | None:
        """Inspect access unit NAL headers and verify reference continuity.

        Args:
            data: Raw H.264 access unit bytes.

        Returns:
            True: If access unit contains a valid clean IDR keyframe (resets tracking).
            False: If access unit is continuous without detected reference packet loss.
            None: If reference gap detected, or stream joined before seeing SPS/PPS.
        """
        result: bool | None = False
        for nal in nal_units(data):
            nal_type, reference_idc = nal[0] & 0x1F, (nal[0] >> 5) & 3
            if nal_type == NAL_SPS:
                sps_id, log2_max_frame_num, separate = _parse_sps(nal)
                self._sps[sps_id] = (log2_max_frame_num, separate)
            elif nal_type == NAL_PPS:
                reader = _Bits(_rbsp(nal[1:]))
                pps_id = reader.unsigned_exp_golomb()
                self._pps[pps_id] = reader.unsigned_exp_golomb()
            elif nal_type in (NAL_SLICE, NAL_IDR):
                reader = _Bits(_rbsp(nal[1:32]))
                if (
                    reader.unsigned_exp_golomb()
                ):  # first_mb_in_slice != 0: only inspect first slice of picture.
                    continue
                reader.unsigned_exp_golomb()  # slice_type
                sps = self._sps.get(self._pps.get(reader.unsigned_exp_golomb(), -1))
                if sps is None:
                    self._previous_reference = None
                    return None  # Stream joined mid-GOP before parameter sets were received.

                log2_max_frame_num, separate = sps
                if separate:
                    reader.bits(2)  # colour_plane_id
                frame_num = reader.bits(log2_max_frame_num)

                if nal_type == NAL_IDR:
                    self._previous_reference = 0
                    result = True
                    continue

                previous = self._previous_reference
                if previous is None or frame_num not in (
                    previous,
                    (previous + 1) % (1 << log2_max_frame_num),
                ):
                    self._previous_reference = None
                    return None  # Frame loss detected!

                if reference_idc:
                    self._previous_reference = frame_num
        return result


class H264Decoder:
    """Decodes H.264 access units from CompressedImage into RGB with loss protection.

    Contract:
        - Output: Returns HWC uint8 RGB ndarray upon valid decode; returns None if the frame
          is corrupt, stream is unsynchronized, or rate-limiting throttles output.
        - Continuity: Decodes EVERY incoming packet regardless of whether it will be output,
          ensuring codec reference pictures remain current and intact.
        - Loss Protection: Uses FrameNumTracker to withhold output during packet loss bursts.
    """

    def __init__(self, *, max_rate_hz: float | None = None, threads: int = 1) -> None:
        if max_rate_hz is not None and not (
            math.isfinite(max_rate_hz) and max_rate_hz > 0
        ):
            raise ValueError("max_rate_hz must be finite and positive or None")
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        self._period_s = 0.0 if max_rate_hz is None else 1.0 / max_rate_hz
        self._threads = threads
        self._codec: Any = None
        self._tracker = FrameNumTracker()
        self._synchronized = False
        self._sequence = 0
        self._last_output = -math.inf
        self.stats = dict(messages=0, frames=0, outputs=0, gaps=0, errors=0)

    def _decoder(self) -> Any:
        """Instantiate PyAV H.264 decoder with low-latency slice multi-threading."""
        if self._codec is None:
            try:
                import av
            except ModuleNotFoundError as error:
                raise ImportError(
                    "H264Decoder requires PyAV; install phyai-robot[video]"
                ) from error
            codec = av.CodecContext.create("h264", "r")
            codec.thread_type = (
                "SLICE"  # SLICE threading avoids multi-frame buffering latency.
            )
            codec.thread_count = self._threads
            self._codec = codec
        return self._codec

    def __call__(self, message: Any) -> NDArray[np.uint8] | None:
        """Process one ROS CompressedImage message containing an H.264 access unit."""
        if "h264" not in str(message.format).lower():
            raise ValueError(f"expected an H.264 stream, got {message.format!r}")
        import av

        data = bytes(message.data)
        self.stats["messages"] += 1

        # Evaluate reference picture continuity.
        try:
            state = self._tracker.update(data)
        except ValueError:
            self._tracker = FrameNumTracker()
            state = None

        if state is None:
            if self._synchronized:
                self.stats["gaps"] += 1
            self._synchronized = False
        elif state:
            self._synchronized = True

        codec = self._decoder()
        self._sequence += 1
        packet = av.Packet(data)
        packet.pts = packet.dts = self._sequence

        try:
            frames = codec.decode(packet)
        except av.error.FFmpegError:
            self.stats["errors"] += 1
            self._synchronized = False
            return None

        self.stats["frames"] += len(frames)
        if not self._synchronized:
            return None  # Withhold output until recovery at next IDR.

        frame = next((f for f in frames if f.pts == self._sequence), None)
        if frame is None or frame.is_corrupt:
            return None

        # Apply output rate throttling if configured.
        now = time.monotonic()
        if now - self._last_output < self._period_s:
            return None
        self._last_output = now
        self.stats["outputs"] += 1
        return frame.to_ndarray(format="rgb24")


class MosaicDecoder:
    """Decodes a composite multi-camera mosaic stream and slices it into named tiles.

    Grid Slicing:
        Decomposes a single decoded video frame of shape (H, W, 3) into an M x N grid of
        equal tiles (height = H // M, width = W // N). Each tile is cropped and optionally
        transformed (e.g. aspect ratio stretching or letterboxing to 224x224).
    """

    def __init__(
        self,
        *,
        tiles: Mapping[str, tuple[int, int]],  # field_name -> (row_idx, col_idx)
        grid: tuple[int, int] = (2, 2),  # (rows, columns)
        transform: Callable[[NDArray[np.uint8]], NDArray[np.uint8]] | None = None,
        decoder: Callable[[Any], NDArray[np.uint8] | None] | None = None,
    ) -> None:
        rows, columns = grid
        if rows < 1 or columns < 1:
            raise ValueError("grid must have positive dimensions")
        if not tiles:
            raise ValueError("at least one tile is required")
        for field, (row, column) in tiles.items():
            if not (0 <= row < rows and 0 <= column < columns):
                raise ValueError(f"{field}: tile ({row}, {column}) is outside the grid")
        if len(set(map(tuple, tiles.values()))) != len(tiles):
            raise ValueError("each tile can belong to only one field")

        self._tiles = {field: (int(r), int(c)) for field, (r, c) in tiles.items()}
        self._grid = (rows, columns)
        self._transform = transform
        self.decoder = H264Decoder() if decoder is None else decoder

    @property
    def fields(self) -> tuple[str, ...]:
        """Return the tuple of field names produced by this mosaic decoder."""
        return tuple(self._tiles)

    def __call__(self, message: Any) -> dict[str, NDArray[np.uint8]] | None:
        """Decode mosaic frame and return dictionary of transformed tile images."""
        frame = self.decoder(message)
        if frame is None:
            return None

        rows, columns = self._grid
        if frame.shape[0] % rows or frame.shape[1] % columns:
            raise ValueError(
                f"frame {frame.shape[:2]} does not split into {self._grid}"
            )

        height, width = frame.shape[0] // rows, frame.shape[1] // columns
        tiles = {}
        for field, (row, column) in self._tiles.items():
            tile = frame[
                row * height : (row + 1) * height, column * width : (column + 1) * width
            ]
            tiles[field] = (
                np.ascontiguousarray(tile)
                if self._transform is None
                else self._transform(tile)
            )
        return tiles
