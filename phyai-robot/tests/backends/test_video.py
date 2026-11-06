"""H.264 decoding with loss detection, using real libx264 streams and no ROS."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

av = pytest.importorskip("av")
from phyai_robot.backends.ros2.video import (
    H264Decoder,
    MosaicDecoder,
    FrameNumTracker,
    nal_units,
)

pytestmark = pytest.mark.skipif(
    "libx264" not in av.codecs_available, reason="needs the libx264 encoder"
)
HEIGHT, WIDTH = 64, 96


def frame_value(index):
    return 8 + 6 * index


def encode(count=30, gop=10, profile="baseline"):
    """Encode flat-colored frames whose gray level identifies their index."""
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height, encoder.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    encoder.time_base = Fraction(1, 30)
    encoder.options = {
        "g": str(gop),
        "keyint_min": str(gop),
        "bf": "0",
        "preset": "ultrafast",
        "tune": "zerolatency",
        "profile": profile,
        "sc_threshold": "0",
    }
    packets = []
    for index in range(count):
        image = np.full((HEIGHT, WIDTH, 3), frame_value(index), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = index
        packets += [bytes(packet) for packet in encoder.encode(frame)]
    packets += [bytes(packet) for packet in encoder.encode(None)]
    assert len(packets) == count
    return [SimpleNamespace(format="h264", data=packet) for packet in packets]


def decode_all(decoder, messages, skip=()):
    outputs = {}
    for index, message in enumerate(messages):
        if index in skip:
            continue
        image = decoder(message)
        outputs[index] = None if image is None else int(round(image.mean()))
    return outputs


@pytest.mark.parametrize("profile", ["baseline", "high"])
def test_every_frame_is_returned_with_its_own_message(profile):
    outputs = decode_all(H264Decoder(), encode(profile=profile))
    for index, value in outputs.items():
        assert value is not None and abs(value - frame_value(index)) <= 2


def test_lost_reference_frame_withholds_output_until_next_idr():
    decoder = H264Decoder()
    outputs = decode_all(decoder, encode(), skip={13})
    assert all(outputs[i] is not None for i in range(13))
    # FFmpeg would return frames 14..19 without any error flag.
    assert all(outputs[i] is None for i in range(14, 20))
    assert all(outputs[i] is not None for i in range(20, 30))
    assert decoder.stats["gaps"] == 1


def test_stream_joined_mid_gop_waits_for_the_first_idr():
    outputs = decode_all(H264Decoder(), encode(), skip={0, 1, 2})
    assert all(outputs[i] is None for i in range(3, 10))
    assert all(abs(outputs[i] - frame_value(i)) <= 2 for i in range(10, 30))


def test_lost_idr_withholds_the_whole_following_gop():
    outputs = decode_all(H264Decoder(), encode(), skip={10})
    assert all(outputs[i] is None for i in range(11, 20))
    assert all(outputs[i] is not None for i in range(20, 30))


def test_rate_limit_skips_conversion_but_keeps_decoding():
    decoder = H264Decoder(max_rate_hz=1e-3)
    outputs = decode_all(decoder, encode())
    assert sum(value is not None for value in outputs.values()) == 1
    assert decoder.stats["frames"] == 30


def test_frame_num_tracker_reports_idr_continuity_and_gaps():
    messages = encode(count=12, gop=6)
    tracker = FrameNumTracker()
    states = [tracker.update(m.data) for m in messages[:4]]
    assert states == [True, False, False, False]
    assert tracker.update(messages[5].data) is None  # Picture 4 was lost.
    assert tracker.update(messages[6].data) is True


def test_nal_units_split_annex_b_streams():
    data = b"\x00\x00\x00\x01\x67\xaa\x00\x00\x01\x68\xbb\x00\x00\x00\x01\x65\xcc"
    assert list(nal_units(data)) == [b"\x67\xaa", b"\x68\xbb", b"\x65\xcc"]


def test_non_h264_messages_and_bad_options_are_rejected():
    with pytest.raises(ValueError, match="H.264"):
        H264Decoder()(SimpleNamespace(format="jpeg", data=b""))
    with pytest.raises(ValueError):
        H264Decoder(max_rate_hz=0)
    with pytest.raises(ValueError):
        H264Decoder(threads=0)


def test_damaged_headers_are_treated_as_a_loss():
    decoder = H264Decoder()
    messages = encode()
    decoder(messages[0])
    assert decoder(SimpleNamespace(format="h264", data=b"\x00\x00\x01\x41")) is None
    assert decoder(messages[1]) is None  # Waits for the next IDR.


def test_mosaic_splits_tiles_by_grid_cell_and_transforms_them():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    for row in range(2):
        for column in range(2):
            frame[row * 2 : row * 2 + 2, column * 3 : column * 3 + 3] = (
                10 * row + column
            )
    mosaic = MosaicDecoder(
        tiles={"a": (0, 0), "b": (1, 1), "c": (1, 0)},
        decoder=lambda message: message,
        transform=lambda tile: tile + 1,
    )
    tiles = mosaic(frame)
    assert mosaic.fields == ("a", "b", "c")
    assert {k: int(v[0, 0, 0]) for k, v in tiles.items()} == {"a": 1, "b": 12, "c": 11}
    assert tiles["a"].shape == (2, 3, 3)
    assert mosaic(None) is None
    with pytest.raises(ValueError, match="split"):
        mosaic(np.zeros((5, 6, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "tiles",
    [{}, {"a": (2, 0)}, {"a": (0, 0), "b": (0, 0)}],
)
def test_mosaic_rejects_invalid_layouts(tiles):
    with pytest.raises(ValueError):
        MosaicDecoder(tiles=tiles, decoder=lambda message: None)
