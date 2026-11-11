"""VectorAdapter maps named robot fields to model vectors without a model stack."""

import time

import numpy as np
import pytest
from phyai_robot import (
    Sample,
    StateSlot,
    CommandSlot,
    FeatureSpec,
    Observation,
    VectorAdapter,
    InferenceContext,
)

OBSERVATIONS = {
    "cam": FeatureSpec((4, 4, 3), "uint8"),
    "q": FeatureSpec((2,), "float64", unit="rad"),
    "grip": FeatureSpec((3,), "float32"),
}
ACTIONS = {
    "q_target": FeatureSpec((2,), "float64", unit="rad"),
    "grip_target": FeatureSpec((1,), "float32"),
}


def adapter(**overrides):
    values = dict(
        observation_schema=OBSERVATIONS,
        action_schema=ACTIONS,
        cameras={"main": "cam"},
        state=[StateSlot("q"), StateSlot("grip", index=(0,), limits=(1.0, 3.0))],
        commands=[CommandSlot("q_target"), CommandSlot("grip_target", clip=(0.0, 1.0))],
    )
    values.update(overrides)
    return VectorAdapter(**values)


def observation():
    now = time.monotonic_ns()
    camera = np.full((4, 4, 3), 7, dtype=np.uint8)
    camera.flags.writeable = False
    return Observation(
        {
            "cam": Sample(camera, now),
            "q": Sample(np.array([0.5, -0.5]), now),
            "grip": Sample(np.array([2.0, 9.0, 9.0], dtype=np.float32), now),
        }
    )


def test_request_orders_state_and_passes_images_without_copy():
    obs = observation()
    vector = adapter()
    request = vector.to_request(obs, InferenceContext())
    assert set(request) == {"image.main", "state"}
    assert request["image.main"] is obs.samples["cam"].value
    np.testing.assert_allclose(request["state"], [0.5, -0.5, 0.5])
    assert request["state"].dtype == np.float32
    assert vector.state_dim == 3 and vector.action_dim == 3
    assert vector.request_schema["state"] == FeatureSpec((3,), "float32")
    assert vector.request_schema["image.main"] == OBSERVATIONS["cam"]


def test_actions_split_rows_cast_dtypes_and_clip():
    rows = np.array([[1.0, 2.0, 1.5], [3.0, 4.0, -0.5]])
    chunk = adapter().to_actions({"actions": rows}, observation())
    assert len(chunk.actions) == 2
    np.testing.assert_array_equal(chunk.actions[0]["q_target"], [1.0, 2.0])
    assert chunk.actions[0]["q_target"].dtype == np.float64
    assert chunk.actions[0]["grip_target"].dtype == np.float32
    assert chunk.actions[0]["grip_target"][0] == 1.0
    assert chunk.actions[1]["grip_target"][0] == 0.0


@pytest.mark.parametrize(
    "rows",
    [
        np.zeros((2, 4)),
        np.zeros(3),
        np.array([[0.0, np.nan, 0.0]]),
        np.zeros((1, 3), int),
    ],
)
def test_actions_reject_wrong_width_rank_or_values(rows):
    with pytest.raises(ValueError):
        adapter().to_actions({"actions": rows}, observation())


@pytest.mark.parametrize(
    "overrides",
    [
        {"cameras": {"main": "q"}},
        {"cameras": {"main": "missing"}},
        {"cameras": {}},
        {"state": [StateSlot("cam")]},
        {"state": [StateSlot("q", index=(2,))]},
        {"state": [StateSlot("q", limits=(1.0, 1.0))]},
        {"commands": [CommandSlot("q_target")]},
        {"commands": [CommandSlot("q_target")] * 2 + [CommandSlot("grip_target")]},
    ],
)
def test_invalid_layouts_fail_at_construction(overrides):
    with pytest.raises(ValueError):
        adapter(**overrides)


def test_resize_with_pad_matches_pil_letterbox():
    pytest.importorskip("PIL")
    from PIL import Image
    from phyai_robot.images import resize_with_pad

    image = np.random.default_rng(1).integers(0, 256, (30, 40, 3), dtype=np.uint8)
    result = resize_with_pad(image, 20)
    expected = Image.new("RGB", (20, 20))
    expected.paste(
        Image.fromarray(image).resize((20, 15), Image.Resampling.BILINEAR), (0, 2)
    )
    np.testing.assert_array_equal(result, np.array(expected))
    same = resize_with_pad(result, 20)
    np.testing.assert_array_equal(same, result)
    assert same is not result
    with pytest.raises(ValueError):
        resize_with_pad(image.astype(np.float32), 20)


def test_resize_stretches_to_the_requested_size():
    pytest.importorskip("PIL")
    from phyai_robot.images import resize, resize_with_pad

    image = np.random.default_rng(2).integers(0, 256, (744, 960, 3), dtype=np.uint8)
    stretched = resize(image, 540, 960)
    assert stretched.shape == (540, 960, 3) and stretched.flags.writeable
    padded = resize_with_pad(stretched, 224)
    content = np.where(padded.max(axis=(1, 2)) > 0)[0]
    assert padded.shape == (224, 224, 3)
    assert content.min() >= 49 and content.max() <= 174  # 126 rows, centered.
    assert resize(stretched, 540, 960) is not stretched
    with pytest.raises(ValueError):
        resize(image, 0, 10)
