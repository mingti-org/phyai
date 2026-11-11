"""Robot-side resizing must equal the model server's resize, pixel for pixel."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from phyai_robot.images import resize_with_pad
from phyai_utils_tools.processing.ops.image_ops import resize_with_pad as server_resize


@pytest.mark.parametrize("shape", [(480, 640, 3), (1984, 2560, 3), (300, 200, 3)])
def test_client_resize_matches_server_pil_backend(shape):
    image = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    client = resize_with_pad(image, 224)
    server = server_resize(
        torch.from_numpy(image).permute(2, 0, 1)[None], 224, 224, backend="pil"
    )[0].permute(1, 2, 0)
    np.testing.assert_array_equal(client, server.numpy())
    # The server leaves an already-sized image unchanged.
    tensor = torch.from_numpy(client).permute(2, 0, 1)[None]
    assert server_resize(tensor, 224, 224, backend="pil") is tensor
