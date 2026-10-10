"""Exact eager-operation parity and buffer contracts for affine modulation."""

from __future__ import annotations

import itertools

import pytest
import torch

import phyai_kernel


DTYPES = [torch.float16, torch.bfloat16, torch.float32]
DEVICES = ["cpu", "cuda"]


def _reference(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual, expected)


def _inputs(
    shape: tuple[int, int, int], device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20261010)
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    parameters = torch.randn(
        2, shape[0], shape[2], device=device, dtype=dtype, generator=generator
    )
    return x, parameters[0], parameters[1]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", [(1, 1, 1), (1, 30, 768), (2, 7, 513), (3, 5, 1025), (2, 13, 4097)]
)
def test_modulate_matches_eager(
    device: str, dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    x, shift, scale = _inputs(shape, device, dtype)
    _assert_exact(phyai_kernel.modulate(x, shift, scale), _reference(x, shift, scale))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["sliced", "permuted", "expanded"])
def test_modulate_noncontiguous_inputs_and_chunked_parameters(
    device: str, dtype: torch.dtype, layout: str
) -> None:
    batch, sequence, dim = 3, 7, 129
    generator = torch.Generator(device=device).manual_seed(517)
    if layout == "sliced":
        storage = torch.randn(
            2 * batch + 1,
            2 * sequence + 1,
            2 * dim + 1,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        x = storage[1::2, 1::2, 1::2]
        assert x.storage_offset() > 0
    elif layout == "permuted":
        x = torch.randn(
            dim, batch, sequence, device=device, dtype=dtype, generator=generator
        ).permute(1, 2, 0)
    else:
        x = torch.randn(
            1, sequence, dim, device=device, dtype=dtype, generator=generator
        ).expand(batch, sequence, dim)
        assert x.stride(0) == 0
    projection = torch.randn(
        batch, 9 * dim, device=device, dtype=dtype, generator=generator
    )
    shift, scale = projection.chunk(9, dim=-1)[1::7]
    assert not x.is_contiguous()
    assert not shift.is_contiguous() and not scale.is_contiguous()
    snapshots = tuple(value.clone() for value in (x, shift, scale))

    actual = phyai_kernel.modulate(x, shift, scale)

    _assert_exact(actual, _reference(x, shift, scale))
    for value, snapshot in zip((x, shift, scale), snapshots, strict=True):
        assert torch.equal(value, snapshot)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "shift_shape,scale_shape",
    list(itertools.product([(3, 17), (1, 17), (3, 1), (1, 1)], repeat=2)),
)
def test_modulate_independent_parameter_broadcasts(
    device: str, shift_shape: tuple[int, int], scale_shape: tuple[int, int]
) -> None:
    x, _, _ = _inputs((3, 5, 17), device, torch.bfloat16)
    generator = torch.Generator(device=device).manual_seed(173)
    parameters = []
    for rows, columns in (shift_shape, scale_shape):
        storage = torch.randn(
            2 * rows + 1,
            2 * columns + 1,
            device=device,
            dtype=x.dtype,
            generator=generator,
        )
        parameters.append(storage[1::2, 1::2])
    shift, scale = parameters
    _assert_exact(phyai_kernel.modulate(x, shift, scale), _reference(x, shift, scale))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_modulate_expanded_parameters(device: str, dtype: torch.dtype) -> None:
    x, shift, scale = _inputs((3, 7, 33), device, dtype)
    shift = shift[:1].expand(3, 33)
    scale = scale[:, :1].expand(3, 33)
    assert shift.stride(0) == scale.stride(1) == 0
    _assert_exact(phyai_kernel.modulate(x, shift, scale), _reference(x, shift, scale))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(0, 7, 13), (2, 0, 13), (2, 7, 0), (0, 0, 0)])
@pytest.mark.parametrize("use_out", [False, True])
def test_modulate_empty_dimensions(
    device: str, dtype: torch.dtype, shape: tuple[int, int, int], use_out: bool
) -> None:
    x, shift, _ = _inputs(shape, device, dtype)
    scale = torch.zeros((1, 1), device=device, dtype=dtype)
    out = torch.empty_like(x) if use_out else None
    actual = phyai_kernel.modulate(x, shift, scale, out=out)
    _assert_exact(actual, _reference(x, shift, scale))
    if use_out:
        assert actual is out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype,mantissa_bits", [(torch.bfloat16, 7), (torch.float16, 10)]
)
def test_modulate_rounds_each_low_precision_operation(
    device: str, dtype: torch.dtype, mantissa_bits: int
) -> None:
    unit = 2.0**-mantissa_bits
    # Column zero detects a missing cast after 1 + scale. Column one detects a
    # missing cast after multiplication, even if the preceding addition rounds.
    x = torch.tensor([[[2.0**mantissa_bits, 1 + unit]]], device=device, dtype=dtype)
    scale = torch.tensor([[unit / 2, 0.5]], device=device, dtype=dtype)
    shift = torch.tensor([[-(2.0**mantissa_bits), -1.5]], device=device, dtype=dtype)
    expected = _reference(x, shift, scale)
    promoted = _reference(x.float(), shift.float(), scale.float()).to(dtype)
    factor_rounded_only = (
        x.float() * (1 + scale).float().unsqueeze(1) + shift.float().unsqueeze(1)
    ).to(dtype)
    assert expected[0, 0, 0] != promoted[0, 0, 0]
    assert expected[0, 0, 1] != factor_rounded_only[0, 0, 1]
    _assert_exact(phyai_kernel.modulate(x, shift, scale), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_modulate_float32_does_not_fuse_multiply_add(device: str) -> None:
    unit = 2.0**-23
    x = torch.tensor([[[1 + unit]]], device=device, dtype=torch.float32)
    scale = torch.tensor([[-unit]], device=device, dtype=torch.float32)
    shift = torch.tensor([[-1.0]], device=device, dtype=torch.float32)
    expected = _reference(x, shift, scale)
    # These binary inputs have an exact float64 product. This computes the
    # correctly rounded single FMA result without relying on compiler fusion.
    fused = (x.double() * (1 + scale).double() + shift.double()).float()
    assert expected.item() == 0.0
    assert fused.item() == -(unit**2)
    _assert_exact(phyai_kernel.modulate(x, shift, scale), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_modulate_preserves_subnormals_and_overflow(
    device: str, dtype: torch.dtype
) -> None:
    info = torch.finfo(dtype)
    smallest = info.tiny * info.eps
    x = torch.tensor(
        [[[smallest, -smallest, info.tiny / 2, info.tiny, 0, info.max, -info.max]]],
        device=device,
        dtype=dtype,
    )
    scale = torch.tensor([[0, 0, 0, -0.5, 0, 1, 1]], device=device, dtype=dtype)
    shift = torch.tensor(
        [[0, 0, 0, 0, info.tiny / 2, 0, 0]], device=device, dtype=dtype
    )
    expected = _reference(x, shift, scale)
    assert torch.all(expected[..., :5] != 0)
    assert torch.all(torch.isinf(expected[..., 5:]))
    _assert_exact(phyai_kernel.modulate(x, shift, scale), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_modulate_reuses_out_without_mutating_inputs_or_previous_outputs(
    device: str, dtype: torch.dtype
) -> None:
    x, shift, scale = _inputs((2, 7, 33), device, dtype)
    snapshots = tuple(value.clone() for value in (x, shift, scale))
    previous = phyai_kernel.modulate(x, shift, scale)
    previous_snapshot = previous.clone()
    out = torch.full_like(x, float("nan"))
    pointer = out.data_ptr()
    assert phyai_kernel.modulate(x, shift, scale, out=out) is out
    _assert_exact(out, previous)
    for value, snapshot in zip((x, shift, scale), snapshots, strict=True):
        assert torch.equal(value, snapshot)

    x.fill_(2)
    shift.fill_(3)
    scale.fill_(-0.5)
    assert phyai_kernel.modulate(x, shift, scale, out=out) is out
    assert out.data_ptr() == pointer
    _assert_exact(out, _reference(x, shift, scale))
    assert torch.equal(previous, previous_snapshot)
    assert not torch.equal(out, previous)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "argument,shape",
    [
        ("x", ()),
        ("x", (2, 17)),
        ("x", (2, 1, 7, 17)),
        ("shift", (17,)),
        ("shift", (2, 1, 17)),
        ("scale", (17,)),
        ("scale", (2, 1, 17)),
        ("shift", (3, 17)),
        ("shift", (2, 16)),
        ("scale", (3, 17)),
        ("scale", (2, 16)),
        ("shift", (0, 17)),
        ("scale", (2, 0)),
    ],
)
def test_modulate_rejects_invalid_input_shapes(
    device: str, argument: str, shape: tuple[int, ...]
) -> None:
    values = dict(
        zip(("x", "shift", "scale"), _inputs((2, 7, 17), device, torch.float32))
    )
    values[argument] = torch.empty(shape, device=device, dtype=torch.float32)
    with pytest.raises(ValueError):
        phyai_kernel.modulate(**values)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("argument", ["shift", "scale"])
def test_modulate_rejects_mismatched_input_dtypes(device: str, argument: str) -> None:
    values = dict(
        zip(("x", "shift", "scale"), _inputs((2, 7, 17), device, torch.float32))
    )
    values[argument] = values[argument].to(torch.float16)
    with pytest.raises(ValueError, match="same dtype and device"):
        phyai_kernel.modulate(**values)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.bool])
def test_modulate_rejects_unsupported_dtypes(device: str, dtype: torch.dtype) -> None:
    x = torch.ones((2, 7, 17), device=device, dtype=dtype)
    parameters = torch.ones((2, 17), device=device, dtype=dtype)
    with pytest.raises(ValueError, match="supports float16, bfloat16, and float32"):
        phyai_kernel.modulate(x, parameters, parameters)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("argument", ["x", "shift", "scale"])
def test_modulate_rejects_sparse_inputs(device: str, argument: str) -> None:
    values = dict(
        zip(("x", "shift", "scale"), _inputs((2, 7, 17), device, torch.float32))
    )
    values[argument] = values[argument].to_sparse()
    with pytest.raises(ValueError, match="strided inputs"):
        phyai_kernel.modulate(**values)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("argument", ["shift", "scale", "out"])
def test_modulate_rejects_mismatched_devices(device: str, argument: str) -> None:
    values = dict(
        zip(("x", "shift", "scale"), _inputs((2, 7, 17), device, torch.float32))
    )
    other_device = "cpu" if device == "cuda" else "meta"
    shape = (2, 7, 17) if argument == "out" else (2, 17)
    values[argument] = torch.empty(shape, device=other_device, dtype=torch.float32)
    with pytest.raises(ValueError):
        phyai_kernel.modulate(**values)


def test_modulate_cpu_rejects_unsupported_device() -> None:
    x, shift, scale = _inputs((2, 7, 17), "cpu", torch.float32)
    with pytest.raises(ValueError, match="requires CPU or CUDA"):
        phyai_kernel.modulate(x.to("meta"), shift.to("meta"), scale.to("meta"))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["shape", "dtype", "noncontiguous", "sparse"])
def test_modulate_rejects_invalid_out(device: str, case: str) -> None:
    x, shift, scale = _inputs((2, 7, 17), device, torch.float32)
    if case == "shape":
        out = torch.empty((2, 7, 16), device=device, dtype=x.dtype)
    elif case == "dtype":
        out = torch.empty_like(x, dtype=torch.bfloat16)
    elif case == "noncontiguous":
        out = torch.empty((2, 17, 7), device=device, dtype=x.dtype).transpose(1, 2)
    else:
        out = torch.empty_like(x).to_sparse()
    with pytest.raises(ValueError):
        phyai_kernel.modulate(x, shift, scale, out=out)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("argument", ["x", "shift", "scale", "disjoint_x"])
def test_modulate_rejects_out_sharing_input_storage(device: str, argument: str) -> None:
    x, shift, scale = _inputs((2, 1, 17), device, torch.float32)
    if argument == "x":
        out = x
    elif argument == "shift":
        out = shift.unsqueeze(1)
    elif argument == "scale":
        out = scale.unsqueeze(1)
    else:
        storage = torch.empty((2 * x.numel(),), device=device, dtype=x.dtype)
        out = storage[x.numel() :].view_as(x)
        x = storage[: x.numel()].view_as(x)
        assert x.data_ptr() != out.data_ptr()
    assert out.is_contiguous()
    with pytest.raises(ValueError, match="share storage"):
        phyai_kernel.modulate(x, shift, scale, out=out)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_out", [False, True])
def test_modulate_cuda_graph_replay_uses_updated_inputs(
    dtype: torch.dtype, use_out: bool
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(371)
    batch, sequence, dim = 2, 7, 129
    x = torch.randn(
        batch, sequence, 2 * dim, device="cuda", dtype=dtype, generator=generator
    )[..., ::2]
    parameters = torch.randn(
        batch, 9 * dim, device="cuda", dtype=dtype, generator=generator
    )
    shift, scale = parameters.chunk(9, dim=-1)[1::7]
    out = torch.empty(x.shape, device="cuda", dtype=dtype) if use_out else None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            phyai_kernel.modulate(x, shift, scale, out=out)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = phyai_kernel.modulate(x, shift, scale, out=out)
    pointer = actual.data_ptr()
    graph.replay()
    _assert_exact(actual, _reference(x, shift, scale))
    original = actual.clone()
    if use_out:
        assert actual is out

    # All three captured pointers remain fixed while their values change.
    x.fill_(2)
    shift.fill_(3)
    scale.fill_(-0.5)
    graph.replay()
    _assert_exact(actual, _reference(x, shift, scale))
    assert actual.data_ptr() == pointer
    assert not torch.equal(actual, original)

    x.fill_(-4)
    shift.fill_(-2)
    scale.fill_(0.25)
    graph.replay()
    _assert_exact(actual, _reference(x, shift, scale))
    graph.reset()
