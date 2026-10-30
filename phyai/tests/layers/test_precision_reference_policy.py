"""Precision contracts must survive real selector policies and input layouts."""

import os

import torch
import pytest
import torch.nn.functional as F
from phyai.kernel.policy import policy_from_mapping
from phyai.layers.linear import ReplicatedLinear
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.bootstrap import kernel_selector_scope
from phyai.layers.layer_norm import RMSNorm, LayerNorm, GemmaRMSNorm


def make_selector(mode, device):
    catalog = build_catalog()
    config = {"schema": "phyai.kernel/v1", "defaults": {"fallback": "reference"}}
    if mode != "default":
        key = "overrides" if mode == "strict" else "rules"
        config[key] = []
        for op, accelerated in (
            ("gemm", "flashinfer.gemm.bf16"),
            ("layernorm", "flashinfer.layernorm"),
            ("rmsnorm", "flashinfer.rmsnorm"),
        ):
            rule = {"id": op, "match": {"op": op}}
            if mode == "torch":
                rule["restrict_to"] = f"torch.{op}*"
            elif mode == "reference":
                rule["restrict_to"] = (
                    "torch.gemm.bf16" if op == "gemm" else f"torch.{op}"
                )
            else:
                rule["use" if mode == "strict" else "prefer"] = (
                    [accelerated] if mode != "strict" else accelerated
                )
            config[key].append(rule)
        if mode == "error":
            config["defaults"]["fallback"] = "error"
    return Selector(catalog, policy_from_mapping(config, catalog), device=device)


@pytest.mark.parametrize("mode", ["default", "torch", "reference", "flashinfer"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [7, 63, 129])
def test_norm_precision_policy_and_strides(mode, dtype, width):
    device = os.environ.get("AUDIT_DEVICE", "cuda")
    torch.manual_seed(width)
    x = (torch.randn(2, 3, 2 * width, dtype=dtype, device=device) * 0.02 + 4)[..., ::2]
    assert not x.is_contiguous()
    with kernel_selector_scope(make_selector(mode, device)):
        for eps in (1e-6, 1e-3):
            ln = LayerNorm(
                width, dtype=dtype, device=device, eps=eps, explicit_fp32_math=True
            )
            rms = RMSNorm(
                width,
                dtype=dtype,
                device=device,
                eps=eps,
                explicit_fp32_math=True,
                cast_before_affine=True,
            )
            ln.weight.data.normal_()
            ln.bias.data.normal_()
            rms.weight.data.normal_()
            centered = x.float() - x.float().mean(-1, keepdim=True)
            expected_ln = (
                centered
                * torch.rsqrt(centered.square().mean(-1, keepdim=True) + eps)
                * ln.weight.float()
                + ln.bias.float()
            ).to(dtype)
            expected_rms = rms.weight * (
                x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            ).to(dtype)
            torch.testing.assert_close(ln(x), expected_ln, atol=0, rtol=0)
            torch.testing.assert_close(rms(x), expected_rms, atol=0, rtol=0)
            for layer, op in ((ln, "layernorm"), (rms, "rmsnorm")):
                assert {
                    h.kernel_id
                    for h in (
                        layer._call if op == "layernorm" else layer._plain_call
                    )._memo.values()
                } == {f"torch.{op}"}


@pytest.mark.parametrize("mode", ["default", "torch", "reference", "flashinfer"])
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((0, 63), torch.bfloat16),
        ((1, 63), torch.float32),
        ((3, 63), torch.float16),
        ((2, 3, 63), torch.bfloat16),
    ],
)
def test_router_reference_real_layer(mode, shape, dtype, fake_mesh):
    fake_mesh()
    device = os.environ.get("AUDIT_DEVICE", "cuda")
    with kernel_selector_scope(make_selector(mode, device)):
        layer = ReplicatedLinear(
            63,
            37,
            bias=True,
            params_dtype=torch.bfloat16,
            device=device,
        )
        layer.weight.data.normal_()
        layer.bias.data.normal_()
        x = torch.randn(*shape[:-1], 126, dtype=dtype, device=device)[..., ::2]
        output, bias = layer(x.float())
        torch.testing.assert_close(
            output,
            F.linear(x.float(), layer.weight.float(), layer.bias.float()),
            atol=0,
            rtol=0,
        )
        assert bias is None and output.dtype == torch.float32
        assert {h.kernel_id for h in layer.gemm_call._memo.values()} == {
            "torch.gemm.bf16"
        }


@pytest.mark.parametrize("mode", ["strict", "error"])
def test_reference_does_not_override_explicit_failure_policy(mode):
    from phyai.kernel.types import KernelQuery

    selector = make_selector(mode, "cpu")
    for op, dtype, attrs, shape, quant in [
        (
            "gemm",
            {"input": "fp32", "weight": "bf16", "output": "fp32"},
            {},
            {"M": 3, "N": 37, "K": 63},
            {"format": "bf16"},
        ),
        (
            "layernorm",
            {"input": "bf16", "weight": "bf16", "bias": "bf16"},
            {"math": "explicit_fp32", "bias": True},
            {"tokens": 3, "hidden": 63},
            None,
        ),
        (
            "rmsnorm",
            {"input": "bf16", "weight": "bf16"},
            {
                "math": "explicit_fp32",
                "cast_before_affine": True,
                "variant": "rms",
                "rank": 2,
            },
            {"tokens": 3, "hidden": 63},
            None,
        ),
    ]:
        query = KernelQuery.build(
            op, device="cpu", dtype=dtype, attrs=attrs, shape=shape, quant=quant
        )
        assert selector.explain(query).selected is None


@pytest.mark.parametrize("mode", ["default", "torch", "reference", "flashinfer"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gemma_explicit_math_uses_existing_reference(mode, dtype):
    device = os.environ.get("AUDIT_DEVICE", "cuda")
    torch.manual_seed(72)
    with kernel_selector_scope(make_selector(mode, device)):
        norm = GemmaRMSNorm(63, dtype=dtype, device=device, explicit_fp32_math=True)
        x = torch.randn(3, 126, dtype=dtype, device=device)[:, ::2]
        for random_affine in (False, True):
            if random_affine:
                norm.weight.data.normal_()
            promoted = x.float()
            expected = (
                promoted
                * torch.rsqrt(
                    promoted.square().mean(-1, keepdim=True) + norm.variance_epsilon
                )
                * (1 + norm.weight.float())
            ).to(dtype)
            torch.testing.assert_close(norm(x), expected, atol=0, rtol=0)
        assert {h.kernel_id for h in norm._plain_call._memo.values()} == {
            "torch.rmsnorm"
        }


@pytest.mark.parametrize("variant", ["rms", "gemma"])
@pytest.mark.parametrize("rounding", ["omitted", None, False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_rmsnorm_optional_rounding_fact(variant, rounding, explicit):
    from phyai.kernel.types import KernelQuery

    selector = make_selector("default", "cpu")
    attrs = {"variant": variant}
    if explicit:
        attrs["math"] = "explicit_fp32"
    if rounding != "omitted":
        attrs["cast_before_affine"] = rounding
    query = KernelQuery.build(
        "rmsnorm",
        device="cpu",
        dtype={"input": "fp32", "weight": "fp32"},
        attrs=attrs,
        shape={"tokens": 3, "hidden": 63},
    )
    expected = None if variant == "gemma" and rounding is True else "torch.rmsnorm"
    assert selector.explain(query).selected == expected


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("tf32", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_dense_policy_load_amp_and_weight_reload(tmp_path, fake_mesh, amp, tf32, bias):
    from phyai.kernel.policy import load_policy

    fake_mesh()
    policy_path = tmp_path / "kernel.yaml"
    policy_path.write_text(
        "schema: phyai.kernel/v1\n"
        "overrides:\n"
        "- id: router-reference\n"
        "  match: {op: gemm, role: router}\n"
        "  use: torch.gemm.bf16\n"
    )
    catalog = build_catalog()
    selector = Selector(catalog, load_policy(policy_path, catalog), device="cuda")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        with kernel_selector_scope(selector):
            layers = [
                ReplicatedLinear(
                    63,
                    37,
                    bias=bias,
                    device="cuda",
                    params_dtype=torch.bfloat16,
                    prefix=f"gate{i}",
                    kernel_role="router",
                )
                for i in range(2)
            ]
            torch.manual_seed(72)
            x = torch.randn(2, 3, 126, device="cuda")[:, :, ::2]
            for _ in range(2):
                for layer in layers:
                    parameter = layer.weight
                    loaded = torch.randn_like(parameter)
                    parameter.weight_loader(parameter, loaded, None)
                    if bias:
                        layer.bias.weight_loader(
                            layer.bias, torch.randn_like(layer.bias), None
                        )
                    layer.post_load()
                    with torch.amp.autocast("cuda", enabled=False):
                        expected = F.linear(
                            x,
                            layer.weight.float(),
                            layer.bias.float() if bias else None,
                        )
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        actual, _ = layer(x)
                    assert (
                        parameter is layer.weight and parameter.dtype == torch.bfloat16
                    )
                    assert actual.dtype == torch.float32
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    assert {h.kernel_id for h in layer.gemm_call._memo.values()} == {
                        "torch.gemm.bf16"
                    }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
