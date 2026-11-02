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


def make_selector(mode, device):
    catalog = build_catalog()
    config = {"schema": "phyai.kernel/v1", "defaults": {"fallback": "reference"}}
    if mode != "default":
        rule = {"id": "gemm", "match": {"op": "gemm"}}
        if mode == "torch":
            rule["restrict_to"] = "torch.gemm.*"
        elif mode == "strict":
            rule["use"] = "flashinfer.gemm.bf16"
        else:
            rule["prefer"] = ["flashinfer.gemm.bf16"]
        config["overrides" if mode == "strict" else "rules"] = [rule]
        if mode == "error":
            config["defaults"]["fallback"] = "error"
    return Selector(catalog, policy_from_mapping(config, catalog), device=device)


@pytest.mark.parametrize("mode", ["default", "torch", "flashinfer"])
@pytest.mark.parametrize("shape", [(0, 63), (1, 63), (3, 63), (2, 3, 63)])
def test_router_fp32_output_and_strides(mode, shape, fake_mesh):
    fake_mesh()
    device = os.environ.get("AUDIT_DEVICE", "cuda")
    with kernel_selector_scope(make_selector(mode, device)):
        layer = ReplicatedLinear(
            63,
            37,
            bias=True,
            params_dtype=torch.bfloat16,
            out_dtype=torch.float32,
            device=device,
            kernel_role="router",
        )
        layer.weight.data.normal_()
        layer.bias.data.normal_()
        x = torch.randn(*shape[:-1], 126, dtype=torch.bfloat16, device=device)[..., ::2]
        output, bias = layer(x)
        # BF16 products are accumulated in FP32; reduction order may differ.
        torch.testing.assert_close(
            output,
            F.linear(x.float(), layer.weight.float(), layer.bias.float()),
            atol=2e-5,
            rtol=2e-6,
        )
        assert bias is None and output.dtype == torch.float32
        assert output.shape == (*shape[:-1], 37)
        expected_kernel = (
            "flashinfer.gemm.bf16"
            if mode != "torch" and torch.device(device).type == "cuda"
            else "torch.gemm.bf16"
        )
        assert {h.kernel_id for h in layer.gemm_call._memo.values()} == {
            expected_kernel
        }


@pytest.mark.parametrize("mode", ["strict", "error"])
def test_router_policy_respects_failure_mode(mode):
    from phyai.kernel.types import KernelQuery

    selector = make_selector(mode, "cpu")
    query = KernelQuery.build(
        "gemm",
        device="cpu",
        dtype={"input": "bf16", "weight": "bf16", "output": "fp32"},
        shape={"M": 3, "N": 37, "K": 63},
        quant={"format": "bf16"},
    )
    assert selector.explain(query).selected is None


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("backend", ["torch", "flashinfer"])
def test_dense_policy_load_amp_and_weight_reload(
    tmp_path, fake_mesh, amp, bias, backend
):
    from phyai.kernel.policy import load_policy

    fake_mesh()
    policy_path = tmp_path / "kernel.yaml"
    policy_path.write_text(
        "schema: phyai.kernel/v1\n"
        "overrides:\n"
        "- id: router\n"
        "  match: {op: gemm, role: router}\n"
        f"  use: {backend}.gemm.bf16\n"
        + ("  params: {backend: cublaslt}\n" if backend == "flashinfer" else "")
    )
    catalog = build_catalog()
    selector = Selector(catalog, load_policy(policy_path, catalog), device="cuda")
    with kernel_selector_scope(selector):
        layers = [
            ReplicatedLinear(
                63,
                37,
                bias=bias,
                device="cuda",
                params_dtype=torch.bfloat16,
                out_dtype=torch.float32,
                prefix=f"gate{i}",
                kernel_role="router",
            )
            for i in range(2)
        ]
        torch.manual_seed(72)
        x = torch.randn(2, 3, 126, dtype=torch.bfloat16, device="cuda")[:, :, ::2]
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
                        x.float(),
                        layer.weight.float(),
                        layer.bias.float() if bias else None,
                    )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    actual, _ = layer(x)
                assert parameter is layer.weight and parameter.dtype == torch.bfloat16
                assert parameter.hf_keys == [(f"{layer.prefix}.weight", None)]
                assert actual.dtype == torch.float32
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)
                assert {h.kernel_id for h in layer.gemm_call._memo.values()} == {
                    f"{backend}.gemm.bf16"
                }
