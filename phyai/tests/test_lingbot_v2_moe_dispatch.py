"""LingBot V2's MoE call reaches FlashInfer through the kernel dispatcher.

Assembling the real ``LingBotV2TokenMoE`` needs an engine mesh and the
released checkpoint, so the module is built the way the AdaRMSNorm test
builds its norm: ``__new__`` plus the attributes ``forward`` actually
reads. Only the leaf CUTLASS call is replaced -- the catalog, selector,
call site and adapter under test are the production ones, so a bypassed
dispatcher or a wrongly chosen row stays observable.
"""

from __future__ import annotations

import pathlib

import torch
import pytest
from phyai.kernel.call import CallSite
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector, NoKernelError
from phyai.kernel.bootstrap import kernel_selector_scope
from phyai.models.lingbot_v2.modeling_lingbotv2 import LingBotV2TokenMoE

pytestmark = pytest.mark.gpu

HIDDEN = 32
INTERMEDIATE = 16
NUM_EXPERTS = 4
TOP_K = 2
TOKENS = 6
KERNEL_ID = "flashinfer.moe.cutlass_fused_moe"

#: First output of the stubbed expert kernel, its decoy second output, and
#: the shared expert's constant. All exact in bfloat16, so the sums compare
#: bitwise.
ROUTED_RESULT = 1.0
DECOY_OUTPUT = -1.0
SHARED_RESULT = 0.5


class _Gate:
    """Stand-in for the replicated router linear."""

    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias


class _SharedExpert:
    """Constant contributor, so the shared path stays separable."""

    def __init__(self, fill):
        self.fill = fill

    def __call__(self, value):
        return torch.full_like(value, self.fill)


@pytest.fixture
def spy(monkeypatch):
    """Replace only the leaf kernel; keep the whole dispatch stack real."""
    from flashinfer import fused_moe

    calls = []

    def fake_cutlass_fused_moe(*args, **kwargs):
        calls.append((args, kwargs))
        hidden = args[0]
        # Two outputs, so ``return outputs[0]`` is observable.
        return (
            torch.full_like(hidden, ROUTED_RESULT),
            torch.full_like(hidden, DECOY_OUTPUT),
        )

    monkeypatch.setattr(fused_moe, "cutlass_fused_moe", fake_cutlass_fused_moe)
    return calls


def build_module(*, tp_size: int = 1, mesh=None):
    """Assemble the module the way its own ``__init__`` would.

    ``tp_size`` and ``mesh`` are parameters because ``forward`` reads both:
    the tensor-parallel collective needs the mesh stored at construction.
    """
    module = LingBotV2TokenMoE.__new__(LingBotV2TokenMoE)
    torch.nn.Module.__init__(module)

    torch.manual_seed(0)
    module.num_experts = NUM_EXPERTS
    module.top_k = TOP_K
    module.routed_scaling_factor = 1.0
    module.tp_size = tp_size
    module.mesh = mesh
    module.tp_rank = 0
    module.intermediate_size_per_partition = INTERMEDIATE
    module.gate = _Gate(
        torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    module.e_score_correction_bias = torch.zeros(
        NUM_EXPERTS, dtype=torch.bfloat16, device="cuda"
    )
    module.expert_gate_up_proj = (
        torch.randn(
            NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda"
        )
        * 0.1
    )
    module.expert_down_proj = (
        torch.randn(
            NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device="cuda"
        )
        * 0.1
    )
    module.shared_expert = _SharedExpert(SHARED_RESULT)
    module._moe_call = CallSite(
        "moe",
        role="lingbot_v2.token_moe",
        dims={"hidden": HIDDEN},
        attrs={"activation": "swiglu"},
    )
    return module


def run_forward(module, hidden_states=None):
    """Drive ``forward`` under a selector profiled as the Thor target."""
    if hidden_states is None:
        hidden_states = torch.randn(2, 3, HIDDEN, dtype=torch.bfloat16, device="cuda")
    selector = Selector(build_catalog(), device="nvidia:sm110")
    with kernel_selector_scope(selector):
        return module(hidden_states), hidden_states


# --------------------------------------------------------------------------- #
# The dispatch is on the path
# --------------------------------------------------------------------------- #


def test_forward_reaches_the_kernel_through_the_dispatcher(spy):
    module = build_module()
    run_forward(module)
    assert len(spy) == 1


def test_call_site_resolves_to_the_registered_flashinfer_kernel(spy):
    module = build_module()
    run_forward(module)
    # Same facts as the forward call, so this is the memoized production
    # selection rather than a fresh one.
    selection = module._moe_call.select(
        device=torch.device("cuda", torch.cuda.current_device()),
        dtype={"input": torch.bfloat16, "weight": torch.bfloat16},
        dims={"tokens": TOKENS},
    )
    assert selection.kernel_id == KERNEL_ID


def test_unsupported_dtype_is_refused_by_selection_not_by_the_kernel(spy):
    """fp16 must raise before the kernel is entered.

    A model that called FlashInfer directly would have reached the spy
    whatever the dtype, so raising here -- with the spy untouched -- is what
    proves the dispatcher sits in the path.
    """
    module = build_module()
    fp16 = torch.randn(2, 3, HIDDEN, dtype=torch.float16, device="cuda")
    with pytest.raises(NoKernelError):
        run_forward(module, hidden_states=fp16)
    assert spy == []


def test_model_no_longer_binds_flashinfer_directly():
    """Guard against a production call path re-entering the model file."""
    import phyai.models.lingbot_v2.modeling_lingbotv2 as modeling

    assert not hasattr(modeling, "_flashinfer_fused_moe")
    source = pathlib.Path(modeling.__file__).read_text()
    assert "from flashinfer" not in source


# --------------------------------------------------------------------------- #
# Argument contract at the kernel boundary
# --------------------------------------------------------------------------- #


def test_adapter_receives_the_documented_argument_contract(spy):
    module = build_module()
    _, hidden_states = run_forward(module)
    args, kwargs = spy[0]

    tokens = hidden_states.numel() // HIDDEN
    assert args[0].shape == (tokens, HIDDEN)
    assert args[0].dtype is torch.bfloat16
    assert args[1].shape == (tokens, TOP_K)
    assert args[1].dtype is torch.int32
    assert args[2].shape == (tokens, TOP_K)
    assert args[2].dtype is torch.float32
    assert args[3].shape == (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    assert args[4].shape == (NUM_EXPERTS, HIDDEN, INTERMEDIATE)
    assert args[5] is torch.bfloat16
    assert kwargs["quant_scales"] is None
    assert kwargs["tp_size"] == 1
    assert kwargs["tp_rank"] == 0


def test_expert_indices_are_int32_and_reference_real_experts(spy):
    module = build_module()
    run_forward(module)
    indices = spy[0][0][1]
    assert indices.dtype is torch.int32
    assert int(indices.min()) >= 0
    assert int(indices.max()) < NUM_EXPERTS


def test_routing_scales_are_fp32_over_bf16_rounded_values(spy):
    """The model rounds to bf16 first; the adapter only widens the dtype."""
    module = build_module()
    run_forward(module)
    scales = spy[0][0][2]
    assert scales.dtype is torch.float32
    assert torch.equal(scales, scales.to(torch.bfloat16).to(torch.float32))


# --------------------------------------------------------------------------- #
# Output semantics
# --------------------------------------------------------------------------- #


def test_only_the_first_output_is_used_and_the_shared_expert_is_added(spy):
    module = build_module()
    out, hidden_states = run_forward(module)
    assert out.shape == hidden_states.shape
    assert out.dtype is torch.bfloat16
    # The decoy second output would give a different constant.
    assert torch.equal(out, torch.full_like(out, ROUTED_RESULT + SHARED_RESULT))


# --------------------------------------------------------------------------- #
# Tensor parallelism
# --------------------------------------------------------------------------- #


def test_tp_collective_covers_the_routed_experts_only(spy, monkeypatch) -> None:
    """``tp_size > 1`` must reduce the routed output and nothing else.

    Two things are pinned here.  The shared expert is replicated, so folding
    it into the reduction would scale its contribution by ``tp_size``; and the
    mesh must survive construction, without which the branch raised
    ``NameError`` before it could reduce anything.

    The collective is intercepted, so this verifies the control flow and the
    reduced operand -- not a real two-rank reduction, which needs two GPUs and
    is NOT VERIFIED here.  ``spy`` still replaces only the leaf kernel, so the
    module's own forward runs.
    """

    import phyai.parallel as P

    recorded: list = []

    def _recording_all_reduce(tensor, **kwargs):
        recorded.append((tensor, kwargs))
        return tensor

    monkeypatch.setattr(P, "all_reduce", _recording_all_reduce)

    # tp_size == 1: the branch is skipped entirely.
    run_forward(build_module(tp_size=1, mesh=None))
    assert recorded == []

    # tp_size == 2: exactly one collective, over the routed output alone.
    sentinel = object()
    run_forward(build_module(tp_size=2, mesh=sentinel))

    assert len(recorded) == 1
    reduced, kwargs = recorded[0]
    assert kwargs["group"] == "moe_tp"
    assert kwargs["mesh"] is sentinel
    # The stub kernel returns ROUTED_RESULT and the shared expert adds
    # SHARED_RESULT, so a reduced value of ROUTED_RESULT alone proves the
    # shared expert was not folded into the reduction.
    assert torch.equal(reduced, torch.full_like(reduced, ROUTED_RESULT))


def test_real_construction_stores_the_mesh_on_the_module() -> None:
    """``__init__`` itself must keep the mesh that ``forward`` reduces with.

    Every other test in this file assembles the module through ``__new__`` and
    assigns ``mesh`` by hand, so this is the only one that exercises the
    assignment added in ``__init__``.  Without it, deleting that line would
    leave the whole suite green while reintroducing the NameError.
    """

    from phyai.engine_config import ParallelConfig
    from phyai.parallel.mesh import Mesh
    from phyai.parallel.state import _meshes, register_mesh
    from phyai.parallel.layout import build_rank_layout
    from phyai.models.lingbot_v2.configuration_lingbotv2 import LingBotV2MoEConfig

    saved = dict(_meshes)
    try:
        register_mesh(Mesh(build_rank_layout(ParallelConfig()), name="model"))
        module = LingBotV2TokenMoE(
            HIDDEN,
            LingBotV2MoEConfig(
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                moe_intermediate_size=INTERMEDIATE,
                shared_expert_intermediate_size=INTERMEDIATE,
                router_activation="sigmoid",
                normalize_topk_prob=True,
                routed_scaling_factor=1.0,
                router_bias=False,
                use_router_correction_bias=True,
                shared_expert_gate=False,
            ),
            params_dtype=torch.bfloat16,
            device="cuda",
            prefix="",
        )

        # An unset attribute would raise AttributeError here, not read as None.
        assert module.mesh is not None
        assert module.tp_size == 1
    finally:
        _meshes.clear()
        _meshes.update(saved)
