import torch
from torch import nn
from phyai.models.lingbot_v2.modeling_lingbotv2 import (
    LingBotV2AdaRMSNorm,
)


class CountingLinear(nn.Module):
    """Count calls to the gamma/beta projections."""

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, width) * 0.01)
        self.bias = nn.Parameter(torch.randn(width) * 0.01)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.nn.functional.linear(x, self.weight, self.bias), None


def test_ada_norm_modulation_reuse():
    torch.manual_seed(42)

    # Construct a minimal real AdaRMSNorm without engine setup.
    norm = LingBotV2AdaRMSNorm.__new__(LingBotV2AdaRMSNorm)
    nn.Module.__init__(norm)

    norm.variance_epsilon = 1e-6
    norm.weight = nn.Parameter(torch.ones(8))
    norm.gamma = CountingLinear(8)
    norm.beta = CountingLinear(8)

    state = torch.randn(1, 8)
    action = torch.randn(1, 5, 8)
    cond = torch.randn(1, 8)

    with torch.no_grad():
        # Original API: projections are computed twice.
        old_state = norm(state, cond)
        old_action = norm(action, cond)

        assert norm.gamma.calls == 2
        assert norm.beta.calls == 2

        norm.gamma.calls = 0
        norm.beta.calls = 0

        # Optimized API: project once and reuse for both streams.
        modulation = norm.compute_modulation(cond)

        new_state = norm(state, cond, modulation=modulation)
        new_action = norm(action, cond, modulation=modulation)

        assert torch.equal(old_state, new_state)
        assert torch.equal(old_action, new_action)
        assert norm.gamma.calls == 1
        assert norm.beta.calls == 1
