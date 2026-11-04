"""Replicated GoalWAM text projections with reference rounding boundaries."""

import torch
from torch.nn import functional as F

from phyai.parallel import Mesh, RankLayout, default_mesh, register_mesh
from phyai.layers.linear import QKVParallelLinear
from phyai.layers.layer_norm import RMSNorm


class GoalWAMRMSNorm(RMSNorm):
    """Keep the FP32 mean and activation-dtype rounding before the affine."""

    def forward(
        self, value: torch.Tensor, *, preserve_3d_layout: bool = False
    ) -> torch.Tensor:
        del preserve_3d_layout
        floating = value.float()
        normalized = floating * torch.rsqrt(
            floating.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * normalized.to(value.dtype)


class GoalWAMTextProjection(QKVParallelLinear):
    """Share PhyAI's fused weight loader, preserving three separate GEMMs."""

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, None]:
        if self.tp_size != 1 or self.bias is not None:
            raise ValueError("GoalWAM text prefill uses replicated, bias-free Q/K/V")
        outputs = (
            F.linear(value, weight)
            for weight in self.weight.split(self.output_partition_sizes, dim=0)
        )
        return torch.cat(tuple(outputs), dim=-1), None


def register_goalwam_text_mesh() -> str:
    """Use singleton layer groups for prefill; the runner shards the result."""
    model = default_mesh()
    size = model.replica_world_size
    groups = {
        name: tuple((rank,) for rank in range(size)) for name in model.group_names
    }
    groups["world"] = (tuple(range(size)),)
    mesh = Mesh(RankLayout(size, groups), rank=model.rank, name="goalwam_text")
    register_mesh(mesh)
    return mesh.name
