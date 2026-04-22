from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch

from .fusion import FusionModule
from .gat_encoder import GATEncoder
from .spatial_encoder import SpatialEncoder


@dataclass
class NetworkOutput:
    logits: torch.Tensor
    value: torch.Tensor
    fused: torch.Tensor


class DualStreamActorCritic(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        in_channels: int,
        action_dim: int,
        gat_dim: int = 128,
        spatial_dim: int = 128,
        fused_dim: int = 256,
        gat_heads: int = 4,
    ):
        super().__init__()
        self.gat = GATEncoder(node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim, embed_dim=gat_dim, heads=gat_heads)
        self.spatial = SpatialEncoder(in_channels=in_channels, embed_dim=spatial_dim)
        self.fusion = FusionModule(gat_dim=gat_dim, spatial_dim=spatial_dim, fused_dim=fused_dim)
        self.policy_head = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.ReLU(), nn.Linear(fused_dim, action_dim))
        self.value_head = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.ReLU(), nn.Linear(fused_dim, 1))

    def forward(self, graph_batch: Batch, spatial_obs: torch.Tensor) -> NetworkOutput:
        g = self.gat(graph_batch)
        s = self.spatial(spatial_obs)
        fused = self.fusion(g, s)
        logits = self.policy_head(fused)
        value = self.value_head(fused).squeeze(-1)
        return NetworkOutput(logits=logits, value=value, fused=fused)

    @staticmethod
    def masked_dist(logits: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
        valid = action_mask > 0
        any_valid = valid.any(dim=-1, keepdim=True)
        safe_valid = torch.where(any_valid, valid, torch.ones_like(valid))
        masked_logits = logits.masked_fill(~safe_valid, -1e9)
        return Categorical(logits=masked_logits)

    def act(
        self,
        graph_batch: Batch,
        spatial_obs: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.forward(graph_batch, spatial_obs)
        dist = self.masked_dist(out.logits, action_mask)
        action = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, out.value
