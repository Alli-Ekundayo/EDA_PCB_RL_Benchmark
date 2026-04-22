from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    def __init__(self, gat_dim: int = 128, spatial_dim: int = 128, fused_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(gat_dim + spatial_dim, fused_dim)
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, gat_embed: torch.Tensor, spatial_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([gat_embed, spatial_embed], dim=-1)
        return self.norm(F.relu(self.proj(combined)))
