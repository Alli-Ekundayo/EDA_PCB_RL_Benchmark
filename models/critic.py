from __future__ import annotations

import torch
import torch.nn as nn


class DualHeadCritic(nn.Module):
    def __init__(self, fused_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head_hpwl = nn.Linear(128, 1)
        self.head_diffp = nn.Linear(128, 1)

    def forward(self, fused_embed: torch.Tensor):
        shared = self.trunk(fused_embed)
        return self.head_hpwl(shared), self.head_diffp(shared)
