from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, sdf_tensor: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(sdf_tensor))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return self.norm(F.relu(self.fc(x)))
