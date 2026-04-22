from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MaskedActor(nn.Module):
    def __init__(self, fused_dim: int = 256, board_w: int = 50, board_h: int = 50, rotations: int = 1):
        super().__init__()
        self.board_cells = board_w * board_h * max(rotations, 1)
        self.fc1 = nn.Linear(fused_dim, 512)
        self.fc2 = nn.Linear(512, self.board_cells)

    def forward(self, fused_embed: torch.Tensor, drc_mask: torch.Tensor) -> Categorical:
        logits = self.fc2(F.relu(self.fc1(fused_embed)))
        # Guard against all-invalid rows; fall back to unmasked logits for those rows.
        valid = drc_mask > 0.0
        any_valid = valid.any(dim=-1, keepdim=True)
        safe_valid = torch.where(any_valid, valid, torch.ones_like(valid))
        masked_logits = logits.masked_fill(~safe_valid, -1e9)
        return Categorical(logits=masked_logits)
