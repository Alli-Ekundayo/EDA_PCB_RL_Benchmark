from __future__ import annotations

from typing import Dict

import numpy as np

from environment.board import Board
from environment.board import occupied_grid
from environment.reward import hpwl, normalized_hpwl


def overlap_rate(board: Board) -> float:
    if not board.components:
        return 0.0
    occ = occupied_grid(board).astype(np.int32)
    overlap = np.clip(occ - 1, 0, None).sum()
    denom = max(1.0, float(board.width * board.height))
    return float(overlap / denom)


def drc_pass_rate(total_actions: int, invalid_actions: int) -> float:
    if total_actions <= 0:
        return 1.0
    return max(0.0, 1.0 - (invalid_actions / total_actions))


def summarize_metrics(board: Board, total_actions: int = 0, invalid_actions: int = 0) -> Dict[str, float]:
    return {
        "hpwl": hpwl(board),
        "normalized_hpwl": normalized_hpwl(board),
        "drc_pass_rate": drc_pass_rate(total_actions, invalid_actions),
        "overlap_rate": overlap_rate(board),
    }
