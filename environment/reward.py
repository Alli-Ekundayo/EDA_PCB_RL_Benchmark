from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .board import Board, component_center, occupied_grid


@dataclass
class RewardWeights:
    hpwl_dense_weight: float = 2.0
    hpwl_terminal_weight: float = 0.5
    drc_penalty: float = 10.0
    routability_weight: float = 1.0


def _centers(board: Board) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for c in board.components:
        if c.placed and c.position is not None:
            out[c.ref] = component_center(c)
    return out


def hpwl(board: Board) -> float:
    centers = _centers(board)
    total = 0.0
    for refs in board.nets.values():
        pts = [centers[r] for r in refs if r in centers]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return float(total)


def normalized_hpwl(board: Board) -> float:
    denom = max(1.0, float(board.width + board.height) * max(len(board.components), 1))
    return hpwl(board) / denom


def compute_overlap_count(board: Board) -> float:
    # Approximate overlap burden as occupied density beyond 1.
    occ = occupied_grid(board).astype(np.int32)
    return float(np.clip(occ - 1, 0, None).sum())


def pattern_routability_proxy(board: Board) -> float:
    # Lightweight proxy: more fully-placed nets with lower span is better.
    centers = _centers(board)
    if not board.nets:
        return 0.0
    score = 0.0
    count = 0
    for refs in board.nets.values():
        pts = [centers[r] for r in refs if r in centers]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        span = (max(xs) - min(xs)) + (max(ys) - min(ys))
        score += 1.0 / (1.0 + span)
        count += 1
    if count == 0:
        return 0.0
    return float(score / count)


def reward_components(
    board_before: Board,
    board_after: Board,
    invalid_action: bool,
    weights: RewardWeights,
) -> Dict[str, float]:
    hpwl_term = -(normalized_hpwl(board_after) - normalized_hpwl(board_before))
    drc_term = -1.0 if invalid_action else 0.0
    overlap_term = -(compute_overlap_count(board_after) - compute_overlap_count(board_before))
    routability_term = pattern_routability_proxy(board_after) - pattern_routability_proxy(board_before)
    total = (
        weights.hpwl_weight * hpwl_term
        + weights.drc_penalty * drc_term
        + 0.25 * weights.drc_penalty * overlap_term
        + weights.routability_weight * routability_term
    )
    return {
        "reward_total": float(total),
        "reward_hpwl": float(hpwl_term),
        "reward_drc": float(drc_term),
        "reward_overlap": float(overlap_term),
        "reward_routability": float(routability_term),
    }
