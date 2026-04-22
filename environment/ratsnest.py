from __future__ import annotations

from typing import Tuple

import numpy as np

from .board import Board, component_center


from scipy.ndimage import gaussian_filter


def compute_ratsnest_maps(board: Board) -> Tuple[np.ndarray, np.ndarray]:
    density = np.zeros((board.width, board.height), dtype=np.float32)
    criticality = np.zeros((board.width, board.height), dtype=np.float32)
    refs = {c.ref: c for c in board.components}

    # 1. Accumulate points at component centers
    for net_id, members in board.nets.items():
        resolved = [refs[r] for r in members if r in refs and refs[r].placed]
        if len(resolved) < 1:
            continue

        for c in resolved:
            cx, cy = component_center(c)
            # Round to nearest pixel for point density
            ix, iy = int(np.clip(cx, 0, board.width - 1)), int(np.clip(cy, 0, board.height - 1))
            density[ix, iy] += 1.0
            criticality[ix, iy] += float(board.net_criticality.get(net_id, 0.3))

    # 2. Apply Gaussian smoothing once
    sigma = 2.0
    density = gaussian_filter(density, sigma=sigma)
    criticality = gaussian_filter(criticality, sigma=sigma)

    max_d = density.max()
    max_c = criticality.max()
    if max_d > 0:
        density /= max_d
    if max_c > 0:
        criticality /= max_c
    return density, criticality
