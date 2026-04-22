from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from .board import Board, occupied_grid
from .ratsnest import compute_ratsnest_maps

NUM_COMPONENT_CLASSES = 4


def get_occupied_cells(board: Board, class_id: int) -> np.ndarray:
    grid = np.zeros((board.width, board.height), dtype=bool)
    for c in board.components:
        if c.class_id != class_id or not c.placed or c.position is None:
            continue
        fp = c.footprint_for_rotation().astype(bool)
        x0, y0 = c.position
        w, h = fp.shape
        grid[x0 : x0 + w, y0 : y0 + h] |= fp
    return grid


def compute_sdf(board: Board, num_classes: int = NUM_COMPONENT_CLASSES) -> np.ndarray:
    tensor = np.zeros((num_classes + 3, board.width, board.height), dtype=np.float32)

    for cls in range(num_classes):
        occupied = get_occupied_cells(board, class_id=cls)
        if occupied.any():
            tensor[cls] = distance_transform_edt(~occupied).astype(np.float32)
        else:
            tensor[cls] = np.full((board.width, board.height), 999.0, dtype=np.float32)

    tensor[num_classes] = distance_transform_edt(~board.keepout).astype(np.float32)
    density, criticality = compute_ratsnest_maps(board)
    tensor[num_classes + 1] = density
    tensor[num_classes + 2] = criticality

    return tensor


def occupied_with_keepout(board: Board) -> np.ndarray:
    return occupied_grid(board) | board.keepout
