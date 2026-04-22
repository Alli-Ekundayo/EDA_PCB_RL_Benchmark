from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from .board import Board, Component, occupied_grid


def _in_bounds(board: Board, x: int, y: int, fp: np.ndarray) -> bool:
    w, h = fp.shape
    return x >= 0 and y >= 0 and (x + w) <= board.width and (y + h) <= board.height


def _placement_violates_drc(board: Board, component: Component, x: int, y: int) -> bool:
    fp = component.footprint_for_rotation().astype(bool)
    if not _in_bounds(board, x, y, fp):
        return True

    occupied = occupied_grid(board, exclude_ref=component.ref)
    blocked = occupied | board.keepout

    if board.min_clearance > 0:
        kernel = np.ones((2 * board.min_clearance + 1, 2 * board.min_clearance + 1), dtype=bool)
        blocked = binary_dilation(blocked, structure=kernel)

    w, h = fp.shape
    region = blocked[x : x + w, y : y + h]
    return bool(np.any(region & fp))


def compute_action_mask(
    board: Board,
    component: Component,
    rotations: tuple[int, ...] = (0, 90, 180, 270),
) -> np.ndarray:
    # 1. Compute the base blocked grid (occupied + keepout) once
    # Note: occupied_grid now returns int, so we convert to bool
    occupied = occupied_grid(board, exclude_ref=component.ref).astype(bool)
    blocked = occupied | board.keepout

    # 2. Apply clearance dilation once if needed
    if board.min_clearance > 0:
        kernel = np.ones((2 * board.min_clearance + 1, 2 * board.min_clearance + 1), dtype=bool)
        blocked = binary_dilation(blocked, structure=kernel)

    masks = []
    for rot in rotations:
        fp = component.footprint.copy()
        k = (int(rot) // 90) % 4
        fp = np.rot90(fp, k=k).astype(bool)
        
        w, h = fp.shape
        # Use binary_dilation with the footprint as the structure to find all invalid positions
        # However, binary_dilation centers the structure. We want top-left placement.
        # A simpler way is to use a manual loop over the footprint bits IF it's sparse,
        # but for small boards, we can just use the fact that:
        # invalid[x, y] = any(blocked[x:x+w, y:y+h] & fp)
        
        mask = np.ones((board.width, board.height), dtype=bool)
        # Pre-filter out-of-bounds positions
        if w > board.width or h > board.height:
            mask[:] = False
        else:
            # Only iterate over possible top-left corners
            mask[board.width - w + 1 :, :] = False
            mask[:, board.height - h + 1 :] = False
            
            # For each active bit in the footprint, "propagate" the blocked cells
            # This is equivalent to: invalid = sum over (i, j) in fp: blocked shifted by (i, j)
            invalid = np.zeros((board.width, board.height), dtype=bool)
            fx, fy = np.where(fp)
            for dx, dy in zip(fx, fy):
                # Shift blocked grid by (dx, dy) and OR it into invalid
                # We only care about the region that can stay in bounds
                if w <= board.width and h <= board.height:
                    invalid[0 : board.width - w + 1, 0 : board.height - h + 1] |= \
                        blocked[dx : dx + board.width - w + 1, dy : dy + board.height - h + 1]
            
            mask &= ~invalid

        masks.append(mask)

    stacked = np.stack(masks, axis=0)
    return stacked.reshape(-1)
