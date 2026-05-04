from __future__ import annotations

import numpy as np


def continuous_action_to_discrete(
    action: np.ndarray,
    action_mask: np.ndarray,
    width: int,
    height: int,
    n_rotations: int,
) -> int:
    x_norm, y_norm, rot_norm = (action + 1.0) / 2.0
    px = int(np.clip(x_norm * width, 0, width - 1))
    py = int(np.clip(y_norm * height, 0, height - 1))
    prot = int(np.clip(rot_norm * n_rotations, 0, n_rotations - 1))
    flat_idx = prot * (width * height) + px * height + py

    if not action_mask.any() or action_mask[int(flat_idx)]:
        return int(flat_idx)

    valid_indices = np.where(action_mask)[0]
    valid_rot = valid_indices // (width * height)
    rem = valid_indices % (width * height)
    valid_x = rem // height
    valid_y = rem % height
    dist = (valid_x - px) ** 2 + (valid_y - py) ** 2 + 0.5 * (valid_rot - prot) ** 2
    return int(valid_indices[int(np.argmin(dist))])
