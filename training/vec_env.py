from __future__ import annotations

from pathlib import Path
from typing import Optional

from gymnasium.vector import AsyncVectorEnv

from environment.pcb_env import PCBEnv
from environment.reward import RewardWeights


def make_vec_env(
    n_envs: int = 16,
    board_dir: str = "data/boards",
    width: int = 32,
    height: int = 32,
    component_rotations: tuple[int, ...] = (0, 90, 180, 270),
    reward_weights: Optional[RewardWeights] = None,
    use_ratsnest: bool = True,
    use_criticality: bool = True,
):
    path = Path(board_dir)
    if path.is_file():
        board_files = [path]
    else:
        board_files = []
        for ext in ("*.pcb", "*.json", "*.net"):
            board_files.extend(path.glob(ext))
    
    if not board_files:
        raise FileNotFoundError(f"No board files found at {board_dir}")

    reward_weights = reward_weights or RewardWeights()

    def make_env(board_path: str, pcb_idx: int):
        def _init():
            return PCBEnv(
                board_path=board_path,
                width=width,
                height=height,
                pcb_idx=pcb_idx,
                component_rotations=component_rotations,
                reward_weights=reward_weights,
                use_ratsnest=use_ratsnest,
                use_criticality=use_criticality,
            )

        return _init

    env_fns = []
    for i in range(n_envs):
        p = str(board_files[i % len(board_files)])
        env_fns.append(make_env(board_path=p, pcb_idx=i % 6))
    return AsyncVectorEnv(env_fns)
