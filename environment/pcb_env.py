from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .board import Board
from .drc_mask import compute_action_mask
from .netlist_parser import ParsedGraph, parse_board_file
from .reward import RewardWeights, reward_components
from .sdf_generator import NUM_COMPONENT_CLASSES, compute_sdf
from .tracker import PlacementTracker


class PCBEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        board_path: Optional[str] = None,
        board_dir: Optional[str] = None,
        width: int = 32,
        height: int = 32,
        component_rotations: tuple[int, ...] = (0, 90, 180, 270),
        max_steps: Optional[int] = None,
        pcb_idx: Optional[int] = None,
        reward_weights: Optional[RewardWeights] = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.board_path = board_path
        self.board_dir = board_dir
        self.board: Optional[Board] = None
        self.graph: Optional[ParsedGraph] = None
        self.current_idx = 0
        self.step_count = 0
        self.max_steps = max_steps
        self.pcb_idx = pcb_idx
        self.rotations = tuple(component_rotations)
        self.reward_weights = reward_weights or RewardWeights()
        self.tracker = PlacementTracker()

        self.observation_space = spaces.Box(
            low=0.0,
            high=1e4,
            shape=(NUM_COMPONENT_CLASSES + 3, width, height),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(width * height * len(self.rotations))

    def _pick_board_file(self) -> str:
        if self.board_path is not None:
            return self.board_path
        if self.board_dir is None:
            raise ValueError("Either board_path or board_dir must be provided")
        
        path = Path(self.board_dir)
        if path.is_file():
            return str(path)
            
        files = []
        for ext in ("*.pcb", "*.json", "*.net"):
            files.extend(path.glob(ext))
        if not files:
            raise FileNotFoundError(f"No board files in {self.board_dir}")
        idx = int(np.random.randint(0, len(files)))
        return str(files[idx])

    def _current_component(self):
        assert self.board is not None
        while self.current_idx < len(self.board.components) and self.board.components[self.current_idx].placed:
            self.current_idx += 1
        if self.current_idx >= len(self.board.components):
            return None
        return self.board.components[self.current_idx]

    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        board_cells = self.width * self.height
        rot_idx = int(action // board_cells)
        cell_idx = int(action % board_cells)
        x = int(cell_idx // self.height)
        y = int(cell_idx % self.height)
        rotation = self.rotations[int(np.clip(rot_idx, 0, len(self.rotations) - 1))]
        return x, y, rotation

    def _obs_info(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        assert self.board is not None and self.graph is not None
        obs = compute_sdf(self.board)
        comp = self._current_component()
        if comp is None:
            action_mask = np.ones(self.action_space.n, dtype=bool)
            current_ref = None
            current_component_idx = -1
        else:
            action_mask = compute_action_mask(self.board, comp, rotations=self.rotations)
            current_ref = comp.ref
            current_component_idx = self.current_idx

        info: Dict[str, Any] = {
            "graph": self.graph,
            "action_mask": action_mask,
            "current_component_ref": current_ref,
            "current_component_idx": current_component_idx,
        }
        return obs.astype(np.float32), info


    def _get_reward_metrics(self) -> Dict[str, float]:
        from .reward import normalized_hpwl, compute_overlap_count, pattern_routability_proxy
        return {
            "hpwl": normalized_hpwl(self.board),
            "overlap": compute_overlap_count(self.board),
            "routability": pattern_routability_proxy(self.board),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        board_file = self._pick_board_file()
        board, _, graph = parse_board_file(board_file, width=self.width, height=self.height, pcb_idx=self.pcb_idx)
        self.board = board
        self.graph = graph
        self.current_idx = 0
        self.step_count = 0
        if self.max_steps is None:
            self.max_steps = max(1, len(self.board.components) * 3)
        
        obs, info = self._obs_info()
        self._last_metrics = self._get_reward_metrics()
        self._last_action_mask = info["action_mask"]
        return obs, info

    def step(self, action: int):
        assert self.board is not None
        comp = self._current_component()
        if comp is None:
            obs, info = self._obs_info()
            return obs, 0.0, True, False, info

        x, y, rotation = self._decode_action(action)
        action_mask = self._last_action_mask
        
        if not action_mask.any():
            obs, info = self._obs_info()
            info["valid_action"] = False
            return obs, -float(self.reward_weights.drc_penalty), True, False, info

        valid = bool(action_mask[int(action)])
        if valid:
            comp.rotation = rotation
            comp.placed = True
            comp.position = (x, y)
            self.current_idx += 1

        # Calculate reward using tracked metrics instead of cloning
        new_metrics = self._get_reward_metrics()
        
        terminated = all(c.placed for c in self.board.components)
        
        # Apply sparse HPWL reward only at the end of the episode to prevent
        # punishing the agent for initial placements that establish the net bounding box.
        if terminated:
            hpwl_term = -new_metrics["hpwl"]
        else:
            hpwl_term = 0.0
            
        drc_term = -1.0 if not valid else 0.0
        overlap_term = -(new_metrics["overlap"] - self._last_metrics["overlap"])
        routability_term = new_metrics["routability"] - self._last_metrics["routability"]
        
        reward = (
            self.reward_weights.hpwl_weight * hpwl_term
            + self.reward_weights.drc_penalty * drc_term
            + 0.25 * self.reward_weights.drc_penalty * overlap_term
            + self.reward_weights.routability_weight * routability_term
        )
        self._last_metrics = new_metrics

        self.step_count += 1
        terminated = all(c.placed for c in self.board.components)
        truncated = self.step_count >= int(self.max_steps)
        obs, info = self._obs_info()
        self._last_action_mask = info["action_mask"]
        
        info["valid_action"] = valid
        info.update(new_metrics)
        from .reward import hpwl
        info["hpwl"] = hpwl(self.board)
        
        from .ratsnest import compute_ratsnest_maps
        from .board import occupied_grid
        density_map, _ = compute_ratsnest_maps(self.board)
        self.tracker.record_step(occupied_grid(self.board), density_map, reward, info["hpwl"])
        
        return obs, float(reward), terminated, truncated, info
