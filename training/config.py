from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    board_width: int = 32
    board_height: int = 32
    resolution_mm: float = 0.5
    num_comp_classes: int = 4

    use_ratsnest: bool = True
    use_criticality: bool = True

    gat_embed_dim: int = 128
    spatial_embed_dim: int = 128
    fused_dim: int = 256
    gat_heads: int = 4

    n_envs: int = 8
    n_steps: int = 256
    n_epochs: int = 4
    batch_size: int = 64
    clip_range: float = 0.2
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    component_rotations: int = 4

    total_timesteps: int = 200_000
    eval_freq: int = 5_000
    checkpoint_freq: int = 10_000
    board_dir: str = "data/boards/rl_pcb/base"
    checkpoint_dir: str = "runs/checkpoints"
    seed: int = 42
    
    # Algorithm Selection
    algo: str = "ppo"  # Options: ppo, td3, sac
    tau: float = 0.005
    alpha: float = 0.2
    replay_buffer_size: int = 100_000
    
    # TD3/SAC specific
    expl_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    pi_hidden_sizes: list = None
    qf_hidden_sizes: list = None


    hpwl_dense_weight: float = 2.0
    hpwl_terminal_weight: float = 0.5
    drc_penalty: float = 10.0
    routability_weight: float = 1.0

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with Path(path).open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        # Filter out keys that aren't in the dataclass to allow for comments/extra yaml data
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
