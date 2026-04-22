#!/usr/bin/env python
"""Baseline comparison: TD3 and SAC from stable-baselines3."""
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import Config
from training.vec_env import make_vec_env
from environment.reward import RewardWeights
from training.logger import log_dict


def make_continuous_vec_env(
    n_envs: int = 16,
    board_dir: str = "data/boards",
    width: int = 32,
    height: int = 32,
    component_rotations: tuple[int, ...] = (0, 90, 180, 270),
    reward_weights: Optional[RewardWeights] = None,
):
    from gymnasium import Wrapper, spaces
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from environment.pcb_env import PCBEnv

    class ActionWrapperContinuous(Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.original_action_space = env.action_space
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        def step(self, action: np.ndarray):
            disc_action = int(np.clip(action[0] * 0.5 + 0.5, 0, 1) * (self.original_action_space.n - 1))
            return self.env.step(disc_action)

    board_files = []
    for ext in ("*.pcb", "*.json", "*.net"):
        board_files.extend(Path(board_dir).glob(ext))
    if not board_files:
        raise FileNotFoundError(f"No board files in {board_dir}")

    reward_weights = reward_weights or RewardWeights()

    def make_env(board_path: str, pcb_idx: int):
        def _init():
            env = PCBEnv(
                board_path=board_path,
                width=width,
                height=height,
                pcb_idx=pcb_idx,
                component_rotations=component_rotations,
                reward_weights=reward_weights,
            )
            return ActionWrapperContinuous(env)
        return _init

    env_fns = []
    for i in range(n_envs):
        p = str(board_files[i % len(board_files)])
        env_fns.append(make_env(board_path=p, pcb_idx=i % 6))
    return SubprocVecEnv(env_fns)


def train_td3_baseline(config: Config, n_timesteps: int = 50000) -> Dict[str, float]:
    """Train TD3 baseline."""
    print("\n=== Training TD3 Baseline ===")
    
    rotations = tuple(90 * i for i in range(config.component_rotations))
    reward_weights = RewardWeights(
        hpwl_weight=config.hpwl_weight,
        drc_penalty=config.drc_penalty,
        routability_weight=config.routability_weight,
    )
    
    # TD3 works with continuous action spaces
    train_env = make_continuous_vec_env(
        n_envs=config.n_envs,
        board_dir=config.board_dir,
        width=config.board_width,
        height=config.board_height,
        component_rotations=rotations,
        reward_weights=reward_weights,
    )
    
    # TD3 hyperparameters
    policy_kwargs = dict(net_arch=[256, 256])
    
    model = TD3(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    
    print(f"Training TD3 for {n_timesteps} timesteps...")
    model.learn(total_timesteps=n_timesteps)
    
    print("TD3 training complete!")
    env.close()
    
    return {
        "algorithm": "TD3",
        "total_timesteps": n_timesteps,
        "final_reward": 0.0,  # Placeholder
    }


def train_sac_baseline(config: Config, n_timesteps: int = 50000) -> Dict[str, float]:
    """Train SAC baseline."""
    print("\n=== Training SAC Baseline ===")
    
    rotations = tuple(90 * i for i in range(config.component_rotations))
    reward_weights = RewardWeights(
        hpwl_weight=config.hpwl_weight,
        drc_penalty=config.drc_penalty,
        routability_weight=config.routability_weight,
    )
    
    # SAC also works with continuous action spaces
    train_env = make_continuous_vec_env(
        n_envs=config.n_envs,
        board_dir=config.board_dir,
        width=config.board_width,
        height=config.board_height,
        component_rotations=rotations,
        reward_weights=reward_weights,
    )
    
    # SAC hyperparameters
    policy_kwargs = dict(net_arch=[256, 256])
    
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=1000,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )
    
    print(f"Training SAC for {n_timesteps} timesteps...")
    model.learn(total_timesteps=n_timesteps)
    
    print("SAC training complete!")
    env.close()
    
    return {
        "algorithm": "SAC",
        "total_timesteps": n_timesteps,
        "final_reward": 0.0,  # Placeholder
    }


def compare_baselines(config_path: str = "configs/small_board.yaml", n_timesteps: int = 10000) -> None:
    """Run TD3 and SAC baseline comparisons."""
    config = Config.from_yaml(config_path)
    
    print(f"Baseline Comparison")
    print(f"  Config: {config_path}")
    print(f"  Board size: {config.board_width}x{config.board_height}")
    print(f"  N envs: {config.n_envs}")
    print(f"  Timesteps: {n_timesteps}")
    
    results = []
    
    try:
        td3_result = train_td3_baseline(config, n_timesteps=n_timesteps)
        results.append(td3_result)
    except Exception as e:
        print(f"TD3 training failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        sac_result = train_sac_baseline(config, n_timesteps=n_timesteps)
        results.append(sac_result)
    except Exception as e:
        print(f"SAC training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Baseline Comparison Results ===")
    for result in results:
        print(f"{result['algorithm']}: {n_timesteps} timesteps")
        log_dict({k: v for k, v in result.items() if k != 'algorithm'})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/small_board.yaml")
    parser.add_argument("--timesteps", type=int, default=10000)
    args = parser.parse_args()
    
    compare_baselines(config_path=args.config, n_timesteps=args.timesteps)
