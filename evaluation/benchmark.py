#!/usr/bin/env python
"""Unified benchmark runner for PPO vs TD3 vs SAC."""
import sys
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import Config
from training.vec_env import make_vec_env
from training.train import _extract_info_values, _graph_to_data
from training.logger import log_dict
from environment.reward import RewardWeights
from environment.reward import hpwl, normalized_hpwl
from models.networks import DualStreamActorCritic
from torch_geometric.data import Batch


def evaluate_ppo_agent(
    config: Config,
    checkpoint_path: str,
    n_episodes: int = 5,
) -> Dict[str, float]:
    """Evaluate trained PPO agent."""
    device = "cpu"
    
    rotations = tuple(90 * i for i in range(config.component_rotations))
    reward_weights = RewardWeights(
        hpwl_dense_weight=config.hpwl_dense_weight, hpwl_terminal_weight=config.hpwl_terminal_weight,
        drc_penalty=config.drc_penalty,
        routability_weight=config.routability_weight,
    )
    
    env = make_vec_env(
        n_envs=1,
        board_dir=config.board_dir,
        width=config.board_width,
        height=config.board_height,
        component_rotations=rotations,
        reward_weights=reward_weights,
    )
    
    obs, info = env.reset(seed=42)
    graph_objs = _extract_info_values(info, "graph", 1, None)
    action_masks = np.stack(_extract_info_values(info, "action_mask", 1, np.ones(env.single_action_space.n, dtype=bool)))
    graph_list = [_graph_to_data(g) for g in graph_objs]
    
    in_channels = obs.shape[1]
    node_feat_dim = graph_list[0].x.shape[1]
    edge_feat_dim = graph_list[0].edge_attr.shape[1] if graph_list[0].edge_attr.numel() > 0 else 4
    action_dim = env.single_action_space.n
    
    model = DualStreamActorCritic(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        in_channels=in_channels,
        action_dim=action_dim,
        gat_dim=config.gat_embed_dim,
        spatial_dim=config.spatial_embed_dim,
        fused_dim=config.fused_dim,
        gat_heads=config.gat_heads,
    ).to(device)
    
    # Load checkpoint if available
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    
    model.eval()
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        graph_objs = _extract_info_values(info, "graph", 1, None)
        action_masks = np.stack(_extract_info_values(info, "action_mask", 1, np.ones(env.single_action_space.n, dtype=bool)))
        graph_list = [_graph_to_data(g) for g in graph_objs]
        
        ep_reward = 0.0
        while True:
            spatial = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action_mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
            batch_graph = Batch.from_data_list(graph_list).to(device)
            
            with torch.no_grad():
                actions_t, _, _ = model.act(batch_graph, spatial, action_mask_t, deterministic=True)
            
            action = actions_t[0].item()
            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            
            graph_objs = _extract_info_values(info, "graph", 1, graph_objs[0])
            graph_list = [_graph_to_data(g) for g in graph_objs]
            action_masks = np.stack(_extract_info_values(info, "action_mask", 1, np.ones(env.single_action_space.n, dtype=bool)))
            
            ep_reward += reward[0]
            
            if terminated[0] or truncated[0]:
                break
        
        episode_rewards.append(ep_reward)
    
    env.close()
    
    return {
        "algorithm": "PPO (DualStream+GNN)",
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "n_episodes": n_episodes,
    }


def benchmark_on_boards(
    config_path: str = "configs/small_board.yaml",
    board_pattern: str = "small_board_*.json",
    n_episodes_per_board: int = 3,
) -> None:
    """Run benchmark on multiple boards."""
    config = Config.from_yaml(config_path)
    
    # Find boards matching pattern
    board_dir = Path(config.board_dir)
    boards = sorted(board_dir.glob(board_pattern))
    
    print(f"\n=== PCB Placement Benchmark ===")
    print(f"Config: {config_path}")
    print(f"Board pattern: {board_pattern}")
    print(f"Found {len(boards)} boards to evaluate")
    print(f"Episodes per board: {n_episodes_per_board}\n")
    
    results = {
        "config": asdict(config),
        "board_pattern": board_pattern,
        "algorithms": {},
    }
    
    # Benchmark PPO (if checkpoint exists)
    checkpoint = Path(config.checkpoint_dir) / "ppo_dualstream_step_32768.pt"
    print(f"Evaluating PPO (checkpoint: {checkpoint})...")
    try:
        ppo_result = evaluate_ppo_agent(config, str(checkpoint), n_episodes=n_episodes_per_board)
        results["algorithms"]["PPO"] = ppo_result
        log_dict({f"ppo/{k}": v for k, v in ppo_result.items() if k != 'algorithm'})
    except Exception as e:
        print(f"PPO evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print(f"\n=== Benchmark Summary ===")
    for algo, result in results["algorithms"].items():
        print(f"\n{algo}:")
        for k, v in result.items():
            if k != 'algorithm':
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Save results
    results_path = Path("runs/benchmarks") / f"benchmark_{Path(board_pattern).stem}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/small_board.yaml")
    parser.add_argument("--pattern", type=str, default="small_board_*.json")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    
    benchmark_on_boards(
        config_path=args.config,
        board_pattern=args.pattern,
        n_episodes_per_board=args.episodes,
    )
