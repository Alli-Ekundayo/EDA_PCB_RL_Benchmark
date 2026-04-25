#!/usr/bin/env python
"""Quick training script for testing."""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from dataclasses import asdict

from training.config import Config
from training.vec_env import make_vec_env
from training.train import _extract_info_values, _graph_to_data, _compute_gae
from environment.reward import RewardWeights
from models.networks import DualStreamActorCritic
from models.ppo_agent import PPOAgent, RolloutBatch
from training.logger import log_dict
from torch_geometric.data import Batch

def quick_train():
    cfg = Config.from_yaml('configs/small_board.yaml')
    # Override for quick test
    cfg.n_steps = 4
    cfg.n_epochs = 1
    cfg.total_timesteps = 32  # Just 1 update
    
    print(f"Starting quick training: {cfg.n_envs} envs, {cfg.n_steps} steps, 1 update")
    sys.stdout.flush()
    
    device = "cpu"
    rotations = tuple(90 * i for i in range(cfg.component_rotations))
    reward_weights = RewardWeights(
        hpwl_dense_weight=cfg.hpwl_dense_weight,
        hpwl_terminal_weight=cfg.hpwl_terminal_weight,
        drc_penalty=cfg.drc_penalty,
        routability_weight=cfg.routability_weight,
    )
    
    print("Creating env...")
    sys.stdout.flush()
    env = make_vec_env(
        n_envs=cfg.n_envs,
        board_dir=cfg.board_dir,
        width=cfg.board_width,
        height=cfg.board_height,
        component_rotations=rotations,
        reward_weights=reward_weights,
    )
    
    print("Resetting env...")
    sys.stdout.flush()
    obs, info = env.reset(seed=42)
    graph_objs = _extract_info_values(info, "graph", cfg.n_envs, None)
    action_masks = np.stack(_extract_info_values(info, "action_mask", cfg.n_envs, np.ones(env.single_action_space.n, dtype=bool)))
    graph_list = [_graph_to_data(g) for g in graph_objs]
    
    print("Creating model...")
    sys.stdout.flush()
    in_channels = obs.shape[1]
    node_feat_dim = graph_list[0].x.shape[1]
    edge_feat_dim = graph_list[0].edge_attr.shape[1] if graph_list[0].edge_attr.numel() > 0 else 4
    action_dim = env.single_action_space.n
    
    model = DualStreamActorCritic(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        in_channels=in_channels,
        action_dim=action_dim,
        gat_dim=cfg.gat_embed_dim,
        spatial_dim=cfg.spatial_embed_dim,
        fused_dim=cfg.fused_dim,
        gat_heads=cfg.gat_heads,
    ).to(device)
    
    agent = PPOAgent(
        model=model,
        lr=cfg.lr,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
    )
    
    n_updates = 1
    print(f"Starting training loop ({n_updates} updates)...")
    sys.stdout.flush()
    
    for update in range(1, n_updates + 1):
        print(f"  Update {update}/{n_updates}")
        sys.stdout.flush()
        
        roll_obs = []
        roll_masks = []
        roll_actions = []
        roll_logp = []
        roll_rewards = []
        roll_dones = []
        roll_values = []
        roll_graphs = []
        
        for step in range(cfg.n_steps):
            print(f"    Step {step+1}/{cfg.n_steps}", end='\r')
            sys.stdout.flush()
            
            spatial = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action_mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
            batch_graph = Batch.from_data_list(graph_list).to(device)
            
            with torch.no_grad():
                actions_t, logp_t, values_t = model.act(batch_graph, spatial, action_mask_t, deterministic=False)
            
            actions = actions_t.detach().cpu().numpy()
            logps = logp_t.detach().cpu().numpy()
            values = values_t.detach().cpu().numpy()
            
            next_obs, rewards, terminated, truncated, next_info = env.step(actions)
            done = np.logical_or(terminated, truncated)
            
            roll_obs.append(obs.copy())
            roll_masks.append(action_masks.copy())
            roll_actions.append(actions.copy())
            roll_logp.append(logps.copy())
            roll_rewards.append(rewards.astype(np.float32))
            roll_dones.append(done.astype(bool))
            roll_values.append(values.astype(np.float32))
            roll_graphs.extend(graph_list)
            
            obs = next_obs
            graph_objs = _extract_info_values(next_info, "graph", cfg.n_envs, graph_objs[0])
            graph_list = [_graph_to_data(g) for g in graph_objs]
            action_masks = np.stack(
                _extract_info_values(next_info, "action_mask", cfg.n_envs, np.ones(action_dim, dtype=bool))
            )
        
        print(f"\n    Computing advantages...")
        sys.stdout.flush()
        
        with torch.no_grad():
            spatial = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action_mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
            batch_graph = Batch.from_data_list(graph_list).to(device)
            next_values = model.forward(batch_graph, spatial).value.detach().cpu().numpy().astype(np.float32)
        
        values_arr = np.vstack([np.stack(roll_values), next_values[None, :]])
        rewards_arr = np.stack(roll_rewards)
        dones_arr = np.stack(roll_dones)
        advantages, returns = _compute_gae(rewards_arr, values_arr, dones_arr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
        
        print(f"    Updating agent...")
        sys.stdout.flush()
        
        flat_obs = torch.as_tensor(np.concatenate(roll_obs, axis=0), dtype=torch.float32, device=device)
        flat_masks = torch.as_tensor(np.concatenate(roll_masks, axis=0), dtype=torch.bool, device=device)
        flat_actions = torch.as_tensor(np.concatenate(roll_actions, axis=0), dtype=torch.long, device=device)
        flat_logp = torch.as_tensor(np.concatenate(roll_logp, axis=0), dtype=torch.float32, device=device)
        flat_adv = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        flat_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        
        rollout = RolloutBatch(
            spatial_obs=flat_obs,
            action_masks=flat_masks,
            actions=flat_actions,
            old_log_probs=flat_logp,
            returns=flat_returns,
            advantages=flat_adv,
        )
        
        metrics = agent.update(
            rollout=rollout,
            graph_list=roll_graphs,
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
        )
        metrics["train/update"] = float(update)
        metrics["train/mean_reward"] = float(rewards_arr.mean())
        
        print(f"\n  Update {update} complete:")
        log_dict(metrics)
    
    env.close()
    print("\nTraining complete!")

if __name__ == "__main__":
    quick_train()
