from __future__ import annotations
import argparse
import sys
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import os
import torch
from torch_geometric.data import Data, Batch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.reward import RewardWeights
from environment.wrappers import ContinuousToDiscrete
from models.networks import DualStreamActorCritic, SpatialEncoder, GATEncoder
from models.ppo_agent import PPOAgent, RolloutBatch
from models.td3_agent import TD3Agent, TD3Actor, TD3Critic
from models.sac_agent import SACAgent, SACActor, SACCritic
from training.config import Config
from training.logger import log_dict
from training.vec_env import make_vec_env
from training.replay_buffer import GraphReplayBuffer

def _graph_to_data(g) -> Data:
    return Data(
        x=torch.as_tensor(g.node_features, dtype=torch.float32),
        edge_index=torch.as_tensor(g.edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(g.edge_attr, dtype=torch.float32),
    )

def _extract_info_values(info: Dict[str, Any], key: str, n_envs: int, default):
    if key not in info: return [default for _ in range(n_envs)]
    value = info[key]
    if isinstance(value, np.ndarray) and len(value) == n_envs: return [value[i] for i in range(n_envs)]
    if isinstance(value, list) and len(value) == n_envs: return value
    return [value for _ in range(n_envs)]

def _compute_gae(rewards, values, dones, gamma: float, gae_lambda: float):
    n_steps, n_envs = rewards.shape
    advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
    gae = np.zeros((n_envs,), dtype=np.float32)
    for t in reversed(range(n_steps)):
        non_terminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns

def train_ppo(config: Config, device):
    rotations = tuple(90 * i for i in range(config.component_rotations))
    reward_weights = RewardWeights(
        hpwl_weight=config.hpwl_weight,
        drc_penalty=config.drc_penalty,
        routability_weight=config.routability_weight,
    )
    env = make_vec_env(
        n_envs=config.n_envs, board_dir=config.board_dir, width=config.board_width,
        height=config.board_height, component_rotations=rotations, reward_weights=reward_weights,
    )
    obs, info = env.reset(seed=config.seed)
    graph_objs = _extract_info_values(info, "graph", config.n_envs, None)
    graph_list = [_graph_to_data(g) for g in graph_objs]
    
    model = DualStreamActorCritic(
        node_feat_dim=graph_list[0].x.shape[1],
        edge_feat_dim=graph_list[0].edge_attr.shape[1] if graph_list[0].edge_attr.numel() > 0 else 4,
        in_channels=obs.shape[1], action_dim=env.single_action_space.n,
        gat_dim=config.gat_embed_dim, spatial_dim=config.spatial_embed_dim,
        fused_dim=config.fused_dim, gat_heads=config.gat_heads,
    ).to(device)
    agent = PPOAgent(model=model, lr=config.lr)
    
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    n_updates = max(1, config.total_timesteps // (config.n_steps * config.n_envs))
    global_step = 0

    for update in range(1, n_updates + 1):
        roll_obs, roll_masks, roll_actions, roll_logp, roll_rewards, roll_dones, roll_values, roll_graphs = [], [], [], [], [], [], [], []
        for _ in range(config.n_steps):
            spatial = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action_mask_t = torch.as_tensor(np.stack(_extract_info_values(info, "action_mask", config.n_envs, np.ones(env.single_action_space.n, dtype=bool))), dtype=torch.bool, device=device)
            batch_graph = Batch.from_data_list(graph_list).to(device)
            with torch.no_grad():
                actions_t, logp_t, values_t = model.act(batch_graph, spatial, action_mask_t, deterministic=False)
            actions, logps, values = actions_t.cpu().numpy(), logp_t.cpu().numpy(), values_t.cpu().numpy()
            next_obs, rewards, terminated, truncated, next_info = env.step(actions)
            done = np.logical_or(terminated, truncated)
            
            roll_obs.append(obs.copy()); roll_masks.append(action_mask_t.cpu().numpy()); roll_actions.append(actions.copy())
            roll_logp.append(logps.copy()); roll_rewards.append(rewards.astype(np.float32)); roll_dones.append(done.astype(bool))
            roll_values.append(values.astype(np.float32)); roll_graphs.extend(graph_list)
            
            obs = next_obs
            graph_objs = _extract_info_values(next_info, "graph", config.n_envs, graph_objs[0])
            graph_list = [_graph_to_data(g) for g in graph_objs]
            info = next_info
            global_step += config.n_envs

        with torch.no_grad():
            next_values = model.forward(Batch.from_data_list(graph_list).to(device), torch.as_tensor(obs, dtype=torch.float32, device=device)).value.cpu().numpy().astype(np.float32)
        
        adv, ret = _compute_gae(np.stack(roll_rewards), np.vstack([np.stack(roll_values), next_values[None, :]]), np.stack(roll_dones), config.gamma, config.gae_lambda)
        metrics = agent.update(RolloutBatch(
            spatial_obs=torch.as_tensor(np.concatenate(roll_obs), device=device),
            action_masks=torch.as_tensor(np.concatenate(roll_masks), device=device),
            actions=torch.as_tensor(np.concatenate(roll_actions), device=device),
            old_log_probs=torch.as_tensor(np.concatenate(roll_logp), device=device),
            returns=torch.as_tensor(ret.reshape(-1), device=device),
            advantages=torch.as_tensor(((adv - adv.mean()) / (adv.std() + 1e-8)).reshape(-1), device=device),
        ), roll_graphs, config.n_epochs, config.batch_size)
        
        metrics.update({"train/global_step": float(global_step), "train/mean_reward": float(np.mean(roll_rewards))})
        log_dict(metrics)
        if update % 2 == 0:
            torch.save({"model": model.state_dict(), "config": asdict(config)}, checkpoint_dir / f"ppo_step_{global_step}.pt")
    
    # Final Save
    torch.save({"model": model.state_dict(), "config": asdict(config)}, checkpoint_dir / f"ppo_final_step_{global_step}.pt")
    env.close()

def train_off_policy(config: Config, device):
    from environment.pcb_env import PCBEnv
    base_env = PCBEnv(board_dir=config.board_dir, width=config.board_width, height=config.board_height, 
                      component_rotations=tuple(90*i for i in range(config.component_rotations)))
    env = ContinuousToDiscrete(base_env)
    obs, info = env.reset(seed=config.seed)
    graph_data = _graph_to_data(info['graph'])
    
    spatial_enc = SpatialEncoder(in_channels=obs.shape[0], embed_dim=config.spatial_embed_dim)
    gat_enc = GATEncoder(graph_data.x.shape[1], graph_data.edge_attr.shape[1] if graph_data.edge_attr.numel() > 0 else 4, embed_dim=config.gat_embed_dim)
    
    # Shared DualStream logic
    class SharedEncoder(torch.nn.Module):
        def __init__(self, s_enc, g_enc, spatial_dim, gat_dim, fused_dim):
            super().__init__()
            self.spatial_enc, self.gat_enc, self.fused_dim = s_enc, g_enc, fused_dim
            self.fusion = torch.nn.Linear(spatial_dim + gat_dim, fused_dim)
        def forward(self, s, g):
            return torch.relu(self.fusion(torch.cat([self.spatial_enc(s), self.gat_enc(g)], dim=-1)))

    pi_hidden = config.pi_hidden_sizes if config.pi_hidden_sizes else [256]
    qf_hidden = config.qf_hidden_sizes if config.qf_hidden_sizes else [256]

    encoder = SharedEncoder(spatial_enc, gat_enc, config.spatial_embed_dim, config.gat_embed_dim, config.fused_dim).to(device)
    if config.algo == "td3":
        agent = TD3Agent(
            TD3Actor(encoder, pi_hidden).to(device), 
            TD3Critic(copy.deepcopy(encoder), qf_hidden).to(device), 
            lr=config.lr, tau=config.tau, gamma=config.gamma,
            policy_noise=config.policy_noise, noise_clip=config.noise_clip, policy_freq=config.policy_freq
        )
    else:
        agent = SACAgent(
            SACActor(encoder, pi_hidden).to(device), 
            SACCritic(copy.deepcopy(encoder), qf_hidden).to(device), 
            lr=config.lr, tau=config.tau, gamma=config.gamma, alpha=config.alpha
        )

    replay_buffer = GraphReplayBuffer(config.replay_buffer_size, device=device)
    global_step = 0
    while global_step < config.total_timesteps:
        # 1. Step
        spatial_t = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
        if config.algo == "td3":
            action = agent.select_action(spatial_t, graph_data, expl_noise=config.expl_noise)
        else:
            action = agent.select_action(spatial_t, graph_data)
        
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        
        # 2. Store
        replay_buffer.push(obs, graph_data, action, next_obs, reward, done)
        
        # 3. Update (Every 10 steps for CPU efficiency)
        if len(replay_buffer) > config.batch_size and global_step % 10 == 0:
            # Perform 5 updates to maintain sample efficiency
            for _ in range(5):
                metrics = agent.update(replay_buffer, config.batch_size)
            metrics.update({"train/global_step": float(global_step), "train/mean_reward": float(reward)})
            log_dict(metrics)
        elif global_step % 100 == 0:
            log_dict({"train/global_step": float(global_step), "train/status": 0.0})
            
        obs = next_obs
        # IMPORTANT: Update graph data every step
        graph_data = _graph_to_data(next_info['graph'])
        if done: 
            obs, info = env.reset()
            graph_data = _graph_to_data(info['graph'])
        global_step += 1
        
    # Final Save
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save({"model": agent.actor.state_dict(), "config": asdict(config)}, Path(config.checkpoint_dir) / f"{config.algo}_final.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--algo", type=str, choices=["ppo", "td3", "sac"])
    parser.add_argument("--total_timesteps", type=int)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.algo: config.algo = args.algo
    if args.total_timesteps: config.total_timesteps = args.total_timesteps
    if args.checkpoint_dir: config.checkpoint_dir = args.checkpoint_dir
    if args.seed is not None: config.seed = args.seed
    if args.log_file:
        import training.logger
        training.logger.LOG_FILE = args.log_file
    print(f"Starting experiment with Algorithm: {config.algo.upper()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.algo == "ppo": train_ppo(config, device)
    else: train_off_policy(config, device)

if __name__ == "__main__":
    main()
