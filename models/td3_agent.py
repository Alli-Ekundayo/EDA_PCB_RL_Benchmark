import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.networks import DualStreamActorCritic # We reuse the encoder logic

class TD3Actor(nn.Module):
    def __init__(self, encoder, hidden_dims=[400, 300], action_dim=3):
        super().__init__()
        self.encoder = encoder
        
        layers = []
        prev_dim = encoder.fused_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        self.net = nn.Sequential(*layers)
        self.action_out = nn.Linear(prev_dim, action_dim)
        
    def forward(self, spatial_obs, graph_data):
        features = self.encoder(spatial_obs, graph_data)
        x = self.net(features)
        return torch.tanh(self.action_out(x))

class TD3Critic(nn.Module):
    def __init__(self, encoder, hidden_dims=[400, 300], action_dim=3):
        super().__init__()
        self.encoder = encoder
        
        layers = []
        prev_dim = encoder.fused_dim + action_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        self.net = nn.Sequential(*layers)
        self.q_out = nn.Linear(prev_dim, 1)
        
    def forward(self, spatial_obs, graph_data, action):
        features = self.encoder(spatial_obs, graph_data)
        x = torch.cat([features, action], dim=-1)
        x = self.net(x)
        return self.q_out(x)

class TD3Agent:
    def __init__(self, actor, critic, lr=3e-4, tau=0.005, gamma=0.99, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic1 = critic
        self.critic1_target = copy.deepcopy(critic)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        
        self.critic2 = copy.deepcopy(critic)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.tau = tau
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.it = 0
        
    def select_action(self, spatial_obs, graph_data, expl_noise=0.0):
        with torch.no_grad():
            action = self.actor(spatial_obs, graph_data).cpu().numpy().flatten()
            if expl_noise > 0:
                noise = np.random.normal(0, expl_noise, size=action.shape)
                action = np.clip(action + noise, -1, 1)
            return action
            
    def update(self, replay_buffer, batch_size=64):
        self.it += 1
        
        # 1. Sample batch
        transitions = replay_buffer.sample(batch_size)
        
        # Helper to batch graph data
        from torch_geometric.data import Batch
        
        # Unpack and prepare tensors
        device = next(self.actor.parameters()).device
        
        spatial_obs = torch.as_tensor(np.stack([t.spatial_obs for t in transitions]), dtype=torch.float32, device=device)
        next_spatial_obs = torch.as_tensor(np.stack([t.next_spatial_obs for t in transitions]), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.stack([t.action for t in transitions]), dtype=torch.float32, device=device)
        rewards = torch.as_tensor(np.stack([t.reward for t in transitions]), dtype=torch.float32, device=device).unsqueeze(-1)
        dones = torch.as_tensor(np.stack([t.done for t in transitions]), dtype=torch.float32, device=device).unsqueeze(-1)
        
        # We now batch BOTH current and next graph data
        graphs = Batch.from_data_list([t.graph_data for t in transitions]).to(device)
        next_graphs = Batch.from_data_list([t.next_graph_data for t in transitions]).to(device)
        
        with torch.no_grad():
            # Select action according to target policy and add clipped noise
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_spatial_obs, next_graphs) + noise).clamp(-1, 1)
            
            # Compute target Q value
            target_Q1 = self.critic1_target(next_spatial_obs, next_graphs, next_actions)
            target_Q2 = self.critic2_target(next_spatial_obs, next_graphs, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * torch.min(target_Q1, target_Q2)
            
        # 2. Update Critics
        current_Q1 = self.critic1(spatial_obs, graphs, actions)
        current_Q2 = self.critic2(spatial_obs, graphs, actions)
        
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)
        
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()
        
        metrics = {
            "train/critic_loss": (critic1_loss + critic2_loss).item() / 2.0,
            "train/mean_q": current_Q1.mean().item()
        }
        
        # 3. Delayed Actor Update
        if self.it % self.policy_freq == 0:
            actor_loss = -self.critic1(spatial_obs, graphs, self.actor(spatial_obs, graphs)).mean()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            # Soft update targets
            for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
                
            metrics["train/actor_loss"] = actor_loss.item()
            
        return metrics
