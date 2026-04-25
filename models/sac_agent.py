import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions import Normal

class SACActor(nn.Module):
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
        self.mu = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Linear(prev_dim, action_dim)
        
    def forward(self, spatial_obs, graph_data):
        features = self.encoder(spatial_obs, graph_data)
        x = self.net(features)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mu, log_std
        
    def sample(self, spatial_obs, graph_data):
        mu, log_std = self.forward(spatial_obs, graph_data)
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound (tanh correction)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

class SACCritic(nn.Module):
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

class SACAgent:
    def __init__(self, actor, critic, lr=3e-4, tau=0.005, gamma=0.99, alpha=0.2):
        self.actor = actor
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic1 = critic
        self.critic1_target = copy.deepcopy(critic)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        
        self.critic2 = copy.deepcopy(critic)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha # Entropy coefficient
        
    def select_action(self, spatial_obs, graph_data):
        with torch.no_grad():
            action, _ = self.actor.sample(spatial_obs, graph_data)
            return action.cpu().numpy().flatten()
            
    def update(self, replay_buffer, batch_size=64):
        transitions = replay_buffer.sample(batch_size)
        from torch_geometric.data import Batch
        device = next(self.actor.parameters()).device
        
        spatial_obs = torch.as_tensor(np.stack([t.spatial_obs for t in transitions]), dtype=torch.float32, device=device)
        next_spatial_obs = torch.as_tensor(np.stack([t.next_spatial_obs for t in transitions]), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.stack([t.action for t in transitions]), dtype=torch.float32, device=device)
        rewards = torch.as_tensor(np.stack([t.reward for t in transitions]), dtype=torch.float32, device=device).unsqueeze(-1)
        dones = torch.as_tensor(np.stack([t.done for t in transitions]), dtype=torch.float32, device=device).unsqueeze(-1)
        
        # Correctly batch BOTH current and next graphs
        graphs = Batch.from_data_list([t.graph_data for t in transitions]).to(device)
        next_graphs = Batch.from_data_list([t.next_graph_data for t in transitions]).to(device)
        
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_spatial_obs, next_graphs)
            target_Q1 = self.critic1_target(next_spatial_obs, next_graphs, next_actions)
            target_Q2 = self.critic2_target(next_spatial_obs, next_graphs, next_actions)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * next_log_probs
            target_Q = rewards + (1 - dones) * self.gamma * target_V
            
        current_Q1 = self.critic1(spatial_obs, graphs, actions)
        current_Q2 = self.critic2(spatial_obs, graphs, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic1_opt.zero_grad(); self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic1_opt.step(); self.critic2_opt.step()
        
        # Actor update
        new_actions, log_probs = self.actor.sample(spatial_obs, graphs)
        Q1 = self.critic1(spatial_obs, graphs, new_actions)
        Q2 = self.critic2(spatial_obs, graphs, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(Q1, Q2)).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # Target update
        for p, pt in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            
        return {
            "train/critic_loss": critic_loss.item(), 
            "train/actor_loss": actor_loss.item(),
            "train/mean_q": current_Q1.mean().item(),
            "train/entropy": -log_probs.mean().item()
        }
