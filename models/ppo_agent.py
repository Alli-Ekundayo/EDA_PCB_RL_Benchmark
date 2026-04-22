from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Batch

from .networks import DualStreamActorCritic


@dataclass
class RolloutBatch:
    spatial_obs: torch.Tensor
    action_masks: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class PPOAgent:
    def __init__(
        self,
        model: DualStreamActorCritic,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        ent_coef: float = 0.02,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.model = model
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = Adam(model.parameters(), lr=lr)

    def update(
        self,
        rollout: RolloutBatch,
        graph_list,
        n_epochs: int = 4,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        n = rollout.spatial_obs.shape[0]
        losses = []
        entropies = []
        policy_losses = []
        value_losses = []

        # Pre-batch the graphs into fixed mini-batches to save CPU time
        # We shuffle the dataset once, divide into mini-batches, and then
        # in the epochs loop we only shuffle the order of the mini-batches.
        idx = torch.randperm(n)
        mini_batches = []
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            batch_graph = Batch.from_data_list([graph_list[int(i)] for i in b.tolist()]).to(rollout.spatial_obs.device)
            mini_batches.append((b, batch_graph))

        for _ in range(n_epochs):
            batch_order = torch.randperm(len(mini_batches))
            for i in batch_order.tolist():
                b, batch_graph = mini_batches[i]
                out = self.model(batch_graph, rollout.spatial_obs[b])
                dist = self.model.masked_dist(out.logits, rollout.action_masks[b])
                logp = dist.log_prob(rollout.actions[b])
                ratio = torch.exp(logp - rollout.old_log_probs[b])
                adv = rollout.advantages[b]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(out.value, rollout.returns[b])
                entropy = dist.entropy().mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses.append(float(loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))

        return {
            "train/loss": float(np.mean(losses)) if losses else 0.0,
            "train/policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "train/value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "train/entropy": float(np.mean(entropies)) if entropies else 0.0,
        }
