from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from environment.pcb_env import PCBEnv
from environment.reward import hpwl
from models.networks import DualStreamActorCritic
from training.config import Config


def _graph_to_data(g) -> Data:
    return Data(
        x=torch.as_tensor(g.node_features, dtype=torch.float32),
        edge_index=torch.as_tensor(g.edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(g.edge_attr, dtype=torch.float32),
    )


def load_model(checkpoint_path: str, config: Config, obs_channels: int, node_feat_dim: int, edge_feat_dim: int, action_dim: int, device: torch.device):
    """Load a trained model from a checkpoint, supporting PPO, TD3, and SAC."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    algo = config.algo.lower()
    
    if algo == "ppo":
        model = DualStreamActorCritic(
            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim,
            in_channels=obs_channels, action_dim=action_dim,
            gat_dim=config.gat_embed_dim, spatial_dim=config.spatial_embed_dim,
            fused_dim=config.fused_dim, gat_heads=config.gat_heads,
        ).to(device)
    elif algo == "td3" or algo == "sac":
        # Off-policy algorithms use a shared encoder but different heads
        from models.networks import SpatialEncoder, GATEncoder
        from models.td3_agent import TD3Actor
        from models.sac_agent import SACActor
        
        spatial_enc = SpatialEncoder(in_channels=obs_channels, embed_dim=config.spatial_embed_dim)
        gat_enc = GATEncoder(node_feat_dim, edge_feat_dim, embed_dim=config.gat_embed_dim)
        
        class SharedEncoder(torch.nn.Module):
            def __init__(self, s_enc, g_enc, fused_dim):
                super().__init__()
                self.spatial_enc, self.gat_enc, self.fused_dim = s_enc, g_enc, fused_dim
                self.fusion = torch.nn.Linear(s_enc.embed_dim + g_enc.embed_dim, fused_dim)
            def forward(self, s, g):
                return torch.relu(self.fusion(torch.cat([self.spatial_enc(s), self.gat_enc(g)], dim=-1)))
        
        encoder = SharedEncoder(spatial_enc, gat_enc, config.fused_dim)
        if algo == "td3":
            model = TD3Actor(encoder, 256).to(device)
        else:
            model = SACActor(encoder, 256).to(device)
            
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate(checkpoint_path: str, config: Config, board_files: Optional[List[str]] = None) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rotations = tuple(90 * i for i in range(config.component_rotations))
    
    if board_files is None:
        path = Path(config.board_dir)
        if path.is_file():
            # If the config points to training.pcb, we automatically look for evaluation.pcb
            if path.name == "training.pcb":
                eval_path = path.parent / "evaluation.pcb"
                board_files = [str(eval_path)] if eval_path.exists() else [str(path)]
            else:
                board_files = [str(path)]
        else:
            board_files = [str(p) for p in path.glob("*.pcb")]
            if not board_files:
                board_files = [str(p) for p in path.glob("*.json")]
    
    if not board_files:
        raise FileNotFoundError(f"No evaluation boards found for {config.board_dir}")

    hpwls: List[float] = []
    invalid_rates: List[float] = []
    lengths: List[int] = []

    model = None
    for board_file in board_files:
        env = PCBEnv(
            board_path=board_file,
            width=config.board_width,
            height=config.board_height,
            component_rotations=rotations,
        )
        obs, info = env.reset(seed=config.seed)
        graph = _graph_to_data(info["graph"])
        action_mask = info["action_mask"]

        if model is None:
            model = _load_model(
                checkpoint_path=checkpoint_path,
                config=config,
                obs_channels=obs.shape[0],
                node_feat_dim=graph.x.shape[1],
                edge_feat_dim=graph.edge_attr.shape[1] if graph.edge_attr.numel() > 0 else 4,
                action_dim=env.action_space.n,
                device=device,
            )

        terminated = truncated = False
        invalid = 0
        steps = 0
        while not (terminated or truncated):
            spatial = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
            graph_batch = Batch.from_data_list([graph]).to(device)
            action_mask_t = torch.as_tensor(action_mask[None, ...], dtype=torch.bool, device=device)
            with torch.no_grad():
                action_t, _, _ = model.act(graph_batch, spatial, action_mask_t, deterministic=True)
            action = int(action_t.item())
            obs, _, terminated, truncated, info = env.step(action)
            graph = _graph_to_data(info["graph"])
            action_mask = info["action_mask"]
            invalid += 0 if info.get("valid_action", True) else 1
            steps += 1

        hpwls.append(hpwl(env.board))
        invalid_rates.append(float(invalid) / max(steps, 1))
        lengths.append(steps)
        env.close()

    return {
        "eval/hpwl_mean": float(np.mean(hpwls)),
        "eval/hpwl_std": float(np.std(hpwls)),
        "eval/invalid_action_rate": float(np.mean(invalid_rates)),
        "eval/episode_length_mean": float(np.mean(lengths)),
    }
