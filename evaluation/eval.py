from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch

from environment.action_projection import continuous_action_to_discrete
from environment.pcb_env import PCBEnv
from environment.reward import hpwl
from models.networks import DualStreamActorCritic, SpatialEncoder, GATEncoder, SharedFusionEncoder
from routing.router import UnifiedPCBRouter
from training.config import Config
from training.graph_utils import graph_to_data


def _graph_to_data(g):
    return graph_to_data(g)


def sync_config_from_checkpoint(checkpoint_path: str, config: Config, device: torch.device):
    """Update a Config object with architecture-relevant settings from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" in checkpoint:
        cp_config = checkpoint["config"]
        # Sync all parameters that affect the model architecture or environment observations
        sync_keys = [
            "gat_embed_dim", "spatial_embed_dim", "fused_dim", "gat_heads", 
            "pi_hidden_sizes", "qf_hidden_sizes", "algo",
            "use_ratsnest", "use_criticality", "component_rotations",
            "board_width", "board_height"
        ]
        if isinstance(cp_config, dict):
            for key in sync_keys:
                if key in cp_config:
                    setattr(config, key, cp_config[key])
        elif isinstance(cp_config, Config):
            for key in sync_keys:
                if hasattr(cp_config, key):
                    setattr(config, key, getattr(cp_config, key))
    return config


def _continuous_action_to_discrete(
    action: np.ndarray,
    action_mask: np.ndarray,
    width: int,
    height: int,
    n_rotations: int,
) -> int:
    return continuous_action_to_discrete(action, action_mask, width, height, n_rotations)


def _select_deterministic_action(
    model,
    algo: str,
    graph_batch: Batch,
    spatial: torch.Tensor,
    action_mask_t: torch.Tensor,
    width: int,
    height: int,
    n_rotations: int,
) -> int:
    if algo == "ppo":
        action_t, _, _ = model.act(graph_batch, spatial, action_mask_t, deterministic=True)
        return int(action_t.item())

    if algo == "td3":
        continuous = model(spatial, graph_batch).detach().cpu().numpy()[0]
        return _continuous_action_to_discrete(continuous, action_mask_t[0].cpu().numpy(), width, height, n_rotations)

    if algo == "sac":
        mu, _log_std = model.forward(spatial, graph_batch)
        continuous = torch.tanh(mu).detach().cpu().numpy()[0]
        return _continuous_action_to_discrete(continuous, action_mask_t[0].cpu().numpy(), width, height, n_rotations)

    raise ValueError(f"Unsupported algorithm for evaluation: {algo}")


def load_model(checkpoint_path: str, config: Config, obs_channels: int, node_feat_dim: int, edge_feat_dim: int, action_dim: int, device: torch.device):
    """Load a trained model from a checkpoint, supporting PPO, TD3, and SAC."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Sync config again just in case, though it should ideally be done before calling this
    sync_config_from_checkpoint(checkpoint_path, config, device)
            
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
        from models.td3_agent import TD3Actor
        from models.sac_agent import SACActor
        
        spatial_enc = SpatialEncoder(in_channels=obs_channels, embed_dim=config.spatial_embed_dim)
        gat_enc = GATEncoder(node_feat_dim, edge_feat_dim, embed_dim=config.gat_embed_dim)
        encoder = SharedFusionEncoder(spatial_enc, gat_enc, config.fused_dim)
        pi_hidden = config.pi_hidden_sizes if config.pi_hidden_sizes else [256]
        
        if algo == "td3":
            model = TD3Actor(encoder, pi_hidden).to(device)
        else:
            model = SACActor(encoder, pi_hidden).to(device)
    else:
        raise ValueError(f"Unsupported algorithm in checkpoint/config: {algo}")

    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint at {checkpoint_path} does not contain 'model' weights")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _infer_raw_kicad_path(board_file: str) -> Optional[str]:
    board_path = Path(board_file)
    raw_dir = board_path.parent.parent / "base_raw"
    if raw_dir.is_dir():
        candidates = sorted(raw_dir.glob("*.kicad_pcb"))
        if candidates:
            return str(candidates[0])
    return None


def evaluate(
    checkpoint_path: str,
    config: Config,
    board_files: Optional[List[str]] = None,
    use_physical_routing: bool = False,
) -> Dict[str, float]:
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sync config from checkpoint to ensure environment matches training
    sync_config_from_checkpoint(checkpoint_path, config, device)
    algo = config.algo.lower()
    
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
    routed_completion_rates: List[float] = []
    routed_wirelengths_mm: List[float] = []

    model = None
    router = UnifiedPCBRouter() if use_physical_routing else None
    for board_file in board_files:
        env = PCBEnv(
            board_path=board_file,
            width=config.board_width,
            height=config.board_height,
            component_rotations=rotations,
            use_ratsnest=config.use_ratsnest,
            use_criticality=config.use_criticality,
        )
        obs, info = env.reset(seed=config.seed)
        graph = graph_to_data(info["graph"])
        action_mask = info["action_mask"]

        if model is None:
            model = load_model(
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
                action = _select_deterministic_action(
                    model=model,
                    algo=algo,
                    graph_batch=graph_batch,
                    spatial=spatial,
                    action_mask_t=action_mask_t,
                    width=config.board_width,
                    height=config.board_height,
                    n_rotations=len(rotations),
                )
            obs, _, terminated, truncated, info = env.step(action)
            graph = graph_to_data(info["graph"])
            action_mask = info["action_mask"]
            invalid += 0 if info.get("valid_action", True) else 1
            steps += 1

        hpwls.append(hpwl(env.board))
        invalid_rates.append(float(invalid) / max(steps, 1))
        lengths.append(steps)

        if use_physical_routing and router is not None:
            routed = router.route(env.board, kicad_pcb_path=_infer_raw_kicad_path(board_file))
            total_nets = max(1, len(env.board.nets))
            routed_count = routed.num_routed_nets()
            routed_completion_rates.append(float(routed_count / total_nets))
            if routed.routed_wirelength >= 0:
                routed_wirelengths_mm.append(float(routed.routed_wirelength))
        env.close()

    out = {
        "eval/hpwl_mean": float(np.mean(hpwls)),
        "eval/hpwl_std": float(np.std(hpwls)),
        "eval/invalid_action_rate": float(np.mean(invalid_rates)),
        "eval/episode_length_mean": float(np.mean(lengths)),
    }
    if use_physical_routing:
        out["eval/routed_completion_rate_mean"] = float(np.mean(routed_completion_rates)) if routed_completion_rates else 0.0
        out["eval/routed_wirelength_mm_mean"] = float(np.mean(routed_wirelengths_mm)) if routed_wirelengths_mm else -1.0
        out["eval/routed_wirelength_available_rate"] = float(len(routed_wirelengths_mm) / max(1, len(board_files)))
    return out
