#!/usr/bin/env python3
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.pcb_env import PCBEnv
from training.config import Config
from evaluation.eval import load_model, sync_config_from_checkpoint, _select_deterministic_action
from evaluation.plotting import plot_placement
from training.graph_utils import graph_to_data
from torch_geometric.data import Batch

def main():
    parser = argparse.ArgumentParser(description="Visualize the placement produced by a trained model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--board", type=str, default="data/boards/rl_pcb/base/evaluation.pcb", help="Path to board file")
    parser.add_argument("--out", type=str, default="placement_viz.png", help="Output image path")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    args = parser.parse_args()

    config = Config()
    device = "cpu"
    
    # 0. Sync config from checkpoint to get correct dimensions/rotations
    sync_config_from_checkpoint(args.checkpoint, config, device)
    
    # 1. Initialize environment (using synced config)
    board_width = args.width if args.width > 0 else config.board_width
    board_height = args.height if args.height > 0 else config.board_height
    rotations = tuple(90 * i for i in range(config.component_rotations))
    env = PCBEnv(
        board_path=args.board, 
        width=board_width, 
        height=board_height,
        component_rotations=rotations,
        use_ratsnest=config.use_ratsnest,
        use_criticality=config.use_criticality,
    )
    obs, info = env.reset()
    graph = graph_to_data(info["graph"])
    
    # 2. Load model (using correct dimensions from environment)
    model = load_model(
        checkpoint_path=args.checkpoint,
        config=config,
        obs_channels=obs.shape[0],
        node_feat_dim=graph.x.shape[1],
        edge_feat_dim=graph.edge_attr.shape[1] if graph.edge_attr.numel() > 0 else 4,
        action_dim=env.action_space.n,
        device=device
    )

    # 3. Step through episode
    terminated = truncated = False
    while not (terminated or truncated):
        spatial = np.expand_dims(obs, axis=0)
        spatial_t = torch.as_tensor(spatial, dtype=torch.float32)
        batch_g = Batch.from_data_list([graph])
        mask_t = torch.as_tensor(info["action_mask"][None, ...], dtype=torch.bool)
        
        with torch.no_grad():
            action = _select_deterministic_action(
                model=model,
                algo=config.algo.lower(),
                graph_batch=batch_g,
                spatial=spatial_t,
                action_mask_t=mask_t,
                width=board_width,
                height=board_height,
                n_rotations=len(rotations),
            )
        obs, reward, terminated, truncated, info = env.step(action)
        graph = graph_to_data(info["graph"])

    # 4. Plot
    board = env.board
    plot_placement(board, args.out)
    print(f"Visualization saved to {args.out}")

if __name__ == "__main__":
    main()
