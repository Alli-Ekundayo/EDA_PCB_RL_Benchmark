#!/usr/bin/env python3
import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.pcb_env import PCBEnv
from training.config import Config
from evaluation.eval import load_model, _graph_to_data, sync_config_from_checkpoint
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
    rotations = tuple(90 * i for i in range(config.component_rotations))
    env = PCBEnv(
        board_path=args.board, 
        width=config.board_width, 
        height=config.board_height,
        component_rotations=rotations
    )
    obs, info = env.reset()
    graph = _graph_to_data(info["graph"])
    
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
            action_t, _, _ = model.act(batch_g, spatial_t, mask_t, deterministic=True)
        
        action = int(action_t.item())
        obs, reward, terminated, truncated, info = env.step(action)
        graph = _graph_to_data(info["graph"])

    # 4. Plot
    board = env.board
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, board.width)
    ax.set_ylim(0, board.height)
    ax.invert_yaxis()
    
    # Draw keepouts
    for x in range(board.width):
        for y in range(board.height):
            if board.keepout[x, y]:
                ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor='#e74c3c', alpha=0.3))
                
    # Draw components
    for comp in board.components:
        if comp.placed:
            x, y = comp.position
            fp = comp.footprint_for_rotation()
            w, h = fp.shape
            for dx in range(w):
                for dy in range(h):
                    if fp[dx, dy]:
                        ax.add_patch(patches.Rectangle((x+dx, y+dy), 1, 1, facecolor='#3498db', alpha=0.7, edgecolor='#2980b9'))
            ax.text(x + w/2, y + h/2, comp.ref, ha='center', va='center', color='black', fontweight='bold', fontsize=8)

    # Draw nets
    centers = {}
    for comp in board.components:
        if comp.placed:
            fp = comp.footprint_for_rotation()
            w, h = fp.shape
            centers[comp.ref] = (comp.position[0] + w/2, comp.position[1] + h/2)
            
    for net_id, refs in board.nets.items():
        placed_refs = [r for r in refs if r in centers]
        for i in range(len(placed_refs)):
            for j in range(i+1, len(placed_refs)):
                p1 = centers[placed_refs[i]]
                p2 = centers[placed_refs[j]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#2ecc71', alpha=0.5, linestyle='--', linewidth=1.5)

    plt.title(f"Placement Results: {os.path.basename(args.checkpoint)}")
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {args.out}")

if __name__ == "__main__":
    main()
