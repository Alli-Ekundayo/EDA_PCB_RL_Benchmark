#!/bin/bash
# PPO Research Run Script
# Replicating the RL_PCB project's experimental workflow

# 1. Setup Environment
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

EXPERIMENT_NAME="01_ppo_research_run"
OUTPUT_DIR="tests/$EXPERIMENT_NAME/results"
mkdir -p $OUTPUT_DIR

echo "=== Starting Research Run: $EXPERIMENT_NAME ==="

# 2. Training Phase
echo "--- Phase 1: Training ---"
python3 training/train.py \
    --config configs/base.yaml \
    --total_timesteps 10000 \
    --checkpoint_dir "$OUTPUT_DIR/checkpoints" \
    --log_file "$OUTPUT_DIR/training.log"

# 3. Evaluation & Video Generation
echo "--- Phase 2: Evaluation & Visualization ---"
LATEST_CHECKPOINT=$(ls -t $OUTPUT_DIR/checkpoints/*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $OUTPUT_DIR/checkpoints"
    exit 1
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"

# We'll use a custom script to generate the video
python3 -c "
import torch
import numpy as np
from environment.pcb_env import PCBEnv
from training.config import Config
from evaluation.eval import load_model
from training.train import _graph_to_data

config = Config()
device = torch.device('cpu')
env = PCBEnv(board_dir='data/boards/rl_pcb/base_opt/evaluation.pcb')
obs, info = env.reset()

# Infer dimensions for model loading
in_channels = obs.shape[0]
graph_data = _graph_to_data(info['graph'])
node_feat_dim = graph_data.x.shape[1]
edge_feat_dim = graph_data.edge_attr.shape[1] if graph_data.edge_attr.numel() > 0 else 4
action_dim = env.action_space.n

model = load_model('$LATEST_CHECKPOINT', config, in_channels, node_feat_dim, edge_feat_dim, action_dim, device)

terminated = truncated = False
while not (terminated or truncated):
    spatial_t = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
    action_mask_t = torch.as_tensor(info['action_mask'][None, ...], dtype=torch.bool, device=device)
    batch_graph = torch.utils.data.DataLoader([graph_data], batch_size=1) # Simplified
    from torch_geometric.data import Batch
    bg = Batch.from_data_list([graph_data]).to(device)
    
    with torch.no_grad():
        action_t, _, _ = model.act(bg, spatial_t, action_mask_t, deterministic=True)
    
    action = action_t.item()
    obs, reward, terminated, truncated, info = env.step(action)
    graph_data = _graph_to_data(info['graph'])

env.tracker.save_video('$OUTPUT_DIR/placement_animation.mp4')
"

# 4. Report Generation
echo "--- Phase 3: Report Generation ---"
python3 scripts/report_generator.py \
    --log "$OUTPUT_DIR/training.log" \
    --out "$OUTPUT_DIR/research_report.pdf"

echo "=== Research Run Completed! ==="
echo "Results available in: $OUTPUT_DIR"
