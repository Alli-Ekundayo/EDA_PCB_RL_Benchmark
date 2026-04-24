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

# 1.5 Clean stale root-level logs so the report always reads fresh per-seed logs
rm -f "$OUTPUT_DIR/training.log"

# 2. Training Phase (Parallel with Auto-Seeds)
echo "--- Phase 1: Training (Parallel) ---"
python3 scripts/scheduler.py \
    --config configs/base.yaml \
    --algos ppo \
    --auto_seeds \
    --num_seeds 4 \
    --run_dir "$OUTPUT_DIR" \
    --max_workers 3

# 3. Evaluation & Video Generation
echo "--- Phase 2: Evaluation & Visualization ---"
# Find the latest checkpoint from any seed directory
LATEST_CHECKPOINT=$(find $OUTPUT_DIR -name "*.pt" -printf '%T+ %p\n' | sort -r | head -1 | cut -d' ' -f2)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $OUTPUT_DIR"
    exit 1
fi

echo "Using checkpoint for visualization: $LATEST_CHECKPOINT"

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
# Use evaluation board
env = PCBEnv(board_path='data/boards/rl_pcb/base/evaluation.pcb')
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
    from torch_geometric.data import Batch
    bg = Batch.from_data_list([graph_data]).to(device)
    
    with torch.no_grad():
        action_t, _, _ = model.act(bg, spatial_t, action_mask_t, deterministic=True)
    
    action = int(action_t.item())
    obs, reward, terminated, truncated, info = env.step(action)
    graph_data = _graph_to_data(info['graph'])

env.tracker.save_video('$OUTPUT_DIR/placement_animation.mp4')
"

# 4. Report Generation (Dual Pipeline)
echo "--- Phase 3: Generating Dual Research Reports ---"

# A. Experiment Report (Training Focus)
python3 scripts/report_generator.py \
    --work_dir "$OUTPUT_DIR" \
    --out "$OUTPUT_DIR/experiment_report.pdf"

# B. Evaluation Report (Physical Layout Focus)
python3 scripts/evaluate_model.py \
    --work_dir "$OUTPUT_DIR" \
    --board_file "data/boards/rl_pcb/base/evaluation.pcb" \
    --config "configs/test_run.yaml" \
    --out_dir "$OUTPUT_DIR"

echo "=== Research Run Completed! ==="
echo "Experiment Report: $OUTPUT_DIR/experiment_report.pdf"
echo "Evaluation Report: $OUTPUT_DIR/evaluation_report.pdf"
echo "Results available in: $OUTPUT_DIR"
