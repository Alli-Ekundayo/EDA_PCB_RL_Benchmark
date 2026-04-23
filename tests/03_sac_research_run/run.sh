#!/bin/bash
# SAC Research Run Script
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

EXPERIMENT_NAME="03_sac_research_run"
OUTPUT_DIR="tests/$EXPERIMENT_NAME/results"
mkdir -p $OUTPUT_DIR

echo "=== Starting Research Run: $EXPERIMENT_NAME ==="

# 1. Training Phase (Parallel with Auto-Seeds)
echo "--- Phase 1: Training (Parallel) ---"
python3 scripts/scheduler.py \
    --config configs/base.yaml \
    --algos sac \
    --auto_seeds \
    --num_seeds 3 \
    --total_timesteps 10000 \
    --run_dir "$OUTPUT_DIR" \
    --max_workers 3

# 2. Evaluation & Video Generation
echo "--- Phase 2: Evaluation & Visualization ---"
# Find the latest checkpoint from any seed directory
LATEST_CHECKPOINT=$(find $OUTPUT_DIR -name "*.pt" -printf '%T+ %p\n' | sort -r | head -1 | cut -d' ' -f2)
if [ -z "$LATEST_CHECKPOINT" ]; then echo "Error: No checkpoint found"; exit 1; fi

echo "Using checkpoint for visualization: $LATEST_CHECKPOINT"

python3 -c "
import torch
from environment.pcb_env import PCBEnv
from environment.wrappers import ContinuousToDiscrete
from training.config import Config
from evaluation.eval import load_model
from training.train import _graph_to_data
from torch_geometric.data import Batch

config = Config(algo='sac')
device = torch.device('cpu')
# Use evaluation board
base_env = PCBEnv(board_path='data/boards/rl_pcb/base/evaluation.pcb')
env = ContinuousToDiscrete(base_env)
obs, info = env.reset()

# Infer dimensions
in_channels = obs.shape[0]
graph_data = _graph_to_data(info['graph'])
node_feat_dim = graph_data.x.shape[1]
edge_feat_dim = graph_data.edge_attr.shape[1] if graph_data.edge_attr.numel() > 0 else 4
action_dim = 3

model = load_model('$LATEST_CHECKPOINT', config, in_channels, node_feat_dim, edge_feat_dim, action_dim, device)

terminated = truncated = False
while not (terminated or truncated):
    spatial_t = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
    bg = Batch.from_data_list([graph_data]).to(device)
    with torch.no_grad():
        # For SAC evaluation, we use the deterministic mean
        mu, _ = model(spatial_t, bg)
        action = torch.tanh(mu).cpu().numpy().flatten()
    obs, reward, terminated, truncated, info = env.step(action)
    graph_data = _graph_to_data(info['graph'])

env.unwrapped.tracker.save_video('$OUTPUT_DIR/placement_animation.mp4')
"

# 3. Aggregated Report Generation
echo "--- Phase 3: Aggregated Report Generation ---"
python3 scripts/report_generator.py \
    --work_dir "$OUTPUT_DIR" \
    --out "$OUTPUT_DIR/research_report.pdf"

echo "=== Research Run Completed! ==="
