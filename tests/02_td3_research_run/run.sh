#!/bin/bash
# TD3 Research Run Script
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

EXPERIMENT_NAME="02_td3_research_run"
OUTPUT_DIR="tests/$EXPERIMENT_NAME/results"
mkdir -p $OUTPUT_DIR

echo "=== Starting Research Run: $EXPERIMENT_NAME ==="

# 1. Training Phase
echo "--- Phase 1: Training ---"
python3 training/train.py \
    --config configs/base.yaml \
    --algo td3 \
    --total_timesteps 10000 \
    --checkpoint_dir "$OUTPUT_DIR/checkpoints" \
    --log_file "$OUTPUT_DIR/training.log"

# 2. Evaluation & Video Generation
echo "--- Phase 2: Evaluation & Visualization ---"
LATEST_CHECKPOINT=$(ls -t $OUTPUT_DIR/checkpoints/*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_CHECKPOINT" ]; then echo "Error: No checkpoint found"; exit 1; fi

python3 -c "
import torch
from environment.pcb_env import PCBEnv
from environment.wrappers import ContinuousToDiscrete
from training.config import Config
from evaluation.eval import load_model
from training.train import _graph_to_data
from torch_geometric.data import Batch

config = Config(algo='td3')
device = torch.device('cpu')
base_env = PCBEnv(board_dir='data/boards/rl_pcb/base_opt/evaluation.pcb')
env = ContinuousToDiscrete(base_env)
obs, info = env.reset()

# Infer dimensions
in_channels = obs.shape[0]
graph_data = _graph_to_data(info['graph'])
node_feat_dim = graph_data.x.shape[1]
edge_feat_dim = graph_data.edge_attr.shape[1] if graph_data.edge_attr.numel() > 0 else 4
action_dim = 3 # Continuous space

model = load_model('$LATEST_CHECKPOINT', config, in_channels, node_feat_dim, edge_feat_dim, action_dim, device)

terminated = truncated = False
while not (terminated or truncated):
    spatial_t = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
    bg = Batch.from_data_list([graph_data]).to(device)
    with torch.no_grad():
        action = model(spatial_t, bg).cpu().numpy().flatten()
    obs, reward, terminated, truncated, info = env.step(action)
    graph_data = _graph_to_data(info['graph'])

env.unwrapped.tracker.save_video('$OUTPUT_DIR/placement_animation.mp4')
"

# 3. Report Generation
echo "--- Phase 3: Report Generation ---"
python3 scripts/report_generator.py --log "$OUTPUT_DIR/training.log" --out "$OUTPUT_DIR/research_report.pdf"
echo "=== Research Run Completed! ==="
