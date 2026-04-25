#!/bin/bash
# Tiny Board Sanity Check
# Goal: Verify learning on an 8x8 board

source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

OUTPUT_DIR="runs/tiny_sanity"
mkdir -p $OUTPUT_DIR

echo "=== Starting Tiny Board Sanity Check ==="
python3 training/train.py \
    --config configs/tiny_board.yaml \
    --algo ppo \
    --total_timesteps 10000 \
    --checkpoint_dir "$OUTPUT_DIR/checkpoints" \
    --log_file "$OUTPUT_DIR/training.log" \
    --seed 42

echo "=== Visualizing Tiny Placement ==="
# Find final checkpoint
LATEST_CHECKPOINT=$(find $OUTPUT_DIR -name "*final*.pt" | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    python3 scripts/visualize_placement.py \
        --checkpoint "$LATEST_CHECKPOINT" \
        --board "data/boards/rl_pcb/base/evaluation.pcb" \
        --width 8 \
        --height 8 \
        --out "$OUTPUT_DIR/tiny_placement.png"
fi

echo "=== Generating Report ==="
python3 scripts/report_generator.py \
    --work_dir "$OUTPUT_DIR" \
    --out "$OUTPUT_DIR/tiny_report.pdf"

echo "Sanity check completed. Check $OUTPUT_DIR/tiny_report.pdf and tiny_placement.png"
