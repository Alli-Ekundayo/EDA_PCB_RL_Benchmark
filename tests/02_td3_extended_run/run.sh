#!/bin/bash
# TD3 Extended Research Run Script
# Implementing recommendations: 4 seeds, 200K steps, optimized stability

# 1. Setup Environment
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

EXPERIMENT_NAME="02_td3_extended_run"
OUTPUT_DIR="tests/$EXPERIMENT_NAME/results"
mkdir -p $OUTPUT_DIR

echo "=== Starting Extended TD3 Research Run: $EXPERIMENT_NAME ==="

# 1.5 Clean stale root-level logs
rm -f "$OUTPUT_DIR/training.log"

# 2. Training Phase (Parallel with 4 Seeds)
echo "--- Phase 1: Training (TD3 @ 200k steps, 4 seeds) ---"
python3 scripts/scheduler.py \
    --config configs/td3_optimized.yaml \
    --algos td3 \
    --auto_seeds \
    --num_seeds 4 \
    --run_dir "$OUTPUT_DIR" \
    --max_workers 2  # Reduced workers to ensure stability on high-memory GAT runs

# 3. Evaluation & Report Generation
echo "--- Phase 2: Generating Reports ---"

# A. Experiment Report (Multi-seed Learning curves)
# This will now correctly filter TD3 hyperparameters thanks to the fix in report_generator.py
python3 scripts/report_generator.py \
    --work_dir "$OUTPUT_DIR" \
    --out "$OUTPUT_DIR/experiment_report.pdf"

# B. Physical Evaluation (using latest final checkpoint)
LATEST_CHECKPOINT=$(find $OUTPUT_DIR -name "*final*.pt" | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Running physical evaluation on: $LATEST_CHECKPOINT"
    python3 scripts/evaluate_model.py \
        --work_dir "$OUTPUT_DIR" \
        --board_file "data/boards/rl_pcb/base/evaluation.pcb" \
        --config "configs/td3_optimized.yaml" \
        --out_dir "$OUTPUT_DIR"
else
    echo "Warning: No final checkpoint found for physical evaluation."
fi

echo "=== Research Run Completed! ==="
echo "Report: $OUTPUT_DIR/experiment_report.pdf"
