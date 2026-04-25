#!/bin/bash
# Diagnostic Run: HPWL Only
# Purpose: Check if model can learn wire-length optimization in isolation

source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

OUTPUT_DIR="runs/diagnostic_hpwl"
mkdir -p $OUTPUT_DIR

echo "=== Starting Diagnostic HPWL-Only Run ==="
python3 training/train.py \
    --config configs/diagnostic_hpwl.yaml \
    --algo ppo \
    --total_timesteps 50000 \
    --checkpoint_dir "$OUTPUT_DIR/checkpoints" \
    --log_file "$OUTPUT_DIR/training.log" \
    --seed 42

echo "=== Generating Diagnostic Report ==="
python3 scripts/report_generator.py \
    --work_dir "$OUTPUT_DIR" \
    --out "$OUTPUT_DIR/diagnostic_report.pdf"

echo "Diagnostic completed. Check $OUTPUT_DIR/diagnostic_report.pdf"
