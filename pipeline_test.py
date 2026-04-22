import os
import subprocess
import yaml
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

def run_integration_test():
    print("=== Starting RL_PCB Integration Test ===")
    
    # 1. Create a dummy config for fast testing
    test_config = {
        "seed": 42,
        "board_dir": "data/boards/synthetic",
        "checkpoint_dir": "runs/test_run/checkpoints",
        "board_width": 32,
        "board_height": 32,
        "n_envs": 2,
        "n_steps": 16,
        "total_timesteps": 64,  # Very short for testing
        "lr": 0.001,
        "batch_size": 16,
        "n_epochs": 1,
        "checkpoint_freq": 32
    }
    
    config_path = Path("configs/test_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)
    
    print(f"Created test config at {config_path}")

    # 2. Run training
    print("\n--- Phase 1: Training ---")
    train_cmd = [
        "python3", "training/train.py",
        "--config", str(config_path)
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":."
    
    try:
        subprocess.run(train_cmd, check=True, env=env)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return

    # 3. Identify the latest checkpoint
    checkpoint_dir = Path(test_config["checkpoint_dir"])
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        print("No checkpoints found!")
        return
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print(f"\nLatest checkpoint: {latest_ckpt}")

    # 4. Run evaluation
    print("\n--- Phase 2: Evaluation ---")
    # We'll use a small script to call evaluation.eval.evaluate
    eval_script = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from evaluation.eval import evaluate
from training.config import Config

config = Config.from_yaml("{config_path}")
results = evaluate("{latest_ckpt}", config)
print("Eval Results:", results)
    """
    try:
        subprocess.run(["python3", "-c", eval_script], check=True, env=env)
        print("Evaluation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}")

    # 5. Mock Notebook Experiment: Visualization
    print("\n--- Phase 3: Notebook Experiment (Visualization) ---")
    # Generate a dummy plot to simulate a notebook result
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, label="Reward Proxy", color='cyan')
    plt.title("RL_PCB Training Progress (Mock)")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    vis_path = Path("runs/test_run/mock_progress.png")
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(vis_path)
    print(f"Mock visualization saved to {vis_path}")

    print("\n=== Integration Test Finished Successfully ===")

if __name__ == "__main__":
    run_integration_test()
