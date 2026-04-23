#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_experiment(algo, seed, config, total_timesteps, run_dir):
    checkpoint_dir = Path(run_dir) / f"{algo}_seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = checkpoint_dir / "training.log"
    
    cmd = [
        sys.executable,
        "training/train.py",
        "--config", config,
        "--algo", algo,
        "--seed", str(seed),
        "--checkpoint_dir", str(checkpoint_dir),
        "--log_file", str(log_file)
    ]
    if total_timesteps:
        cmd.extend(["--total_timesteps", str(total_timesteps)])
    
    print(f"Starting {algo} (seed {seed}) in {checkpoint_dir}")
    try:
        # Use subprocess.run to execute the command
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with open(checkpoint_dir / "process.out", "w") as f:
            f.write(process.stdout)
        
        if process.returncode == 0:
            print(f"[SUCCESS] {algo} (seed {seed})")
            return True
        else:
            print(f"[FAILED] {algo} (seed {seed}). See {checkpoint_dir / 'process.out'}")
            return False
    except Exception as e:
        print(f"[ERROR] {algo} (seed {seed}): {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Schedule parallel training runs")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to base config file")
    parser.add_argument("--algos", type=str, nargs="+", default=["ppo", "td3", "sac"], help="List of algorithms to run (e.g. ppo td3 sac)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45], help="List of seeds")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--run_dir", type=str, default="runs/experiments", help="Base directory for experiment runs")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers")
    
    args = parser.parse_args()
    
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    
    tasks = []
    for algo in args.algos:
        for seed in args.seeds:
            tasks.append((algo, seed, args.config, args.total_timesteps, args.run_dir))
            
    print(f"Scheduling {len(tasks)} runs across {args.max_workers} workers...")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_experiment, algo, seed, cfg, ts, rd): (algo, seed)
            for algo, seed, cfg, ts, rd in tasks
        }
        
        for future in as_completed(futures):
            algo, seed = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                print(f"[EXCEPTION] {algo} (seed {seed}): {e}")
                
    print(f"Finished {success_count}/{len(tasks)} runs successfully.")

if __name__ == "__main__":
    main()
