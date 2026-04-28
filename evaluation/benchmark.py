#!/usr/bin/env python
"""Unified benchmark runner for trained policy evaluation."""
import sys
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import Config
from training.logger import log_dict
from evaluation.eval import evaluate


def resolve_checkpoint(checkpoint_dir: str, algo: str = "ppo") -> Path:
    path = Path(checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    final_pattern = f"{algo}_final*.pt" if algo in ("td3", "sac") else f"{algo}_final_step_*.pt"
    finals = sorted(path.glob(final_pattern))
    if finals:
        return finals[-1]

    general = sorted(path.glob(f"{algo}*.pt"))
    if general:
        return general[-1]

    raise FileNotFoundError(f"No {algo.upper()} checkpoint found in {checkpoint_dir}")


def benchmark_on_boards(
    config_path: str = "configs/small_board.yaml",
    board_pattern: str = "small_board_*.json",
    n_episodes_per_board: int = 3,
) -> None:
    """Run benchmark on multiple boards."""
    config = Config.from_yaml(config_path)
    
    board_dir = Path(config.board_dir)
    search_dir = board_dir.parent if board_dir.is_file() else board_dir
    boards = sorted(search_dir.glob(board_pattern))
    if not boards and board_dir.is_file():
        boards = [board_dir]
    if not boards:
        raise FileNotFoundError(f"No boards found in {search_dir} matching {board_pattern}")
    
    print(f"\n=== PCB Placement Benchmark ===")
    print(f"Config: {config_path}")
    print(f"Board pattern: {board_pattern}")
    print(f"Found {len(boards)} boards to evaluate")
    print(f"Episodes per board: {n_episodes_per_board}\n")
    
    results = {
        "config": asdict(config),
        "board_pattern": board_pattern,
        "algorithms": {},
    }
    
    checkpoint = resolve_checkpoint(config.checkpoint_dir, algo=config.algo.lower())
    print(f"Evaluating {config.algo.upper()} (checkpoint: {checkpoint})...")
    board_eval_set: List[str] = [str(p) for p in boards for _ in range(max(1, n_episodes_per_board))]
    eval_metrics = evaluate(str(checkpoint), config, board_files=board_eval_set)
    ppo_result = {
        "algorithm": f"{config.algo.upper()}",
        "n_boards": len(boards),
        "episodes_per_board": n_episodes_per_board,
        **eval_metrics,
    }
    results["algorithms"][config.algo.upper()] = ppo_result
    log_dict({
        **{f"{config.algo}/{k}": v for k, v in eval_metrics.items()},
        f"{config.algo}/n_boards": float(len(boards)),
    })
    
    # Summary
    print(f"\n=== Benchmark Summary ===")
    for algo, result in results["algorithms"].items():
        print(f"\n{algo}:")
        for k, v in result.items():
            if k != 'algorithm':
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Save results
    results_path = Path("runs/benchmarks") / f"benchmark_{Path(board_pattern).stem}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/small_board.yaml")
    parser.add_argument("--pattern", type=str, default="small_board_*.json")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    
    benchmark_on_boards(
        config_path=args.config,
        board_pattern=args.pattern,
        n_episodes_per_board=args.episodes,
    )
