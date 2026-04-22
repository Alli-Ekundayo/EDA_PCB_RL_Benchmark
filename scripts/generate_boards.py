#!/usr/bin/env python
"""Generate synthetic PCB boards for training."""
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np


def generate_component_placement(
    n_components: int,
    n_nets: int,
    seed: int = None,
) -> Tuple[List[dict], dict]:
    """Generate synthetic component and net data."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Component classes: MCU, Passive, Connector, Power
    class_names = ["MCU", "Passive", "Connector", "Power"]
    
    components = []
    for i in range(n_components):
        ref = f"{['U', 'R', 'C', 'J', 'P'][i % 5]}{i+1}"
        class_id = i % 4
        # Randomly assign nets to this component (2-4 nets per component)
        n_comp_nets = random.randint(2, min(4, n_nets))
        nets = [int(x) for x in np.random.choice(n_nets, size=n_comp_nets, replace=False)]
        
        components.append({
            "ref": ref,
            "class_id": class_id,
            "nets": nets,
        })
    
    # Build nets from component connections
    nets = {}
    for net_id in range(n_nets):
        refs = [c["ref"] for c in components if net_id in c["nets"]]
        if len(refs) >= 2:  # Only include nets with 2+ connections
            nets[str(net_id)] = refs
    
    return components, nets


def create_synthetic_boards(
    output_dir: Path,
    n_boards: int = 10,
    small_size: Tuple[int, int] = (5, 15),
    medium_size: Tuple[int, int] = (15, 50),
    large_size: Tuple[int, int] = (50, 200),
) -> None:
    """Create synthetic PCB boards with varying complexity."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    boards_created = []
    
    # Create small boards (training)
    for i in range(n_boards // 3):
        n_comp = random.randint(small_size[0], small_size[1])
        n_nets = random.randint(n_comp // 2, n_comp)
        components, nets = generate_component_placement(n_comp, n_nets, seed=42+i)
        
        payload = {
            "components": components,
            "nets": nets,
        }
        
        fname = output_dir / f"small_board_{i:03d}.json"
        with fname.open("w") as f:
            json.dump(payload, f, indent=2)
        boards_created.append(f"small_board_{i:03d}.json ({n_comp} components, {len(nets)} nets)")
    
    # Create medium boards (training/eval)
    for i in range(n_boards // 3):
        n_comp = random.randint(medium_size[0], medium_size[1])
        n_nets = random.randint(n_comp // 2, n_comp)
        components, nets = generate_component_placement(n_comp, n_nets, seed=100+i)
        
        payload = {
            "components": components,
            "nets": nets,
        }
        
        fname = output_dir / f"medium_board_{i:03d}.json"
        with fname.open("w") as f:
            json.dump(payload, f, indent=2)
        boards_created.append(f"medium_board_{i:03d}.json ({n_comp} components, {len(nets)} nets)")
    
    # Create large boards (eval/stress test)
    for i in range(n_boards // 3):
        n_comp = random.randint(large_size[0], large_size[1])
        n_nets = random.randint(n_comp // 2, n_comp)
        components, nets = generate_component_placement(n_comp, n_nets, seed=200+i)
        
        payload = {
            "components": components,
            "nets": nets,
        }
        
        fname = output_dir / f"large_board_{i:03d}.json"
        with fname.open("w") as f:
            json.dump(payload, f, indent=2)
        boards_created.append(f"large_board_{i:03d}.json ({n_comp} components, {len(nets)} nets)")
    
    print(f"Created {len(boards_created)} synthetic boards in {output_dir}")
    for board in boards_created:
        print(f"  - {board}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/boards/rl_pcb/base")
    parser.add_argument("--n-boards", type=int, default=9)
    args = parser.parse_args()
    
    create_synthetic_boards(
        output_dir=Path(args.output),
        n_boards=args.n_boards,
    )


if __name__ == "__main__":
    main()
