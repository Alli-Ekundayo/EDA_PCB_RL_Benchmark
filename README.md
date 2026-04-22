# RL-PCB v2: PPO + GNN + SDF Architecture for Automated PCB Placement & Routing

A fully open-source, end-to-end AI-driven EDA pipeline combining:
- **PPO actor-critic** agent with hard DRC action masking
- **Dual-stream encoder**: Graph Attention Network (GAT) + shallow CNN
- **Multi-channel SDF observation space** with criticality weighting
- **Constraint-aware routing** for differential pairs and general nets

## Quick Start

```bash
# 1. Setup environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate sample board data
python scripts/preprocess_boards.py

# 4. Run tests (all pass ✓)
python -m pytest tests/ -v

# 5. Train on sample board
python training/train.py --config configs/base.yaml
```

## Project Structure

```
EDA_PCB/
├── environment/          # Phase 1: Gymnasium PCB environment
│   ├── board.py         # Board state and component definitions
│   ├── pcb_env.py       # Main Gymnasium Env class
│   ├── netlist_parser.py # Parse .kicad_pcb / .net files
│   ├── sdf_generator.py # Multi-channel SDF computation
│   ├── drc_mask.py      # Design Rule Check + action masking
│   ├── ratsnest.py      # Ratsnest density & criticality maps
│   └── reward.py        # HPWL + criticality reward
│
├── models/              # Phase 2 & 3: Neural network modules
│   ├── gat_encoder.py   # Graph Attention Network (topological)
│   ├── spatial_encoder.py # Shallow CNN (spatial stream)
│   ├── fusion.py        # Embedding fusion module
│   ├── actor.py         # PPO Actor + masking
│   ├── critic.py        # Dual-head Critic network
│   └── ppo_agent.py     # PPO update loop, rollout buffer
│
├── training/            # Phase 4: Training pipeline
│   ├── train.py         # Main training entry point
│   ├── config.py        # Hyperparameters dataclass
│   ├── vec_env.py       # Vectorized env wrapper (128 parallel)
│   └── logger.py        # W&B + console logging
│
├── routing/             # Phase 5: Routing extension
│   ├── router.py        # Unified routing algorithm interface
│   ├── diff_pair.py     # Differential pair constraint handler
│   └── via_manager.py   # Via insertion, layer assignment
│
├── evaluation/          # Phase 6: Benchmarking
│   ├── eval.py          # Placement evaluation runner
│   ├── metrics.py       # HPWL, overlap, DRC pass rate
│   └── compare_baselines.py # TD3/SAC vs PPO comparison
│
├── tests/               # Unit & integration tests
│   ├── test_env.py
│   ├── test_sdf.py
│   ├── test_drc.py
│   ├── test_models.py
│   └── test_routing.py
│
├── scripts/             # Utilities
│   ├── preprocess_boards.py # Convert KiCad files
│   └── visualize_placement.py # Render results
│
├── configs/             # Hyperparameter overrides
│   ├── base.yaml        # Default (50x50 board, 16 envs)
│   ├── small_board.yaml # 20x20 board, 8 envs
│   └── large_board.yaml # 100x100 board, 32 envs
│
├── data/
│   ├── boards/          # .kicad_pcb training boards
│   ├── netlists/        # .net files
│   └── benchmarks/      # Held-out test boards
│
├── requirements.txt
├── setup.py
└── README.md
```

## Core Components

### Environment (Phase 1)
- **Board**: Dataclass holding component placement state, nets, keepout zones, and criticality scores
- **SDF Generator**: Computes multi-channel distance transforms per component class + ratsnest density
- **DRC Masking**: Guarantees zero invalid placements via hard masking (not penalties)
- **Ratsnest Maps**: Dynamic density + criticality heatmaps updated after each placement
- **Reward**: Combined HPWL reduction + criticality bonus

### Models (Phase 2 & 3)
- **GATEncoder**: Graph-based topological stream (netlist relationships + edge criticality)
- **SpatialEncoder**: CNN-based spatial stream (SDF channels + ratsnest)
- **FusionModule**: Concatenate + project GAT + CNN embeddings → 256-dim fused state
- **MaskedActor**: Policy outputs probability over grid positions, masked by DRC
- **DualHeadCritic**: Separate value heads for HPWL and differential-pair feasibility

### PPO Training (Phase 4)
- **Config system**: YAML-based hyperparameter management (overridable per experiment)
- **Vectorized environments**: 16–128 parallel PCB placements on CPU, GPU-batched neural updates
- **Rollout buffer**: Stores trajectories, computes advantages with GAE
- **PPO update**: Clipped surrogate + value losses + entropy regularization

### Routing (Phase 5)
- **UnifiedPCBRouter**: Interface for post-placement routing
- **DiffPairRouter**: Enforces matched length, spacing, impedance for differential pairs
- **ViaManager**: Layer assignment and via constraint tracking

### Evaluation (Phase 6)
- **Metrics**: HPWL, DRC pass rate, overlap, diff-pair compliance
- **Baselines**: Comparison harness for TD3, SAC, and PPO

## Configuration

All training hyperparameters are in `configs/base.yaml`. Override any value:

```bash
# Use different board size
python training/train.py --config configs/large_board.yaml

# Or create custom config
cat > configs/custom.yaml <<EOF
board_width: 75
n_envs: 24
total_timesteps: 500000
EOF
python training/train.py --config configs/custom.yaml
```

### Key Hyperparameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `n_envs` | 16 | Parallel environments (CPU vectorized) |
| `n_steps` | 256 | Rollout length per update |
| `n_epochs` | 4 | PPO epochs per rollout |
| `batch_size` | 256 | Minibatch size for gradient updates |
| `clip_range` | 0.2 | PPO clip ratio ε |
| `lr` | 3e-4 | Learning rate (linear decay to 1e-5) |
| `ent_coef` | 0.01 | Entropy bonus (exploration) |
| `vf_coef_hpwl` | 0.5 | Weight for HPWL critic head |
| `vf_coef_diffp` | 0.2 | Weight for diff-pair critic head |

## Test Suite

All unit tests pass:
```bash
pytest tests/ -v
```

- `test_sdf.py`: SDF shape, non-negative values, occupied = 0
- `test_drc.py`: Masking blocks all overlaps + keepout violations
- `test_env.py`: Environment step/reset contract, termination logic
- `test_models.py`: Actor probability sums to 1, critic shapes correct
- `test_routing.py`: Router returns valid RoutedBoard structure

## Dataset

Sample board generated automatically by `scripts/preprocess_boards.py`:
- Located at `data/boards/sample_board.json`
- 3 components (U1=MCU, R1=Resistor, C1=Capacitor)
- 2 nets with criticality scoring

To add your own boards:
1. Export from KiCad as `.kicad_pcb` or `.net`
2. Place in `data/boards/`
3. Parser auto-detects format (handles both)

### RL_PCB Imported Dataset + Benchmarks

This repo now includes reference assets extracted from `Projects/RL_PCB`:
- Dataset snapshots in `data/boards/rl_pcb/{base,base_opt,base_raw}`
- Hyperparameter defaults in `data/benchmarks/hp_td3.json` and `data/benchmarks/hp_sac.json`
- Expected evaluation reports in `data/benchmarks/rl_pcb_reports/`
- Parsed benchmark targets in `data/benchmarks/rl_pcb_model_benchmarks.json`

`evaluation/compare_baselines.py` reads these benchmark targets (TD3/SAC/DreamerV3) and exposes them as the baseline for current implementation checks.

## Key Design Decisions

### Hard DRC Masking
Action masking prevents invalid placements structurally, not via penalties. This ensures:
- All 128 parallel episodes remain valid
- No exploration waste on impossible actions
- Faster convergence

### Multi-Channel SDF
Separate distance transforms per component class allow the spatial encoder to:
- Learn class-specific avoidance patterns
- Capture geometric relationships
- Provide smooth gradients (vs. binary grids)

### Dual-Head Critic
- **Head 1 (HPWL)**: Primary optimization target, weighted 0.5
- **Head 2 (Diff-Pair)**: Secondary signal for high-priority nets, weighted 0.2
- Disentangles competing objectives without needing multi-task learning tricks

### GAT with Edge Criticality
Graph attention naturally learns to weight critical connections (e.g., differential pairs) higher, without explicit architecture changes.

## Roadmap

| Milestone | Status |
|-----------|--------|
| Phase 1 (Environment) | ✓ Complete |
| Phase 2 (Encoders) | ✓ Complete |
| Phase 3 (PPO Agent) | ✓ Complete |
| Phase 4 (Training) | ✓ Complete |
| Phase 5 (Routing) | Scaffolded |
| Phase 6 (Eval) | Scaffolded |
| M3: Train on simple board | Ready |
| M4: Baseline comparison | Ready |
| M5: Full pipeline | Next |

## References

- Hafner et al. (2023). Mastering Diverse Domains through World Models. arXiv:2301.04104
- Fujimoto et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods. ICML
- Haarnoja et al. (2018). Soft Actor-Critic. ICML
- Mirhoseini et al. (2021). Graph placement for chip design. Nature 594
- ASPDAC 2021. Unified PCB routing with differential pairs. https://doi.org/10.1145/3394885.3431568
