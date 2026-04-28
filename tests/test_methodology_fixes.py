from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from evaluation import benchmark as benchmark_mod
from evaluation import eval as eval_mod
from training import train as train_mod
from training.config import Config


def test_eval_select_action_for_td3_without_ppo_act():
    class DummyTD3:
        def __call__(self, spatial, graph_batch):
            return torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    action_mask_t = torch.as_tensor([[False, False, True, False]], dtype=torch.bool)
    action = eval_mod._select_deterministic_action(
        model=DummyTD3(),
        algo="td3",
        graph_batch=object(),
        spatial=torch.zeros((1, 1), dtype=torch.float32),
        action_mask_t=action_mask_t,
        width=2,
        height=2,
        n_rotations=1,
    )
    assert action == 2


def test_eval_select_action_for_ppo_uses_act():
    class DummyPPO:
        def act(self, graph_batch, spatial, action_mask_t, deterministic=True):
            return torch.tensor([1]), torch.tensor([0.0]), torch.tensor([0.0])

    action = eval_mod._select_deterministic_action(
        model=DummyPPO(),
        algo="ppo",
        graph_batch=object(),
        spatial=torch.zeros((1, 1), dtype=torch.float32),
        action_mask_t=torch.as_tensor([[True, True]], dtype=torch.bool),
        width=1,
        height=2,
        n_rotations=1,
    )
    assert action == 1


def test_benchmark_uses_discovered_boards_and_repeats_episodes(tmp_path, monkeypatch):
    boards_dir = tmp_path / "boards"
    boards_dir.mkdir(parents=True)
    (boards_dir / "a.json").write_text("{}", encoding="utf-8")
    (boards_dir / "b.json").write_text("{}", encoding="utf-8")

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ppo_final_step_10.pt").write_bytes(b"x")

    cfg_path = tmp_path / "cfg.yaml"
    cfg = {
        "algo": "ppo",
        "board_dir": str(boards_dir),
        "checkpoint_dir": str(ckpt_dir),
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    captured = {}

    def fake_evaluate(checkpoint_path, config, board_files=None):
        captured["checkpoint_path"] = checkpoint_path
        captured["board_files"] = list(board_files or [])
        return {"eval/hpwl_mean": 1.0, "eval/hpwl_std": 0.0, "eval/invalid_action_rate": 0.0, "eval/episode_length_mean": 3.0}

    monkeypatch.setattr(benchmark_mod, "evaluate", fake_evaluate)
    benchmark_mod.benchmark_on_boards(str(cfg_path), board_pattern="*.json", n_episodes_per_board=2)

    assert len(captured["board_files"]) == 4
    assert str(boards_dir / "a.json") in captured["board_files"]
    assert str(boards_dir / "b.json") in captured["board_files"]


def test_benchmark_raises_when_checkpoint_missing(tmp_path):
    boards_dir = tmp_path / "boards"
    boards_dir.mkdir(parents=True)
    (boards_dir / "a.json").write_text("{}", encoding="utf-8")

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"algo": "ppo", "board_dir": str(boards_dir), "checkpoint_dir": str(ckpt_dir)}),
        encoding="utf-8",
    )

    try:
        benchmark_mod.benchmark_on_boards(str(cfg_path), board_pattern="*.json", n_episodes_per_board=1)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for missing checkpoint.")


def test_set_global_seed_reproducible():
    train_mod.set_global_seed(123, torch_deterministic=False)
    a_np = np.random.rand()
    a_t = torch.rand(1).item()

    train_mod.set_global_seed(123, torch_deterministic=False)
    b_np = np.random.rand()
    b_t = torch.rand(1).item()

    assert a_np == b_np
    assert a_t == b_t


def test_train_ppo_wires_config_into_agent(tmp_path, monkeypatch):
    captured = {}

    class DummyGraph:
        node_features = np.zeros((1, 4), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 4), dtype=np.float32)

    class DummyEnv:
        single_action_space = SimpleNamespace(n=4)

        def reset(self, seed=None):
            obs = np.zeros((1, 1, 2, 2), dtype=np.float32)
            info = {"graph": [DummyGraph()], "action_mask": [np.array([True, True, True, True], dtype=bool)]}
            return obs, info

        def step(self, actions):
            obs = np.zeros((1, 1, 2, 2), dtype=np.float32)
            rewards = np.array([0.0], dtype=np.float32)
            terminated = np.array([True], dtype=bool)
            truncated = np.array([False], dtype=bool)
            info = {
                "graph": [DummyGraph()],
                "action_mask": [np.array([True, True, True, True], dtype=bool)],
                "reward_hpwl_dense": [0.0],
                "reward_hpwl_terminal": [0.0],
                "reward_drc": [0.0],
                "reward_overlap": [0.0],
                "reward_routability": [0.0],
            }
            return obs, rewards, terminated, truncated, info

        def close(self):
            return None

    class DummyModel:
        def to(self, device):
            return self

        def act(self, graph_batch, spatial, action_mask_t, deterministic=False):
            return torch.tensor([0]), torch.tensor([0.0]), torch.tensor([0.0])

        def forward(self, graph_batch, spatial):
            return SimpleNamespace(value=torch.tensor([0.0], dtype=torch.float32))

        def state_dict(self):
            return {}

    class FakePPOAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def update(self, rollout, graph_list, n_epochs, batch_size):
            return {}

    monkeypatch.setattr(train_mod, "make_vec_env", lambda **kwargs: DummyEnv())
    monkeypatch.setattr(train_mod, "DualStreamActorCritic", lambda **kwargs: DummyModel())
    monkeypatch.setattr(train_mod, "PPOAgent", FakePPOAgent)
    monkeypatch.setattr(train_mod.torch, "save", lambda *args, **kwargs: None)

    cfg = Config(
        n_envs=1,
        n_steps=1,
        n_epochs=1,
        batch_size=1,
        total_timesteps=1,
        board_dir=str(tmp_path),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        clip_range=0.11,
        ent_coef=0.22,
        vf_coef=0.33,
        max_grad_norm=0.44,
    )
    train_mod.train_ppo(cfg, torch.device("cpu"))

    assert captured["lr"] == cfg.lr
    assert captured["clip_range"] == cfg.clip_range
    assert captured["ent_coef"] == cfg.ent_coef
    assert captured["vf_coef"] == cfg.vf_coef
    assert captured["max_grad_norm"] == cfg.max_grad_norm
