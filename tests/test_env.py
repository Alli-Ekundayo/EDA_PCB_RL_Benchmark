from pathlib import Path

from environment.pcb_env import PCBEnv


def test_env_step_reset_contract():
    board_file = Path("data/boards/sample_board.json")
    assert board_file.exists(), "Run scripts/preprocess_boards.py first"
    env = PCBEnv(board_path=str(board_file), width=20, height=20, component_rotations=(0, 90, 180, 270))
    obs, info = env.reset()
    assert obs.shape[0] >= 4
    assert info["action_mask"].dtype == bool
    assert info["action_mask"].shape[0] == env.action_space.n
    action = int(info["action_mask"].argmax())
    next_obs, reward, terminated, truncated, next_info = env.step(action)
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "action_mask" in next_info
    assert "hpwl" in next_info
