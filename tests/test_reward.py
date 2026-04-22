import numpy as np

from environment.board import Board, Component
from environment.reward import RewardWeights, normalized_hpwl, reward_components


def test_reward_components_are_finite():
    a = Component(ref="U1", class_id=0, footprint=np.ones((2, 2), dtype=bool), nets=[1], placed=False)
    b = Component(ref="R1", class_id=1, footprint=np.ones((1, 1), dtype=bool), nets=[1], placed=False)
    before = Board(
        width=10,
        height=10,
        resolution=0.5,
        components=[a, b],
        nets={1: ["U1", "R1"]},
        keepout=np.zeros((10, 10), dtype=bool),
        net_criticality={1: 0.3},
    )
    after = before.clone()
    after.components[0].placed = True
    after.components[0].position = (1, 1)
    out = reward_components(before, after, invalid_action=False, weights=RewardWeights())
    assert np.isfinite(out["reward_total"])
    assert np.isfinite(out["reward_hpwl"])
    assert out["reward_drc"] == 0.0


def test_invalid_action_adds_drc_penalty():
    comp = Component(ref="U1", class_id=0, footprint=np.ones((2, 2), dtype=bool), nets=[1], placed=False)
    board = Board(
        width=8,
        height=8,
        resolution=0.5,
        components=[comp],
        nets={1: ["U1"]},
        keepout=np.zeros((8, 8), dtype=bool),
        net_criticality={1: 0.3},
    )
    out_ok = reward_components(board, board.clone(), invalid_action=False, weights=RewardWeights())
    out_bad = reward_components(board, board.clone(), invalid_action=True, weights=RewardWeights())
    assert out_bad["reward_total"] < out_ok["reward_total"]
    assert out_bad["reward_drc"] == -1.0
    assert normalized_hpwl(board) >= 0.0
