import numpy as np

from environment.board import Board, Component
from environment.drc_mask import compute_action_mask


def test_drc_blocks_overlap():
    placed = Component(ref="U1", class_id=0, footprint=np.ones((2, 2), dtype=bool), nets=[1], placed=True, position=(3, 3))
    target = Component(ref="R1", class_id=1, footprint=np.ones((2, 2), dtype=bool), nets=[1], placed=False)
    board = Board(
        width=10,
        height=10,
        resolution=0.5,
        components=[placed, target],
        nets={1: ["U1", "R1"]},
        keepout=np.zeros((10, 10), dtype=bool),
        net_criticality={1: 0.3},
    )
    mask = compute_action_mask(board, target)
    assert mask.reshape(4, 10, 10)[0, 3, 3] == False
