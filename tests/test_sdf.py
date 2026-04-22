import numpy as np

from environment.board import Board, Component
from environment.sdf_generator import NUM_COMPONENT_CLASSES, compute_sdf


def test_sdf_shape_and_values():
    comp = Component(ref="U1", class_id=0, footprint=np.ones((2, 2), dtype=bool), nets=[1], placed=True, position=(1, 1))
    board = Board(
        width=10,
        height=10,
        resolution=0.5,
        components=[comp],
        nets={1: ["U1"]},
        keepout=np.zeros((10, 10), dtype=bool),
        net_criticality={1: 0.3},
    )
    obs = compute_sdf(board)
    assert obs.shape == (NUM_COMPONENT_CLASSES + 3, 10, 10)
    assert np.all(obs >= 0.0)
    assert obs[0, 1, 1] == 0.0
