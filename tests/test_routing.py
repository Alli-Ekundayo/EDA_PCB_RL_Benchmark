import numpy as np

from environment.board import Board, Component
from routing.router import UnifiedPCBRouter


def test_router_interface_returns_routed_board():
    comp = Component(ref="U1", class_id=0, footprint=np.ones((2, 2), dtype=bool), nets=[1], placed=True, position=(0, 0))
    board = Board(
        width=10,
        height=10,
        resolution=0.5,
        components=[comp],
        nets={1: ["U1"]},
        keepout=np.zeros((10, 10), dtype=bool),
        net_criticality={1: 1.0},
    )
    router = UnifiedPCBRouter()
    routed = router.route(board)
    assert routed.board.width == 10
