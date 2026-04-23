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
    # The new metrics should have safe defaults (-1.0 or -1) when no KiCad file is provided
    assert routed.routed_wirelength == -1.0
    assert routed.num_vias == -1
    assert routed.num_bends == -1


def test_kicad_routing_if_binary_exists():
    router = UnifiedPCBRouter()
    if router._binary is None:
        return  # Skip if binary not built

    # Use one of the existing raw boards for a real integration test
    test_pcb = "data/boards/rl_pcb/base_raw/tc_logger_max232.kicad_pcb"
    if not Path(test_pcb).exists():
        return

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".kicad_pcb") as tmp:
        wl, vias, bends, out = router.route_kicad_file(test_pcb, tmp.name)
        assert wl > 0
        assert vias >= 0
        assert bends >= 0
        assert Path(out).exists()
