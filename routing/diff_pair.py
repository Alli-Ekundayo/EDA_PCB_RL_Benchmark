from __future__ import annotations

from typing import Dict, List, Tuple


class DiffPairRouter:
    def __init__(self, tolerance_mm: float = 0.127) -> None:
        self.tolerance_mm = tolerance_mm

    def route(self, placed_board, diff_pairs: Dict[int, List[str]], base_routes: Dict[int, List[Tuple[int, int]]]):
        # Placeholder policy: preserve existing routes and mark diff pair compliance externally.
        routes = {}
        for net_id in diff_pairs:
            routes[net_id] = list(base_routes.get(net_id, []))
        return routes
