from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from environment.board import Board

from .diff_pair import DiffPairRouter


@dataclass
class RoutedBoard:
    board: Board
    general_routes: Dict[int, List[Tuple[int, int]]]
    diff_routes: Dict[int, List[Tuple[int, int]]]


class UnifiedPCBRouter:
    def _classify_general(self, nets):
        return {nid: refs for nid, refs in nets.items() if nid % 2 == 0}

    def _classify_diff_pairs(self, nets):
        return {nid: refs for nid, refs in nets.items() if nid % 2 == 1}

    def _route_general(self, placed_board: Board, general_nets):
        routes = {}
        for net_id, _ in general_nets.items():
            routes[net_id] = []
        return routes

    def route(self, placed_board: Board) -> RoutedBoard:
        general_nets = self._classify_general(placed_board.nets)
        diff_pairs = self._classify_diff_pairs(placed_board.nets)
        general_routes = self._route_general(placed_board, general_nets)
        diff_routes = DiffPairRouter().route(placed_board, diff_pairs, general_routes)
        return RoutedBoard(placed_board, general_routes, diff_routes)
