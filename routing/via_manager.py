from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Via:
    x: int
    y: int
    from_layer: int
    to_layer: int


class ViaManager:
    def __init__(self) -> None:
        self.vias: List[Via] = []

    def add_via(self, x: int, y: int, from_layer: int, to_layer: int) -> Via:
        via = Via(x=x, y=y, from_layer=from_layer, to_layer=to_layer)
        self.vias.append(via)
        return via

    def all_positions(self) -> List[Tuple[int, int]]:
        return [(v.x, v.y) for v in self.vias]
