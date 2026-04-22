from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Component:
    ref: str
    class_id: int
    footprint: np.ndarray
    nets: List[int]
    placed: bool = False
    position: Optional[Tuple[int, int]] = None
    rotation: int = 0
    target_position: Optional[Tuple[int, int]] = None
    movable: bool = True

    def footprint_for_rotation(self) -> np.ndarray:
        k = (self.rotation // 90) % 4
        return np.rot90(self.footprint, k=k)


@dataclass
class Board:
    width: int
    height: int
    resolution: float
    components: List[Component]
    nets: Dict[int, List[str]]
    keepout: np.ndarray
    net_criticality: Dict[int, float]
    min_clearance: int = 1

    def clone(self) -> "Board":
        components = [
            Component(
                ref=c.ref,
                class_id=c.class_id,
                footprint=c.footprint.copy(),
                nets=list(c.nets),
                placed=c.placed,
                position=None if c.position is None else (c.position[0], c.position[1]),
                rotation=c.rotation,
                target_position=None if c.target_position is None else (c.target_position[0], c.target_position[1]),
                movable=c.movable,
            )
            for c in self.components
        ]
        return Board(
            width=self.width,
            height=self.height,
            resolution=self.resolution,
            components=components,
            nets={k: list(v) for k, v in self.nets.items()},
            keepout=self.keepout.copy(),
            net_criticality=dict(self.net_criticality),
            min_clearance=self.min_clearance,
        )

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["keepout"] = self.keepout.astype(int).tolist()
        for comp in payload["components"]:
            comp["footprint"] = np.asarray(comp["footprint"]).astype(int).tolist()
        return payload


def occupied_grid(board: Board, exclude_ref: Optional[str] = None) -> np.ndarray:
    grid = np.zeros((board.width, board.height), dtype=np.int32)
    for comp in board.components:
        if not comp.placed or comp.position is None:
            continue
        if exclude_ref is not None and comp.ref == exclude_ref:
            continue
        fp = comp.footprint_for_rotation().astype(np.int32)
        x0, y0 = comp.position
        w, h = fp.shape
        grid[x0 : x0 + w, y0 : y0 + h] += fp
    return grid


def component_center(component: Component) -> Tuple[float, float]:
    if component.position is None:
        return (0.0, 0.0)
    fp = component.footprint_for_rotation()
    w, h = fp.shape
    return (component.position[0] + w / 2.0, component.position[1] + h / 2.0)
