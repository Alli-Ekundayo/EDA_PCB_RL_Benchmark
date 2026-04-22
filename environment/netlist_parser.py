from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .board import Board, Component


DEFAULT_CLASS_MAP = {
    "U": 0,
    "IC": 0,
    "C": 1,
    "R": 1,
    "L": 1,
    "J": 2,
    "P": 2,
    "TP": 3,
}


@dataclass
class ParsedGraph:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray


def _default_footprint() -> np.ndarray:
    return np.ones((2, 2), dtype=bool)


def infer_criticality(net_name: str) -> float:
    lname = net_name.lower()
    if "diff" in lname or lname.endswith("_p") or lname.endswith("_n"):
        return 1.0
    if "vcc" in lname or "gnd" in lname or "power" in lname:
        return 0.7
    return 0.3


def _class_from_ref(ref: str) -> int:
    upper = ref.upper()
    for prefix, cid in DEFAULT_CLASS_MAP.items():
        if upper.startswith(prefix):
            return cid
    return 1


def _as_grid_size(mm_w: float, mm_h: float, board_w_mm: float, board_h_mm: float, width: int, height: int) -> Tuple[int, int]:
    # Convert mm footprint to grid cells, keeping a minimum 1x1 occupancy.
    gx = max(1, int(round((mm_w / max(board_w_mm, 1e-6)) * width)))
    gy = max(1, int(round((mm_h / max(board_h_mm, 1e-6)) * height)))
    return gx, gy


def _as_grid_pos(mm_x: float, mm_y: float, bb_min_x: float, bb_min_y: float, board_w_mm: float, board_h_mm: float, width: int, height: int) -> Tuple[int, int]:
    rx = (mm_x - bb_min_x) / max(board_w_mm, 1e-6)
    ry = (mm_y - bb_min_y) / max(board_h_mm, 1e-6)
    gx = int(np.clip(round(rx * (width - 1)), 0, width - 1))
    gy = int(np.clip(round(ry * (height - 1)), 0, height - 1))
    return gx, gy


def _parse_rlpcb_block(block: str, width: int, height: int) -> Tuple[Board, nx.Graph, ParsedGraph]:
    lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]

    bb_min_x = 0.0
    bb_min_y = 0.0
    bb_max_x = float(width)
    bb_max_y = float(height)

    node_rows: List[List[str]] = []
    edge_rows: List[List[str]] = []
    in_nodes = False
    in_edges = False

    for line in lines:
        s = line.strip()
        if s == "nodes begin":
            in_nodes = True
            continue
        if s == "nodes end":
            in_nodes = False
            continue
        if s == "edges begin":
            in_edges = True
            continue
        if s == "edges end":
            in_edges = False
            continue
        if s.startswith("bb_min_x,"):
            bb_min_x = float(s.split(",", 1)[1])
        elif s.startswith("bb_min_y,"):
            bb_min_y = float(s.split(",", 1)[1])
        elif s.startswith("bb_max_x,"):
            bb_max_x = float(s.split(",", 1)[1])
        elif s.startswith("bb_max_y,"):
            bb_max_y = float(s.split(",", 1)[1])
        elif in_nodes and s and s[0].isdigit():
            node_rows.append([x.strip().strip('"') for x in s.split(",")])
        elif in_edges and s and s[0].isdigit():
            edge_rows.append([x.strip().strip('"') for x in s.split(",")])

    board_w_mm = max(bb_max_x - bb_min_x, 1.0)
    board_h_mm = max(bb_max_y - bb_min_y, 1.0)

    idx_to_ref: Dict[int, str] = {}
    comps: List[Component] = []
    for row in node_rows:
        # Format: idx,ref,w,h,x,y,rotation,...
        if len(row) < 7:
            continue
        idx = int(float(row[0]))
        ref = row[1]
        mm_w = float(row[2])
        mm_h = float(row[3])
        mm_x = float(row[4])
        mm_y = float(row[5])
        rotation = int(round(float(row[6]))) % 360
        fw, fh = _as_grid_size(mm_w, mm_h, board_w_mm, board_h_mm, width, height)
        gx, gy = _as_grid_pos(mm_x, mm_y, bb_min_x, bb_min_y, board_w_mm, board_h_mm, width, height)

        comps.append(
            Component(
                ref=ref,
                class_id=_class_from_ref(ref),
                footprint=np.ones((fw, fh), dtype=bool),
                nets=[],
                placed=False,
                position=None,
                rotation=rotation,
                target_position=(gx, gy),
                movable=True,
            )
        )
        idx_to_ref[idx] = ref

    ref_to_comp: Dict[str, Component] = {c.ref: c for c in comps}
    net_name_to_id: Dict[str, int] = {}
    nets: Dict[int, List[str]] = {}

    for row in edge_rows:
        # Row has many fields, but source/dest component indices are first and ninth tokens.
        if len(row) < 18:
            continue
        src_idx = int(float(row[0]))
        dst_idx = int(float(row[8]))
        net_name = row[-2] if len(row) >= 2 else f"net_{src_idx}_{dst_idx}"
        if net_name not in net_name_to_id:
            net_name_to_id[net_name] = len(net_name_to_id) + 1
        net_id = net_name_to_id[net_name]
        src_ref = idx_to_ref.get(src_idx)
        dst_ref = idx_to_ref.get(dst_idx)
        if src_ref is None or dst_ref is None:
            continue
        if net_id not in nets:
            nets[net_id] = []
        if src_ref not in nets[net_id]:
            nets[net_id].append(src_ref)
        if dst_ref not in nets[net_id]:
            nets[net_id].append(dst_ref)

    for net_id, members in nets.items():
        for ref in members:
            comp = ref_to_comp.get(ref)
            if comp is not None and net_id not in comp.nets:
                comp.nets.append(net_id)

    net_criticality = {nid: infer_criticality(name) for name, nid in net_name_to_id.items()}

    board = Board(
        width=width,
        height=height,
        resolution=max(board_w_mm / max(width, 1), board_h_mm / max(height, 1)),
        components=comps,
        nets=nets,
        keepout=np.zeros((width, height), dtype=bool),
        net_criticality=net_criticality,
        min_clearance=1,
    )
    graph = build_netlist_graph(board)
    features = graph_to_features(board, graph)
    return board, graph, features


def _parse_json_board(path: Path, width: int, height: int) -> Tuple[Board, nx.Graph, ParsedGraph]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    comps = []
    for item in payload.get("components", []):
        ref = item["ref"]
        cls = item.get("class_id")
        if cls is None:
            cls = _class_from_ref(ref)
        comps.append(Component(ref=ref, class_id=cls, footprint=_default_footprint(), nets=item["nets"]))
    nets = {int(k): v for k, v in payload["nets"].items()}

    net_criticality = {nid: infer_criticality(f"net_{nid}") for nid in nets.keys()}
    board = Board(
        width=width,
        height=height,
        resolution=0.5,
        components=comps,
        nets=nets,
        keepout=np.zeros((width, height), dtype=bool),
        net_criticality=net_criticality,
    )
    graph = build_netlist_graph(board)
    features = graph_to_features(board, graph)
    return board, graph, features


def _parse_net_fallback(path: Path, width: int, height: int) -> Tuple[Board, nx.Graph, ParsedGraph]:
    comps_by_ref: Dict[str, Component] = {}
    nets: Dict[int, List[str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("NET "):
                continue
            left, right = line.split(":", maxsplit=1)
            _, net_id_str, _net_name = left.split(maxsplit=2)
            net_id = int(net_id_str)
            refs = [r.strip() for r in right.split(",") if r.strip()]
            nets[net_id] = refs
            for ref in refs:
                if ref not in comps_by_ref:
                    comps_by_ref[ref] = Component(
                        ref=ref,
                        class_id=_class_from_ref(ref),
                        footprint=_default_footprint(),
                        nets=[],
                    )
                comps_by_ref[ref].nets.append(net_id)
    comps = list(comps_by_ref.values())
    net_criticality = {nid: infer_criticality(f"net_{nid}") for nid in nets.keys()}
    board = Board(
        width=width,
        height=height,
        resolution=0.5,
        components=comps,
        nets=nets,
        keepout=np.zeros((width, height), dtype=bool),
        net_criticality=net_criticality,
    )
    graph = build_netlist_graph(board)
    features = graph_to_features(board, graph)
    return board, graph, features


def parse_board_file(path: str, width: int = 50, height: int = 50, pcb_idx: Optional[int] = None) -> Tuple[Board, nx.Graph, ParsedGraph]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".json":
        return _parse_json_board(p, width, height)

    if p.suffix.lower() == ".pcb":
        text = p.read_text(encoding="utf-8", errors="ignore")
        blocks: List[str] = []
        start = "pcb begin"
        end = "pcb end"
        sidx = 0
        while True:
            i = text.find(start, sidx)
            if i == -1:
                break
            j = text.find(end, i)
            if j == -1:
                break
            blocks.append(text[i : j + len(end)])
            sidx = j + len(end)
        if not blocks:
            raise ValueError(f"No pcb blocks found in {path}")
        chosen = 0 if pcb_idx is None else int(np.clip(pcb_idx, 0, len(blocks) - 1))
        return _parse_rlpcb_block(blocks[chosen], width, height)

    return _parse_net_fallback(p, width, height)


def build_netlist_graph(board: Board) -> nx.Graph:
    g = nx.Graph()
    refs = {c.ref: c for c in board.components}
    for c in board.components:
        g.add_node(c.ref, class_id=c.class_id, pin_count=len(c.nets), placed=float(c.placed))

    for net_id, members in board.nets.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a not in refs or b not in refs:
                    continue
                g.add_edge(
                    a,
                    b,
                    net_id=net_id,
                    criticality=board.net_criticality.get(net_id, 0.3),
                    connection_count=len(members),
                    is_differential_pair=float(board.net_criticality.get(net_id, 0.3) >= 0.95),
                )
    return g


def graph_to_features(board: Board, graph: nx.Graph) -> ParsedGraph:
    refs = [c.ref for c in board.components]
    idx = {ref: i for i, ref in enumerate(refs)}
    n = len(refs)
    class_count = max((c.class_id for c in board.components), default=0) + 1
    x = np.zeros((n, class_count + 4), dtype=np.float32)

    for i, c in enumerate(board.components):
        x[i, c.class_id] = 1.0
        x[i, class_count] = float(len(c.nets)) / 16.0
        x[i, class_count + 1] = 1.0 if c.placed else 0.0
        x[i, class_count + 2] = float(c.rotation) / 360.0
        x[i, class_count + 3] = (sum(c.nets) / max(len(c.nets), 1)) / 128.0

    edges = []
    edge_attr = []
    for a, b, data in graph.edges(data=True):
        ia, ib = idx[a], idx[b]
        feat = [
            float(data.get("criticality", 0.3)),
            float(data.get("net_id", 0)) / 128.0,
            float(data.get("connection_count", 2)) / 16.0,
            float(data.get("is_differential_pair", 0.0)),
        ]
        edges.append([ia, ib])
        edges.append([ib, ia])
        edge_attr.append(feat)
        edge_attr.append(feat)

    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_attr = np.asarray(edge_attr, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 4), dtype=np.float32)

    return ParsedGraph(node_features=x, edge_index=edge_index, edge_attr=edge_attr)
