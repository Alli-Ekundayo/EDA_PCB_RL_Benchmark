from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_placement(board, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, board.width)
    ax.set_ylim(0, board.height)
    ax.invert_yaxis()

    ax.set_xticks(np.arange(0, board.width + 1, 1))
    ax.set_yticks(np.arange(0, board.height + 1, 1))
    ax.grid(color="gray", linestyle="-", linewidth=0.5, alpha=0.15)

    for x in range(board.width):
        for y in range(board.height):
            if board.keepout[x, y]:
                ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor="#e74c3c", alpha=0.3))

    for comp in board.components:
        if not comp.placed:
            continue
        x, y = comp.position
        fp = comp.footprint_for_rotation()
        w, h = fp.shape
        for dx in range(w):
            for dy in range(h):
                if fp[dx, dy]:
                    ax.add_patch(
                        patches.Rectangle(
                            (x + dx, y + dy),
                            1,
                            1,
                            facecolor="#3498db",
                            alpha=0.7,
                            edgecolor="#2980b9",
                        )
                    )
        ax.text(x + w / 2, y + h / 2, comp.ref, ha="center", va="center", color="black", fontweight="bold", fontsize=8)

    centers = {}
    for comp in board.components:
        if not comp.placed:
            continue
        fp = comp.footprint_for_rotation()
        w, h = fp.shape
        centers[comp.ref] = (comp.position[0] + w / 2, comp.position[1] + h / 2)

    for _net_id, refs in board.nets.items():
        placed_refs = [r for r in refs if r in centers]
        for i in range(len(placed_refs)):
            for j in range(i + 1, len(placed_refs)):
                p1 = centers[placed_refs[i]]
                p2 = centers[placed_refs[j]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#2ecc71", alpha=0.5, linestyle="--", linewidth=1.5)

    plt.title("Physical PCB Placement & Routing Skeleton", fontsize=18, fontweight="bold", pad=20)
    ax.set_aspect("equal")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
