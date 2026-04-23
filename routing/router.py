from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from environment.board import Board

from .diff_pair import DiffPairRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate the compiled Rust binary
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BIN_CANDIDATES: List[Path] = [
    _REPO_ROOT / "bin" / "pcb_router",
    _REPO_ROOT / "vendor" / "PcbRouter" / "pcb_router_rs" / "target" / "release" / "pcb_router",
]


def _find_binary() -> Optional[Path]:
    """Return the first pcb_router binary that exists and is executable."""
    for candidate in _BIN_CANDIDATES:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    # Also honour PATH
    found = shutil.which("pcb_router")
    return Path(found) if found else None


# ---------------------------------------------------------------------------
# Result dataclass (keeps full backward-compat with evaluate_model.py)
# ---------------------------------------------------------------------------
@dataclass
class RoutedBoard:
    """Holds the placed board together with routing results.

    Attributes
    ----------
    board : Board
        The placed (but unrouted) board object from the RL agent.
    general_routes : Dict[int, List[Tuple[int, int]]]
        Per-net grid-space waypoint lists for single-ended nets.
    diff_routes : Dict[int, List[Tuple[int, int]]]
        Per-net grid-space waypoint lists for differential-pair nets.
    routed_wirelength : float
        Total wirelength reported by the Rust router (mm), or -1.0 if
        routing was skipped.
    num_vias : int
        Number of vias inserted by the Rust router, or -1 if skipped.
    num_bends : int
        Number of bends reported by the Rust router, or -1 if skipped.
    output_kicad_pcb : Optional[str]
        Absolute path to the routed ``*_routed.kicad_pcb`` file written by
        the Rust binary, if a KiCad file was supplied and routing succeeded.
    """

    board: Board
    general_routes: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    diff_routes: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    routed_wirelength: float = -1.0
    num_vias: int = -1
    num_bends: int = -1
    output_kicad_pcb: Optional[str] = None


# ---------------------------------------------------------------------------
# Stdout parser
# ---------------------------------------------------------------------------
_WL_RE = re.compile(
    r"Routed WL:\s*([\d.]+),\s*#\s*vias:\s*(\d+),\s*#\s*bends:\s*(\d+)",
    re.IGNORECASE,
)


def _parse_router_stdout(stdout: str) -> Tuple[float, int, int]:
    """Extract (wirelength, vias, bends) from the Rust binary's stdout."""
    m = _WL_RE.search(stdout)
    if m:
        return float(m.group(1)), int(m.group(2)), int(m.group(3))
    return -1.0, -1, -1


# ---------------------------------------------------------------------------
# Main router class
# ---------------------------------------------------------------------------
class UnifiedPCBRouter:
    """PCB router that drives the compiled Rust ``pcb_router`` binary.

    The router supports two modes:

    1. **KiCad mode** – supply a ``kicad_pcb_path`` to ``route()``.
       The binary reads and writes the file directly; the method returns
       a :class:`RoutedBoard` with real wirelength/via metrics populated.

    2. **Board-object mode** – supply only a :class:`~environment.board.Board`.
       The router attempts to find a matching ``.kicad_pcb`` file via
       ``kicad_pcb_path`` on the object (if present as a custom attribute),
       otherwise it runs in *metric-only* mode: it classifies nets, runs
       the diff-pair heuristic, and returns stub route lists.  Real routing
       statistics will be -1.

    Parameters
    ----------
    grid_scale : float
        Passed as ``grid_scale`` to the Rust binary (controls grid
        resolution; default ``1.0``).
    num_iterations : int
        Rip-up / re-route iterations (default ``3``).
    enlarge_boundary : int
        Grid boundary padding in grid units (default ``10``).
    layer_change_weight : float
        Cost multiplier for layer-change (via) insertion (default ``3.0``).
    track_obstacle_weight : float
        Cost multiplier for crossing existing traces (default ``1.0``).
    timeout : int
        Subprocess timeout in seconds (default ``300``).
    """

    def __init__(
        self,
        grid_scale: float = 1.0,
        num_iterations: int = 3,
        enlarge_boundary: int = 10,
        layer_change_weight: float = 3.0,
        track_obstacle_weight: float = 1.0,
        timeout: int = 300,
    ) -> None:
        self.grid_scale = grid_scale
        self.num_iterations = num_iterations
        self.enlarge_boundary = enlarge_boundary
        self.layer_change_weight = layer_change_weight
        self.track_obstacle_weight = track_obstacle_weight
        self.timeout = timeout

        self._binary: Optional[Path] = _find_binary()
        if self._binary is None:
            logger.warning(
                "pcb_router binary not found. "
                "Build it with:\n"
                "  cd vendor/PcbRouter/pcb_router_rs && "
                "~/.cargo/bin/cargo build --release\n"
                "  cp target/release/pcb_router ../../bin/\n"
                "Falling back to stub routing."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def route(
        self,
        placed_board: Board,
        kicad_pcb_path: Optional[str] = None,
    ) -> RoutedBoard:
        """Route the board, optionally driving the Rust binary.

        Parameters
        ----------
        placed_board : Board
            The placed board from the RL agent (or any other placer).
        kicad_pcb_path : str, optional
            Absolute (or relative) path to a ``.kicad_pcb`` file that
            corresponds to ``placed_board``.  When supplied *and* the
            Rust binary is available, the binary is invoked and a real
            routed file is produced alongside the input.

        Returns
        -------
        RoutedBoard
        """
        # Classify nets into general / diff-pair buckets
        general_nets = self._classify_general(placed_board.nets)
        diff_nets = self._classify_diff_pairs(placed_board.nets)

        # Build stub route lists (grid waypoints stay empty; real paths are
        # in the .kicad_pcb output file when available)
        general_routes = {nid: [] for nid in general_nets}
        diff_routes = DiffPairRouter().route(placed_board, diff_nets, general_routes)

        result = RoutedBoard(
            board=placed_board,
            general_routes=general_routes,
            diff_routes=diff_routes,
        )

        # ------------------------------------------------------------------
        # Attempt to invoke the Rust binary
        # ------------------------------------------------------------------
        if kicad_pcb_path is None:
            # Check whether the Board carries the path as a custom attribute
            kicad_pcb_path = getattr(placed_board, "kicad_pcb_path", None)

        if kicad_pcb_path is not None and self._binary is not None:
            result = self._run_rust_router(placed_board, kicad_pcb_path, result)
        elif kicad_pcb_path is not None and self._binary is None:
            logger.warning(
                "kicad_pcb_path supplied but binary missing – skipping physical routing."
            )
        else:
            logger.info(
                "No kicad_pcb_path supplied; returning stub routes only."
            )

        return result

    def route_kicad_file(
        self,
        kicad_pcb_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[float, int, int, str]:
        """Directly route a ``.kicad_pcb`` file and return metrics.

        This is a convenience method for scripts that already have a KiCad
        file and don't need the ``Board`` object interface.

        Parameters
        ----------
        kicad_pcb_path : str
            Path to the input ``.kicad_pcb`` file.
        output_path : str, optional
            Destination for the routed file.  Defaults to
            ``<stem>_routed.kicad_pcb`` next to the input.

        Returns
        -------
        (wirelength, num_vias, num_bends, output_path) : Tuple
        """
        if self._binary is None:
            raise RuntimeError(
                "Rust pcb_router binary not found. "
                "Build it first (see UnifiedPCBRouter docstring)."
            )
        wl, vias, bends, out = self._invoke_binary(kicad_pcb_path, output_path)
        return wl, vias, bends, out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _classify_general(self, nets: Dict) -> Dict:
        return {nid: refs for nid, refs in nets.items() if nid % 2 == 0}

    def _classify_diff_pairs(self, nets: Dict) -> Dict:
        return {nid: refs for nid, refs in nets.items() if nid % 2 == 1}

    def _run_rust_router(
        self,
        placed_board: Board,
        kicad_pcb_path: str,
        result: RoutedBoard,
    ) -> RoutedBoard:
        """Invoke the Rust binary and populate routing metrics in *result*."""
        pcb_path = Path(kicad_pcb_path)
        if not pcb_path.exists():
            logger.error("kicad_pcb_path does not exist: %s", kicad_pcb_path)
            return result

        # Determine output path (sibling of input, same directory)
        stem = pcb_path.stem
        output_path = str(pcb_path.parent / f"{stem}_routed{pcb_path.suffix}")

        try:
            wl, vias, bends, out = self._invoke_binary(
                str(pcb_path), output_path
            )
            result.routed_wirelength = wl
            result.num_vias = vias
            result.num_bends = bends
            result.output_kicad_pcb = out if Path(out).exists() else None

            logger.info(
                "Rust router finished: WL=%.4f mm, vias=%d, bends=%d → %s",
                wl, vias, bends, result.output_kicad_pcb or "(no output file)",
            )
        except subprocess.TimeoutExpired:
            logger.error("Rust pcb_router timed out after %ds.", self.timeout)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Rust pcb_router exited with code %d:\n%s",
                exc.returncode, exc.stderr or "",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error invoking pcb_router: %s", exc)

        return result

    def _invoke_binary(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[float, int, int, str]:
        """Run the binary and return (wirelength, vias, bends, output_path)."""
        cmd: List[str] = [
            str(self._binary),
            input_path,
        ]
        if output_path:
            cmd.append(output_path)
        cmd += [
            str(self.grid_scale),
            str(self.num_iterations),
            str(self.enlarge_boundary),
            str(self.layer_change_weight),
            str(self.track_obstacle_weight),
        ]

        logger.debug("Running: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=True,
        )

        stdout = proc.stdout
        logger.debug("pcb_router stdout:\n%s", stdout)
        if proc.stderr:
            logger.debug("pcb_router stderr:\n%s", proc.stderr)

        wl, vias, bends = _parse_router_stdout(stdout)

        # Resolve the actual output file the binary wrote
        if output_path is None:
            p = Path(input_path)
            output_path = str(p.parent / f"{p.stem}_routed{p.suffix}")

        return wl, vias, bends, output_path
