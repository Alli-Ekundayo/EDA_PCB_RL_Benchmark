#!/usr/bin/env python3
import os
import glob
import argparse
import platform
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import Config
from environment.pcb_env import PCBEnv
from evaluation.eval import load_model, sync_config_from_checkpoint, _select_deterministic_action
from evaluation.plotting import plot_placement
from training.graph_utils import graph_to_data
from torch_geometric.data import Batch
from routing.router import UnifiedPCBRouter
from environment.reward import pattern_routability_proxy
from evaluation.metrics import summarize_metrics

def _is_fully_routed(metrics: dict) -> bool:
    total_routes = float(metrics.get("routed_nets_established", 0))
    total_nets = float(metrics.get("total_nets", 1))
    return total_routes >= total_nets

def find_best_checkpoint(work_dir):
    """Find the most recently modified .pt checkpoint, preferring those marked 'final'."""
    checkpoints = list(Path(work_dir).rglob("*.pt"))
    if not checkpoints:
        return None
    finals = [c for c in checkpoints if "final" in c.name.lower()]
    if finals:
        return str(sorted(finals, key=os.path.getmtime)[-1])
    return str(sorted(checkpoints, key=os.path.getmtime)[-1])

def build_pdf_report(metrics, plot_path, output_pdf, title="Reinforcement Learning - Evaluation Report"):
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    header_style = ParagraphStyle('Header', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor("#1A3A3A"), spaceAfter=20)
    story.append(Paragraph(title, header_style))
    story.append(Paragraph("Focus: Physical Placement Quality and Routability Validation", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    # Environment Info
    env_info = [
        ["Hardware & Software", ""],
        ["Platform", platform.platform()],
        ["Python", platform.python_version()],
        ["PyTorch", torch.__version__],
    ]
    story.append(Paragraph("Validation Environment", styles['Heading2']))
    t_env = Table(env_info, colWidths=[2 * inch, 4 * inch])
    t_env.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#D1E8E2")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(t_env)
    story.append(Spacer(1, 0.4 * inch))

    # Plot Image
    if os.path.exists(plot_path):
        story.append(Paragraph("Placement Visualization", styles['Heading2']))
        story.append(Image(plot_path, width=5.5*inch, height=5.5*inch))
        story.append(Spacer(1, 0.2*inch))
        
    story.append(PageBreak())

    # Metrics Table
    story.append(Paragraph("Physical Routing & Placement Metrics", styles['Heading2']))
    story.append(Paragraph("These metrics represent the 'ground truth' for layout quality.", styles['Italic']))
    story.append(Spacer(1, 0.1 * inch))
    
    table_data = [["Metric Name", "Value", "Description"]]
    
    descriptions = {
        "hpwl": "Half-Perimeter Wire Length (Estimated)",
        "routed_wirelength_mm": "Actual Routed Wire Length from Rust Router",
        "invalid_actions": "DRC Violations (Overlaps/Keepouts hit)",
        "routability_proxy": "Heuristic estimate of routing success",
        "num_vias": "Number of layer transitions (vias)",
        "total_nets": "Total number of connections in netlist",
        "components_placed": "Count of successfully placed components",
        "routed_nets_established": "Nets with confirmed routing results",
    }

    for k, v in metrics.items():
        formatted_name = k.replace("_", " ").title()
        formatted_val = f"{v:.4f}" if isinstance(v, (float, np.float64, np.float32)) else str(v)
        desc = descriptions.get(k.lower(), "Physical metric")
        table_data.append([formatted_name, formatted_val, desc])
        
    t = Table(table_data, colWidths=[1.8 * inch, 1.2 * inch, 3.0 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#D1E8E2")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    
    # Routability Summary
    is_fully_routed = _is_fully_routed(metrics)
    status_color = colors.green if is_fully_routed else colors.red
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph(f"<b>Final Routing Status: {'SUCCESS' if is_fully_routed else 'PARTIAL'}</b>", 
                          ParagraphStyle('Status', parent=styles['Normal'], textColor=status_color, fontSize=12)))

    doc.build(story)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PCB agent and run the physical router.")
    parser.add_argument("--work_dir", type=str, default="runs/experiments", help="Directory containing model checkpoints")
    parser.add_argument("--board_file", type=str, default="data/boards/rl_pcb/base/evaluation.pcb", help="Evaluation board file")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Configuration file used for training")
    parser.add_argument("--out_dir", type=str, default="runs/evaluation", help="Output directory for reports and images")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print("--- Starting Physical PCB Evaluation (Evaluation Report Focus) ---")
    
    config = Config.from_yaml(args.config)
    checkpoint_path = find_best_checkpoint(args.work_dir)
    if not checkpoint_path:
        print(f"Error: No model checkpoints found in {args.work_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Sync configuration from the checkpoint first
    # This is critical because flags like use_ratsnest affect environment observation shape
    print(f"Syncing configuration from: {checkpoint_path}")
    sync_config_from_checkpoint(checkpoint_path, config, device)

    # 2. Now initialize environment with the (synced) config
    rotations = tuple(90 * i for i in range(config.component_rotations))
    print(f"Initializing environment with {args.board_file}...")
    try:
        env = PCBEnv(
            board_path=args.board_file,
            width=config.board_width,
            height=config.board_height,
            component_rotations=rotations,
            use_ratsnest=config.use_ratsnest,
            use_criticality=config.use_criticality,
        )
        obs, info = env.reset()
        graph = graph_to_data(info["graph"])
    except FileNotFoundError:
        print(f"Error: Could not find board file at {args.board_file}")
        return
    
    action_dim = env.action_space.n
    
    # 3. Load model with correct environment dimensions
    print("Loading model weights...")
    model = load_model(
        checkpoint_path=checkpoint_path, 
        config=config, 
        obs_channels=obs.shape[0], 
        node_feat_dim=graph.x.shape[1], 
        edge_feat_dim=graph.edge_attr.shape[1] if graph.edge_attr.numel() > 0 else 4, 
        action_dim=action_dim, 
        device=device
    )
    
    terminated = truncated = False
    invalid_actions = 0
    total_actions = 0
    
    print("Agent is making sequential placement decisions...")
    while not (terminated or truncated):
        spatial = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
        graph_batch = Batch.from_data_list([graph]).to(device)
        action_mask = info["action_mask"]
        action_mask_t = torch.as_tensor(action_mask[None, ...], dtype=torch.bool, device=device)
        
        with torch.no_grad():
            action = _select_deterministic_action(
                model=model,
                algo=config.algo.lower(),
                graph_batch=graph_batch,
                spatial=spatial,
                action_mask_t=action_mask_t,
                width=config.board_width,
                height=config.board_height,
                n_rotations=len(rotations),
            )
        obs, _, terminated, truncated, info = env.step(action)
        graph = graph_to_data(info["graph"])
        
        if not info.get("valid_action", True):
            invalid_actions += 1
        total_actions += 1

    board = env.board
    print(f"Placement complete. Total Steps: {total_actions}, DRC Violations: {invalid_actions}")
    
    print("Invoking physical router (Rust pcb_router via UnifiedPCBRouter)...")
    router = UnifiedPCBRouter()
    raw_kicad_path: Optional[str] = None
    raw_dir = Path(args.board_file).parent.parent / "base_raw"
    if raw_dir.is_dir():
        candidates = sorted(raw_dir.glob("*.kicad_pcb"))
        if candidates:
            raw_kicad_path = str(candidates[0])
            print(f"  Using KiCad source: {raw_kicad_path}")
    routed_board = router.route(board, kicad_pcb_path=raw_kicad_path)

    metrics = summarize_metrics(board, total_actions, invalid_actions)
    metrics["routability_proxy"] = pattern_routability_proxy(board)
    metrics["total_nets"] = len(board.nets)
    metrics["components_placed"] = len([c for c in board.components if c.placed])
    metrics["total_components"] = len(board.components)
    metrics["routed_nets_established"] = routed_board.num_routed_nets()
    
    if routed_board.routed_wirelength >= 0:
        metrics["routed_wirelength_mm"] = routed_board.routed_wirelength
        metrics["num_vias"] = float(routed_board.num_vias)
        metrics["num_bends"] = float(routed_board.num_bends)
    
    png_path = os.path.join(args.out_dir, "placement_visualization.png")
    plot_placement(board, png_path)
    
    pdf_output = os.path.join(args.out_dir, "evaluation_report.pdf")
    build_pdf_report(metrics, png_path, pdf_output)
    
    print(f"\nEvaluation Report generated at: {pdf_output}")
    print("\n--- Final Physical Metrics ---")
    for k, v in metrics.items():
        print(f"  {k.replace('_', ' ').title()}: {v:.4f}" if isinstance(v, (float, np.float64)) else f"  {k.replace('_', ' ').title()}: {v}")

if __name__ == "__main__":
    main()
