import os
import json
import glob
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import torch

def parse_log(log_file):
    data = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if 'train/' in line:
                    parts = line.strip().split(' ')
                    entry = {}
                    for p in parts:
                        if '=' in p:
                            parts_p = p.split('=')
                            if len(parts_p) == 2:
                                k, v = parts_p
                                try:
                                    entry[k.replace('train/', '')] = float(v)
                                except:
                                    continue
                    if 'global_step' in entry and 'mean_reward' in entry:
                        data.append(entry)
    return pd.DataFrame(data)

def extract_hyperparams(work_dir):
    """Try to extract hyperparameters from .pt files or config.yaml."""
    # Priority 1: .pt checkpoints
    pt_files = glob.glob(os.path.join(work_dir, "**", "*.pt"), recursive=True)
    for pt in pt_files:
        try:
            checkpoint = torch.load(pt, map_location='cpu')
            if 'config' in checkpoint:
                cfg = checkpoint['config']
                if hasattr(cfg, 'to_dict'): return cfg.to_dict()
                return cfg
        except:
            continue
    
    # Priority 2: config.yaml or base.yaml
    search_paths = [
        os.path.join(work_dir, "**", "config.yaml"),
        os.path.join(work_dir, "config.yaml"),
        "configs/base.yaml"
    ]
    for path in search_paths:
        files = glob.glob(path, recursive=True)
        if files:
            try:
                import yaml
                with open(files[0], 'r') as f:
                    return yaml.safe_load(f)
            except:
                continue
    return None

def _detect_device():
    """Detect the compute device string for the report title."""
    if torch.cuda.is_available():
        return f"cuda_{torch.cuda.current_device()}"
    return "cpu"

def _algo_from_path(log_file: str) -> str:
    """Infer algorithm from full log path (covers seed-subdir names like ppo_seed_42)."""
    lower = log_file.lower().replace(os.sep, '/')
    if 'td3' in lower: return 'td3'
    if 'sac' in lower: return 'sac'
    return 'ppo'

def generate_report(work_dir, output_pdf, title=None):
    # 1. Sweep directory for log files
    all_logs = glob.glob(os.path.join(work_dir, "**", "training.log"), recursive=True)
    if not all_logs:
        print(f"No training.log files found in {work_dir}")
        return

    # If per-seed subdirectory logs exist, drop the root-level log to avoid
    # double-counting (the root log is a leftover from earlier manual runs).
    subdir_logs = [f for f in all_logs if os.path.dirname(os.path.abspath(f)) != os.path.abspath(work_dir)]
    log_files = sorted(subdir_logs if subdir_logs else all_logs)

    # Collect per-algorithm, per-run data
    # Structure: { algo: [ (run_dir_name, DataFrame), ... ] }
    algo_runs = {}

    for log_file in log_files:
        parent_dir = os.path.basename(os.path.dirname(log_file))
        algo = _algo_from_path(log_file)

        df = parse_log(log_file)
        if df.empty:
            continue
        # Deduplicate by global_step: keep last logged value per step
        # (heartbeat entries may repeat the previous episodic return at new steps)
        df = df.drop_duplicates(subset=['global_step'], keep='last').reset_index(drop=True)
        if not df.empty:
            if algo not in algo_runs: algo_runs[algo] = []
            algo_runs[algo].append((parent_dir if parent_dir else "root", df))

    if not algo_runs:
        print("No valid training data found in logs.")
        return

    # Auto-generate title if not provided
    device_str = _detect_device()
    if title is None:
        algos_str = "_".join(sorted(algo_runs.keys())).upper()
        title = f"{algos_str}_{device_str}:{algos_str}"

    # 2. Process Data and Create Plots
    os.makedirs("tmp_plots", exist_ok=True)
    plot_path = "tmp_plots/learning_curves.png"
    
    plt.figure(figsize=(10, 6))
    colors_map = {'ppo': '#3498db', 'td3': '#e74c3c', 'sac': '#2ecc71'}

    # Per-algo run statistics for the table section
    # { algo: [ { 'run_name': str, 'mean': float, 'std': float }, ... ] }
    algo_run_stats = {}

    for algo, runs in algo_runs.items():
        color = colors_map.get(algo.lower(), '#9b59b6')
        dfs = [r[1] for r in runs]
        all_steps = pd.concat([df['global_step'] for df in dfs]).sort_values().unique()
        
        if len(all_steps) == 0: continue

        algo_run_stats[algo] = []
        
        interp_rewards = []
        for run_name, df in runs:
            df = df.drop_duplicates(subset=['global_step'])
            rewards = df['mean_reward'].values
            run_mean = float(np.mean(rewards))
            run_std = float(np.std(rewards))
            algo_run_stats[algo].append({
                'run_name': run_name,
                'mean': run_mean,
                'std': run_std,
            })

            if len(df) == 1:
                # Handle single-point runs
                plt.scatter(df['global_step'], df['mean_reward'], color=color, alpha=0.3, s=20)
                interp_r = np.full_like(all_steps, df['mean_reward'].iloc[0], dtype=np.float64)
            else:
                interp_r = np.interp(all_steps, df['global_step'], df['mean_reward'])
                plt.plot(all_steps, interp_r, color=color, alpha=0.15, linewidth=1)
            interp_rewards.append(interp_r)
            
        interp_rewards = np.array(interp_rewards)
        mean_r = np.mean(interp_rewards, axis=0)
        
        if len(all_steps) > 1:
            std_r = np.std(interp_rewards, axis=0)
            plt.plot(all_steps, mean_r, color=color, label=f"{algo.upper()} (Mean)", linewidth=2.5, marker='o', markersize=4)
            plt.fill_between(all_steps, mean_r - std_r, mean_r + std_r, color=color, alpha=0.2)
        else:
            plt.scatter(all_steps, mean_r, color=color, label=f"{algo.upper()} (Point)", s=100, edgecolors='black', zorder=5)

    plt.title('Average Return vs Time Step', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step')
    plt.ylabel('Average Return')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # 3. Build PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor("#2E5077"), alignment=1)
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.4 * inch))
    
    # Metadata
    story.append(Paragraph("System Environment", styles['Heading2']))
    meta = [
        ["Platform", platform.platform()],
        ["Python", platform.python_version()],
        ["PyTorch", torch.__version__],
        ["CUDA Available", str(torch.cuda.is_available())],
    ]
    if torch.cuda.is_available():
        meta.append(["GPU", torch.cuda.get_device_name(0)])
    t_meta = Table(meta, colWidths=[1.5*inch, 4.5*inch])
    t_meta.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke)]))
    story.append(t_meta)
    story.append(Spacer(1, 0.3 * inch))

    # Hyperparameters
    hp = extract_hyperparams(work_dir)
    if hp:
        story.append(Paragraph("Hyperparameters", styles['Heading2']))
        hp_list = [[str(k), str(v)] for k, v in list(hp.items())[:20]] # Limit to 20 for space
        t_hp = Table([["Parameter", "Value"]] + hp_list, colWidths=[2.5*inch, 3.5*inch])
        t_hp.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D1E8E2")), ('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
        story.append(t_hp)
        story.append(Spacer(1, 0.3 * inch))
    
    # Plot
    if os.path.exists(plot_path):
        story.append(Paragraph("Average Return vs Time Step", styles['Heading2']))
        story.append(Image(plot_path, width=6*inch, height=3.5*inch))
        story.append(Spacer(1, 0.3 * inch))

    # Per-Algorithm Run Statistics Table
    for algo in sorted(algo_run_stats.keys()):
        run_stats = algo_run_stats[algo]
        algo_title = f"{algo}_{device_str}:{algo.upper()}"
        story.append(Paragraph(f"title {algo_title}", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))

        # Build the per-run table
        table_data = [["Run", "Mean \u00b1 Std"]]
        all_means = []
        all_stds = []
        run_dir_names = []

        for i, stat in enumerate(run_stats):
            table_data.append([
                f"run #{i}",
                f"{stat['mean']:.4f} \u00b1 {stat['std']:.4f}"
            ])
            all_means.append(stat['mean'])
            all_stds.append(stat['std'])
            run_dir_names.append(stat['run_name'])

        # Overall mean row
        overall_mean = float(np.mean(all_means))
        overall_std = float(np.mean(all_stds))
        table_data.append(["mean", f"{overall_mean:.4f} \u00b1 {overall_std:.4f}"])

        t_runs = Table(table_data, colWidths=[1.5*inch, 3*inch])
        t_runs.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#D1E8E2")),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#E8F4FD")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]))
        story.append(t_runs)
        story.append(Spacer(1, 0.15 * inch))

        # Runs involved listing
        runs_involved_str = f"runs_involved={run_dir_names}"
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        story.append(Paragraph(runs_involved_str, styles['Normal']))
        story.append(Paragraph(timestamp_str, styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))
    
    doc.build(story)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--out", default="experiment_report.pdf")
    args = parser.parse_args()
    generate_report(args.work_dir, args.out)
