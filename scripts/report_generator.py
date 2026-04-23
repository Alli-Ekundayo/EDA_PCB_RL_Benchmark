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

def generate_report(work_dir, output_pdf, title="Reinforcement Learning - Experiment Report"):
    # 1. Sweep directory for log files
    log_files = glob.glob(os.path.join(work_dir, "**", "training.log"), recursive=True)
    if not log_files:
        # Fallback to root log if seed dirs don't exist
        log_files = glob.glob(os.path.join(work_dir, "training.log"))
        
    if not log_files:
        print(f"No training.log files found in {work_dir}")
        return

    algo_data = {}
    run_details = []

    for log_file in log_files:
        parent_dir = os.path.basename(os.path.dirname(log_file))
        # Logic to extract algo from path or filename
        algo = "ppo" # Default
        if "td3" in log_file.lower() or "td3" in parent_dir.lower(): algo = "td3"
        elif "sac" in log_file.lower() or "sac" in parent_dir.lower(): algo = "sac"
        
        df = parse_log(log_file)
        if not df.empty:
            if algo not in algo_data: algo_data[algo] = []
            algo_data[algo].append(df)
            
            rewards = df['mean_reward'].values
            run_details.append([
                algo.upper(),
                parent_dir if parent_dir else "root",
                f"{np.mean(rewards):.4f}",
                f"{np.std(rewards):.4f}",
                f"{rewards[-1]:.4f}"
            ])

    if not algo_data:
        print("No valid training data found in logs.")
        return

    # 2. Process Data and Create Plots
    os.makedirs("tmp_plots", exist_ok=True)
    plot_path = "tmp_plots/learning_curves.png"
    
    plt.figure(figsize=(10, 6))
    colors_map = {'ppo': '#3498db', 'td3': '#e74c3c', 'sac': '#2ecc71'}
    summary_data = [["Algorithm", "Runs", "Max Mean Reward", "Final Mean Reward"]]

    for algo, dfs in algo_data.items():
        color = colors_map.get(algo.lower(), '#9b59b6')
        all_steps = pd.concat([df['global_step'] for df in dfs]).sort_values().unique()
        
        if len(all_steps) == 0: continue
        
        interp_rewards = []
        for df in dfs:
            df = df.drop_duplicates(subset=['global_step'])
            if len(df) == 1:
                # Handle single-point runs
                plt.scatter(df['global_step'], df['mean_reward'], color=color, alpha=0.3, s=20)
                interp_r = np.full_like(all_steps, df['mean_reward'].iloc[0])
            else:
                interp_r = np.interp(all_steps, df['global_step'], df['mean_reward'])
                plt.plot(all_steps, interp_r, color=color, alpha=0.15, linewidth=1)
            interp_rewards.append(interp_r)
            
        interp_rewards = np.array(interp_rewards)
        mean_r = np.mean(interp_rewards, axis=0)
        
        if len(all_steps) > 1:
            std_r = np.std(interp_rewards, axis=0)
            plt.plot(all_steps, mean_r, color=color, label=f"{algo.upper()} (Mean)", linewidth=2.5)
            plt.fill_between(all_steps, mean_r - std_r, mean_r + std_r, color=color, alpha=0.2)
        else:
            plt.scatter(all_steps, mean_r, color=color, label=f"{algo.upper()} (Point)", s=100, edgecolors='black', zorder=5)
        
        summary_data.append([algo.upper(), str(len(dfs)), f"{np.max(mean_r):.4f}", f"{mean_r[-1]:.4f}"])

    plt.title('Training Convergence: Return vs Timesteps', fontsize=14, fontweight='bold')
    plt.xlabel('Global Step')
    plt.ylabel('Mean Return')
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
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.4 * inch))
    
    # Metadata
    story.append(Paragraph("System Environment", styles['Heading2']))
    meta = [
        ["Platform", platform.platform()],
        ["Python", platform.python_version()],
        ["PyTorch", torch.__version__],
        ["CUDA Available", str(torch.cuda.is_available())],
    ]
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
        story.append(Paragraph("Learning Curves", styles['Heading2']))
        story.append(Image(plot_path, width=6*inch, height=3.5*inch))
        story.append(Spacer(1, 0.3 * inch))

    # Summary Table
    story.append(Paragraph("Performance Summary", styles['Heading2']))
    t_sum = Table(summary_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1.5*inch])
    t_sum.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#D1E8E2")), ('GRID', (0,0), (-1,-1), 0.5, colors.black), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(t_sum)
    
    doc.build(story)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--out", default="experiment_report.pdf")
    args = parser.parse_args()
    generate_report(args.work_dir, args.out)
