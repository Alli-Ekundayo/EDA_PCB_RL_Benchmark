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
                                entry[k.replace('train/', '')] = float(v)
                    if 'global_step' in entry and 'mean_reward' in entry:
                        data.append(entry)
    return pd.DataFrame(data)

def extract_hyperparams(work_dir):
    """Try to extract hyperparameters from .pt files or config.yaml."""
    pt_files = glob.glob(os.path.join(work_dir, "**", "*.pt"), recursive=True)
    if pt_files:
        try:
            checkpoint = torch.load(pt_files[0], map_location='cpu')
            if 'config' in checkpoint:
                return checkpoint['config']
        except:
            pass
    
    config_files = glob.glob(os.path.join(work_dir, "**", "config.yaml"), recursive=True)
    if not config_files:
        # Check parent dir
        config_files = glob.glob(os.path.join(os.path.dirname(work_dir), "configs", "*.yaml"))
        
    if config_files:
        try:
            import yaml
            with open(config_files[0], 'r') as f:
                return yaml.safe_load(f)
        except:
            pass
    return None

def generate_report(work_dir, output_pdf, title="Reinforcement Learning - Experiment Report"):
    # 1. Sweep directory for log files
    log_files = glob.glob(os.path.join(work_dir, "**", "training.log"), recursive=True)
    if not log_files:
        print(f"No training.log files found in {work_dir}")
        return

    # Dictionary: algo -> list of DataFrames (each representing a run/seed)
    algo_data = {}
    run_details = [] # List of (algo, run_name, mean, std, final)

    for log_file in log_files:
        parent_dir = os.path.basename(os.path.dirname(log_file))
        algo = parent_dir.split('_')[0] if '_' in parent_dir else "unknown"
        
        df = parse_log(log_file)
        if not df.empty:
            if algo not in algo_data:
                algo_data[algo] = []
            algo_data[algo].append(df)
            
            # Calculate stats for this individual run
            rewards = df['mean_reward'].values
            run_details.append([
                algo.upper(),
                parent_dir,
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
    colors_map = {'ppo': '#3498db', 'td3': '#e74c3c', 'sac': '#2ecc71', 'unknown': '#9b59b6'}
    
    summary_data = [["Algorithm", "Runs", "Max Mean Reward", "Final Mean Reward"]]

    for algo, dfs in algo_data.items():
        # Align all runs to a common step grid
        all_steps = pd.concat([df['global_step'] for df in dfs]).sort_values().unique()
        color = colors_map.get(algo.lower(), colors_map['unknown'])
        
        interp_rewards = []
        for i, df in enumerate(dfs):
            df = df.drop_duplicates(subset=['global_step'])
            interp_r = np.interp(all_steps, df['global_step'], df['mean_reward'])
            interp_rewards.append(interp_r)
            # Plot individual run on the same axis with low alpha
            plt.plot(all_steps, interp_r, color=color, alpha=0.15, linewidth=1)
            
        interp_rewards = np.array(interp_rewards)
        mean_r = np.mean(interp_rewards, axis=0)
        std_r = np.std(interp_rewards, axis=0)
        
        plt.plot(all_steps, mean_r, color=color, label=f"{algo.upper()} (Aggregate Mean)", linewidth=2.5)
        plt.fill_between(all_steps, mean_r - std_r, mean_r + std_r, color=color, alpha=0.2)
        
        max_r = np.max(mean_r)
        final_r = mean_r[-1]
        summary_data.append([algo.upper(), str(len(dfs)), f"{max_r:.4f}", f"{final_r:.4f}"])

    plt.title('Learning Curves: Return vs Timesteps (All Runs)', fontsize=14, fontweight='bold')
    plt.xlabel('Global Step (Timesteps)')
    plt.ylabel('Mean Return / Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Build PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title Page
    header_style = ParagraphStyle(
        'Header', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor("#2E5077"), spaceAfter=15
    )
    story.append(Paragraph(title, header_style))
    story.append(Paragraph("<b>Focus: Training Performance, Stability and Hyperparameters</b>", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))
    
    # System Info
    sys_info = [
        ["Hardware & Libraries", ""],
        ["Platform", platform.platform()],
        ["Python", platform.python_version()],
        ["PyTorch", torch.__version__],
        ["CUDA Available", str(torch.cuda.is_available())],
    ]
    story.append(Paragraph("Environment Metadata", styles['Heading2']))
    t_sys = Table(sys_info, colWidths=[2 * inch, 4 * inch])
    t_sys.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#D1E8E2")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(t_sys)
    story.append(Spacer(1, 0.4 * inch))

    # Hyperparameters
    story.append(Paragraph("Hyperparameter Logging", styles['Heading2']))
    hp = extract_hyperparams(work_dir)
    if hp:
        hp_data = [["Parameter", "Value"]]
        if isinstance(hp, dict):
            for k, v in hp.items():
                hp_data.append([str(k), str(v)])
        
        t_hp = Table(hp_data, colWidths=[2.5 * inch, 3.5 * inch])
        t_hp.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#D1E8E2")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))
        story.append(t_hp)
    else:
        story.append(Paragraph("<i>No explicit hyperparameter data recovered from checkpoints or configs.</i>", styles['Normal']))
    
    story.append(PageBreak())

    # Add Plot
    if os.path.exists(plot_path):
        story.append(Paragraph("Learning Curves (All Runs)", styles['Heading2']))
        story.append(Paragraph("Light lines indicate individual seeds. Bold lines indicate the algorithm mean.", styles['Italic']))
        story.append(Image(plot_path, width=6*inch, height=3.5*inch))
        story.append(Spacer(1, 0.3*inch))
    
    # Aggregate Summary
    story.append(Paragraph("Algorithm Performance (Aggregate)", styles['Heading2']))
    t_sum = Table(summary_data, colWidths=[1.5 * inch, 1.0 * inch, 1.5 * inch, 1.5 * inch])
    t_sum.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#D1E8E2")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_sum)
    story.append(Spacer(1, 0.4 * inch))

    # Individual Run Details
    story.append(Paragraph("Detailed Run Statistics", styles['Heading2']))
    story.append(Paragraph("Individual scores (Mean ± Dev) for every run/seed.", styles['Italic']))
    run_headers = [["Algo", "Run Name", "Mean Reward", "Std Dev", "Final Reward"]]
    t_runs = Table(run_headers + run_details, colWidths=[0.8*inch, 2.0*inch, 1.1*inch, 1.1*inch, 1.1*inch])
    t_runs.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#D1E8E2")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(t_runs)

    doc.build(story)
    print(f"Experiment Report generated: {output_pdf}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate PDF Experiment Report from parallel runs")
    parser.add_argument("--work_dir", default="runs/experiments", help="Directory containing run logs")
    parser.add_argument("--out", default="experiment_report.pdf", help="Output PDF file")
    args = parser.parse_args()
    generate_report(args.work_dir, args.out)
