import os
import json
import glob
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
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
                            k, v = p.split('=')
                            entry[k.replace('train/', '')] = float(v)
                    if 'global_step' in entry and 'mean_reward' in entry:
                        data.append(entry)
    return pd.DataFrame(data)

def generate_report(work_dir, output_pdf, title="RL PCB Parallel Training Report"):
    # 1. Sweep directory for log files
    log_files = glob.glob(os.path.join(work_dir, "**", "training.log"), recursive=True)
    if not log_files:
        print(f"No training.log files found in {work_dir}")
        return

    # Dictionary: algo -> list of DataFrames
    algo_data = {}
    for log_file in log_files:
        # Assuming directory structure like ppo_seed_42
        parent_dir = os.path.basename(os.path.dirname(log_file))
        algo = parent_dir.split('_')[0] if '_' in parent_dir else "unknown"
        
        df = parse_log(log_file)
        if not df.empty:
            if algo not in algo_data:
                algo_data[algo] = []
            algo_data[algo].append(df)

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
        
        interp_rewards = []
        for df in dfs:
            # Drop duplicates to allow interpolation
            df = df.drop_duplicates(subset=['global_step'])
            # Interpolate onto the common step grid
            interp_r = np.interp(all_steps, df['global_step'], df['mean_reward'])
            interp_rewards.append(interp_r)
            
        interp_rewards = np.array(interp_rewards)
        mean_r = np.mean(interp_rewards, axis=0)
        std_r = np.std(interp_rewards, axis=0)
        
        color = colors_map.get(algo.lower(), colors_map['unknown'])
        plt.plot(all_steps, mean_r, color=color, label=f"{algo.upper()} (mean)", linewidth=2)
        plt.fill_between(all_steps, mean_r - std_r, mean_r + std_r, color=color, alpha=0.2)
        
        max_r = np.max(mean_r)
        final_r = mean_r[-1]
        summary_data.append([algo.upper(), str(len(dfs)), f"{max_r:.4f}", f"{final_r:.4f}"])

    plt.title('Learning Curves (Reward vs Timesteps)', fontsize=14, fontweight='bold')
    plt.xlabel('Global Step')
    plt.ylabel('Mean Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Build PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    header_style = ParagraphStyle(
        'Header', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor("#2c3e50"), spaceAfter=20
    )
    story.append(Paragraph(title, header_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))
    
    # System Info
    sys_info = [
        ["System Environment", ""],
        ["Platform", platform.platform()],
        ["Processor", platform.processor()],
        ["Python Version", platform.python_version()],
        ["PyTorch Version", torch.__version__],
        ["CUDA Available", str(torch.cuda.is_available())],
    ]
    if torch.cuda.is_available():
        sys_info.append(["CUDA Device", torch.cuda.get_device_name(0)])
        
    story.append(Paragraph("Machine & Library Information", styles['Heading2']))
    t_sys = Table(sys_info, colWidths=[2 * inch, 4 * inch])
    t_sys.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#ecf0f1")),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    story.append(t_sys)
    story.append(Spacer(1, 0.4 * inch))

    # Add Plot
    if os.path.exists(plot_path):
        story.append(Paragraph("Learning Curves", styles['Heading2']))
        story.append(Image(plot_path, width=6*inch, height=3.5*inch))
        story.append(Spacer(1, 0.2*inch))
    
    # Summary Table
    story.append(Paragraph("Experiment Summary", styles['Heading2']))
    t = Table(summary_data, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#ecf0f1")),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * inch))
    
    doc.build(story)
    print(f"Report generated: {output_pdf}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate PDF report from parallel runs")
    parser.add_argument("--work_dir", default="runs/experiments", help="Directory containing run logs")
    parser.add_argument("--out", default="experiment_report.pdf", help="Output PDF file")
    args = parser.parse_args()
    generate_report(args.work_dir, args.out)
