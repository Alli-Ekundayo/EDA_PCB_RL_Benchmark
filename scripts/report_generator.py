import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def generate_report(log_file, output_pdf, title="RL PCB Training Report"):
    # 1. Parse log file
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
                    data.append(entry)
    
    df = pd.DataFrame(data)
    
    # 2. Create Plots
    os.makedirs("tmp_plots", exist_ok=True)
    plot_path = "tmp_plots/metrics.png"
    
    plt.figure(figsize=(10, 8))
    
    # Plot 1: Reward
    plt.subplot(2, 1, 1)
    if 'mean_reward' in df.columns:
        plt.plot(df['global_step'], df['mean_reward'], color='#2ecc71', linewidth=2)
        plt.title('Training Reward', fontsize=14, fontweight='bold')
        plt.ylabel('Mean Reward')
        plt.grid(True, alpha=0.3)
    
    # Plot 2: Entropy or Critic Loss
    plt.subplot(2, 1, 2)
    if 'entropy' in df.columns:
        plt.plot(df['global_step'], df['entropy'], color='#e74c3c', linewidth=2)
        plt.title('Policy Entropy', fontsize=14, fontweight='bold')
        plt.ylabel('Entropy')
        plt.grid(True, alpha=0.3)
    elif 'critic_loss' in df.columns:
        plt.plot(df['global_step'], df['critic_loss'], color='#3498db', linewidth=2)
        plt.title('Critic Loss', fontsize=14, fontweight='bold')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    plt.xlabel('Global Step')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Build PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    header_style = ParagraphStyle(
        'Header',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=20
    )
    story.append(Paragraph(title, header_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.5 * inch))
    
    # Add Plot
    if os.path.exists(plot_path):
        story.append(Image(plot_path, width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 0.2*inch))
    
    # Summary Table
    story.append(Paragraph("Experiment Summary", styles['Heading2']))
    last_row = df.iloc[-1]
    summary_data = [
        ["Metric", "Final Value"],
        ["Global Steps", f"{last_row['global_step']:.0f}"],
        ["Final Reward", f"{last_row.get('mean_reward', 0):.4f}"],
    ]
    if 'critic_loss' in last_row:
        summary_data.append(["Final Critic Loss", f"{last_row['critic_loss']:.4f}"])
    if 'entropy' in last_row:
        summary_data.append(["Final Entropy", f"{last_row['entropy']:.4f}"])
    summary_data.append(["Max Reward Achieved", f"{df['mean_reward'].max():.4f}"])
    
    t = Table(summary_data, colWidths=[2 * inch, 2 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * inch))
    
    # Tables and Summary are already added.
    
    doc.build(story)
    print(f"Report generated: {output_pdf}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="training_run.log")
    parser.add_argument("--out", default="experiment_report.pdf")
    args = parser.parse_args()
    generate_report(args.log, args.out)
