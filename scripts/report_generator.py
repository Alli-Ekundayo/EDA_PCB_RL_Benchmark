import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def generate_report(log_files, output_pdf, title="RL PCB Training Report", window_size=10):
    if isinstance(log_files, str):
        log_files = [log_files]

    # 1. Parse log files
    dfs = []
    for log_file in log_files:
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
        if data:
            dfs.append(pd.DataFrame(data))

    if not dfs:
        print("No valid logs found.")
        return

    # 2. Create Plots
    os.makedirs("tmp_plots", exist_ok=True)
    plot_path = "tmp_plots/metrics.png"
    
    plt.figure(figsize=(10, 8))
    
    # Plot 1: Reward
    plt.subplot(2, 1, 1)
    
    if all('mean_reward' in df.columns and 'global_step' in df.columns for df in dfs):
        # Merge all dataframes on global_step
        merged_df = dfs[0][['global_step', 'mean_reward']].copy().rename(columns={'mean_reward': 'reward_0'})
        for i, df in enumerate(dfs[1:], 1):
            temp = df[['global_step', 'mean_reward']].rename(columns={'mean_reward': f'reward_{i}'})
            merged_df = pd.merge(merged_df, temp, on='global_step', how='outer').sort_values('global_step')
        
        # Forward fill to handle uneven steps
        merged_df = merged_df.ffill()
        
        reward_cols = [c for c in merged_df.columns if c.startswith('reward_')]
        merged_df['mean_reward'] = merged_df[reward_cols].mean(axis=1)
        merged_df['std_reward'] = merged_df[reward_cols].std(axis=1).fillna(0)
        
        # Smoothing
        merged_df['smoothed_mean'] = merged_df['mean_reward'].rolling(window=window_size, min_periods=1).mean()
        merged_df['smoothed_std'] = merged_df['std_reward'].rolling(window=window_size, min_periods=1).mean()
        
        plt.plot(merged_df['global_step'], merged_df['smoothed_mean'], color='#2ecc71', linewidth=2, label='Smoothed Mean Reward')
        if len(dfs) > 1:
            plt.fill_between(merged_df['global_step'], 
                             merged_df['smoothed_mean'] - merged_df['smoothed_std'], 
                             merged_df['smoothed_mean'] + merged_df['smoothed_std'], 
                             color='#2ecc71', alpha=0.3, label='Std Dev')
        else:
            plt.plot(merged_df['global_step'], merged_df['mean_reward'], color='#2ecc71', alpha=0.3, label='Raw Reward')
            
        plt.title('Training Reward', fontsize=14, fontweight='bold')
        plt.ylabel('Mean Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 2: Entropy or Critic Loss (using the first log for simplicity)
    plt.subplot(2, 1, 2)
    df0 = dfs[0]
    if 'entropy' in df0.columns:
        smoothed_entropy = df0['entropy'].rolling(window=window_size, min_periods=1).mean()
        plt.plot(df0['global_step'], smoothed_entropy, color='#e74c3c', linewidth=2)
        plt.title('Policy Entropy (Run 1)', fontsize=14, fontweight='bold')
        plt.ylabel('Entropy')
        plt.grid(True, alpha=0.3)
    elif 'critic_loss' in df0.columns:
        smoothed_critic = df0['critic_loss'].rolling(window=window_size, min_periods=1).mean()
        plt.plot(df0['global_step'], smoothed_critic, color='#3498db', linewidth=2)
        plt.title('Critic Loss (Run 1)', fontsize=14, fontweight='bold')
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
    
    if os.path.exists(plot_path):
        story.append(Image(plot_path, width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Experiment Summary", styles['Heading2']))
    if 'merged_df' in locals() and not merged_df.empty:
        last_row = merged_df.iloc[-1]
        summary_data = [
            ["Metric", "Final Value"],
            ["Global Steps", f"{last_row['global_step']:.0f}"],
            ["Final Mean Reward", f"{last_row.get('smoothed_mean', 0):.4f}"],
            ["Max Mean Reward Achieved", f"{merged_df['smoothed_mean'].max():.4f}"]
        ]
        
        t = Table(summary_data, colWidths=[2 * inch, 2 * inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5 * inch))
    
    doc.build(story)
    print(f"Report generated: {output_pdf}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", nargs='+', default=["training_run.log"], help="Path(s) to log files")
    parser.add_argument("--out", default="experiment_report.pdf", help="Output PDF path")
    parser.add_argument("--window", type=int, default=10, help="Moving average window size")
    args = parser.parse_args()
    generate_report(args.log, args.out, window_size=args.window)
