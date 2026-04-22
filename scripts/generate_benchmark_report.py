import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

plt.rc("axes", titlesize=16)
plt.rc("axes", labelsize=14)
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
plt.rc("legend", fontsize=12)

def generate_benchmark_report(log_files, labels, output_pdf, title="RL PCB Benchmark Comparison", window_size=10):
    if len(log_files) != len(labels):
        print("Error: Number of log files must match number of labels.")
        return

    dfs = []
    valid_labels = []
    
    # 1. Parse log files
    for log_file, label in zip(log_files, labels):
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
            valid_labels.append(label)

    if not dfs:
        print("No valid logs found to plot.")
        return

    # 2. Create Plot (Learning Curve with Moving Average)
    os.makedirs("tmp_plots", exist_ok=True)
    plot_path = "tmp_plots/benchmark_reward.png"
    
    plt.figure(figsize=(10, 6))
    
    colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f']
    
    final_metrics = []
    
    for i, (df, label) in enumerate(zip(dfs, valid_labels)):
        color = colors_list[i % len(colors_list)]
        
        if 'mean_reward' in df.columns and 'global_step' in df.columns:
            # Sort and fill
            df = df.sort_values('global_step')
            
            # Smoothing
            smoothed_mean = df['mean_reward'].rolling(window=window_size, min_periods=1).mean()
            
            # If we had multiple runs per agent we could do std dev, but here each log is one agent run
            plt.plot(df['global_step'], smoothed_mean, color=color, linewidth=2, label=f"{label} (smoothed)")
            plt.plot(df['global_step'], df['mean_reward'], color=color, alpha=0.2)
            
            final_metrics.append([
                label,
                f"{df['global_step'].iloc[-1]:.0f}",
                f"{smoothed_mean.iloc[-1]:.4f}",
                f"{df['mean_reward'].max():.4f}"
            ])
            
    plt.title('Training Reward Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Global Step')
    plt.ylabel('Mean Reward')
    plt.legend()
    plt.grid(True, which="both", axis="both", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Build PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=A4, rightMargin=0.75*inch, leftMargin=0.75*inch, topMargin=1*inch, bottomMargin=1*inch)
    styles = getSampleStyleSheet()
    story = []
    
    header_style = ParagraphStyle('Header', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor("#2c3e50"), spaceAfter=20)
    normal_style = styles['Normal']
    
    story.append(Paragraph(title, header_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Paragraph(f"Smoothing Window Size: {window_size}", normal_style))
    story.append(Spacer(1, 0.5 * inch))
    
    # Add Plot
    if os.path.exists(plot_path):
        img = Image(plot_path)
        img_width = 6.5 * inch
        img_height = img_width * (6 / 10) # preserve aspect ratio
        img.drawWidth = img_width
        img.drawHeight = img_height
        story.append(img)
        story.append(Spacer(1, 0.4 * inch))
    
    # Summary Table
    story.append(Paragraph("Algorithm Performance Summary", styles['Heading2']))
    story.append(Spacer(1, 0.2 * inch))
    
    table_data = [["Algorithm", "Total Steps", "Final Smoothed Reward", "Max Raw Reward"]] + final_metrics
    
    t = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.75*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#bdc3c7")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    
    story.append(t)
    story.append(Spacer(1, 0.5 * inch))
    
    # Conclusion text
    story.append(Paragraph("Conclusion", styles['Heading2']))
    best_algo = max(final_metrics, key=lambda x: float(x[2]))[0]
    conclusion_text = (f"Based on the training runs up to the max timesteps, the <b>{best_algo}</b> algorithm "
                       f"achieved the highest final smoothed reward. This visualization smoothed the "
                       f"learning curves using a moving average window of {window_size} to clearly display "
                       f"the learning trends across the different models.")
    story.append(Paragraph(conclusion_text, normal_style))
    
    doc.build(story)
    print(f"Benchmark report generated: {output_pdf}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs='+', required=True, help="Path(s) to log files")
    parser.add_argument("--labels", nargs='+', required=True, help="Labels for each log file")
    parser.add_argument("--out", default="benchmark_report.pdf", help="Output PDF path")
    parser.add_argument("--window", type=int, default=10, help="Moving average window size")
    args = parser.parse_args()
    generate_benchmark_report(args.logs, args.labels, args.out, window_size=args.window)
