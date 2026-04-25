import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# RL PCB Benchmark & Diagnostics\n",
                "This notebook performs a comprehensive evaluation of PPO, TD3, and SAC for PCB component placement, including training diagnostics and physical layout visualization."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Setup Environment\n",
                "!git clone https://github.com/Alli-Ekundayo/EDA_PCB_RL_Benchmark.git\n",
                "%cd EDA_PCB_RL_Benchmark\n",
                "!pip install -r requirements.txt\n",
                "!pip install torch-geometric reportlab pandas matplotlib\n",
                "!sed -i 's/source .venv\\/bin\\/activate/# source .venv\\/bin\\/activate/g' tests/*/run.sh"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Phase 1: Tiny Board Sanity Check (8x8)\n",
                "We start with a simplified 8x8 board to verify that the agent can learn basic spatial relationships and connectivity optimization."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!bash tests/04_tiny_sanity_check/run.sh\n",
                "from IPython.display import Image, display\n",
                "display(Image('runs/tiny_sanity/tiny_placement.png'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Phase 2: Full Benchmark Comparison\n",
                "Running PPO and TD3 for a fair comparison over 100,000 steps with optimized hyperparameters."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!bash tests/01_ppo_research_run/run.sh\n",
                "!bash tests/02_td3_extended_run/run.sh"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Phase 3: Diagnostic Analysis\n",
                "Visualizing the training progress and reward distribution."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import re\n",
                "\n",
                "def load_metrics(log_path):\n",
                "    data = []\n",
                "    with open(log_path, 'r') as f:\n",
                "        for line in f:\n",
                "            entry = {}\n",
                "            for pair in line.strip().split(' '):\n",
                "                if '=' in pair:\n",
                "                    k, v = pair.split('=')\n",
                "                    entry[k.replace('train/', '')] = float(v)\n",
                "            if entry: data.append(entry)\n",
                "    return pd.DataFrame(data)\n",
                "\n",
                "ppo_df = load_metrics('tests/01_ppo_research_run/results/ppo_seed_42/training.log')\n",
                "td3_df = load_metrics('tests/02_td3_extended_run/results/td3_seed_42/training.log')\n",
                "\n",
                "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
                "\n",
                "# 1. Learning Curves\n",
                "axes[0,0].plot(ppo_df['global_step'], ppo_df['mean_reward'], label='PPO')\n",
                "axes[0,0].plot(td3_df['global_step'], td3_df['mean_reward'], label='TD3')\n",
                "axes[0,0].set_title('Mean Reward')\n",
                "axes[0,0].legend()\n",
                "\n",
                "# 2. Critic Loss\n",
                "if 'value_loss' in ppo_df: axes[0,1].plot(ppo_df['global_step'], ppo_df['value_loss'], label='PPO Value Loss')\n",
                "if 'critic_loss' in td3_df: axes[0,1].plot(td3_df['global_step'], td3_df['critic_loss'], label='TD3 Critic Loss')\n",
                "axes[0,1].set_title('Critic/Value Loss')\n",
                "axes[0,1].legend()\n",
                "\n",
                "# 3. Reward Distribution\n",
                "axes[1,0].hist(td3_df['mean_reward'], bins=30, alpha=0.5, label='TD3 Rewards')\n",
                "axes[1,0].set_title('Reward Distribution (TD3)')\n",
                "\n",
                "# 4. Diagnostics (Q-Value/Entropy)\n",
                "if 'mean_q' in td3_df: axes[1,1].plot(td3_df['global_step'], td3_df['mean_q'], label='TD3 Mean Q')\n",
                "if 'entropy' in ppo_df: axes[1,1].plot(ppo_df['global_step'], ppo_df['entropy'], label='PPO Entropy')\n",
                "axes[1,1].set_title('Q-Value / Entropy')\n",
                "axes[1,1].legend()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Phase 4: Placement Visualization\n",
                "Visualizing the actual PCB layout produced by the trained TD3 agent."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python3 scripts/visualize_placement.py --checkpoint tests/02_td3_extended_run/results/td3_seed_42/td3_final.pt --out final_td3_placement.png\n",
                "display(Image('final_td3_placement.png'))"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("EDA_PCB_Colab_Benchmark.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
