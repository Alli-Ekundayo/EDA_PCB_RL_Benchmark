import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# RL PCB Benchmark Comparison\n",
                "This notebook pulls the EDA_PCB_RL_Benchmark repository, sets up the environment on Google Colab, trains three RL models (PPO, TD3, SAC) for 10,000 timesteps each, and compiles the results to draw a conclusion."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Clone the repository\n",
                "!git clone https://github.com/Alli-Ekundayo/EDA_PCB_RL_Benchmark.git\n",
                "%cd EDA_PCB_RL_Benchmark"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Install dependencies\n",
                "!pip install -r requirements.txt\n",
                "!pip install torch-geometric reportlab pandas matplotlib\n",
                "# Disable virtual environment sourcing in the bash scripts as Colab handles packages globally\n",
                "!sed -i 's/source .venv\\/bin\\/activate/# source .venv\\/bin\\/activate/g' tests/*/run.sh"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Run PPO Benchmark\n",
                "!bash tests/01_ppo_research_run/run.sh"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Run TD3 Benchmark\n",
                "!bash tests/02_td3_research_run/run.sh"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Run SAC Benchmark\n",
                "!bash tests/03_sac_research_run/run.sh"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Compile Results into a Single Benchmark Report\n",
                "!python3 scripts/generate_benchmark_report.py \\\n",
                "    --logs tests/01_ppo_research_run/results/training.log \\\n",
                "           tests/02_td3_research_run/results/training.log \\\n",
                "           tests/03_sac_research_run/results/training.log \\\n",
                "    --labels PPO TD3 SAC \\\n",
                "    --out benchmark_results_compiled.pdf\n",
                "\n",
                "from IPython.display import IFrame\n",
                "IFrame('benchmark_results_compiled.pdf', width=800, height=600)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("EDA_PCB_Colab_Benchmark.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
