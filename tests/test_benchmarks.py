from __future__ import annotations

import json
from pathlib import Path

from evaluation.compare_baselines import compare_baselines


def test_rl_pcb_assets_imported():
    required_paths = [
        "data/boards/rl_pcb/base/training.pcb",
        "data/boards/rl_pcb/base/evaluation.pcb",
        "data/boards/rl_pcb/base_opt/training.pcb",
        "data/boards/rl_pcb/base_opt/evaluation.pcb",
        "data/boards/rl_pcb/base_raw/voltage_datalogger_adc0.kicad_pcb",
        "data/boards/rl_pcb/base_raw/voltage_datalogger_adc2.kicad_pcb",
        "data/boards/rl_pcb/base_raw/voltage_datalogger_afe.kicad_pcb",
        "data/benchmarks/rl_pcb_model_benchmarks.json",
        "data/benchmarks/hp_td3.json",
        "data/benchmarks/hp_sac.json",
        "data/benchmarks/rl_pcb_reports/td3_cpu_eval_mean.pdf",
        "data/benchmarks/rl_pcb_reports/td3_cuda_eval_mean.pdf",
        "data/benchmarks/rl_pcb_reports/sac_cpu_eval_mean.pdf",
        "data/benchmarks/rl_pcb_reports/sac_cuda_eval_mean.pdf",
        "data/benchmarks/rl_pcb_reports/dreamerv3_cuda_eval_mean.pdf",
        "data/benchmarks/rl_pcb_reports/dreamerv3_cpu_fast_eval_mean.pdf",
    ]
    for path in required_paths:
        assert Path(path).exists(), f"Missing imported RL_PCB artifact: {path}"


def test_compare_baselines_uses_expected_rl_pcb_metrics():
    benchmark_file = Path("data/benchmarks/rl_pcb_model_benchmarks.json")
    benchmark_data = json.loads(benchmark_file.read_text(encoding="utf-8"))
    models = benchmark_data["models"]
    out = compare_baselines()

    assert out["td3_hpwl_mean"] == models["td3_cuda"]["hpwl_20_mean"]
    assert out["sac_hpwl_mean"] == models["sac_cuda"]["hpwl_20_mean"]
    assert out["dreamerv3_hpwl_mean"] == models["dreamerv3_cuda"]["hpwl_20_mean"]
    assert out["td3_routed_mean"] == models["td3_cuda"]["routed_20_mean"]
    assert out["sac_routed_mean"] == models["sac_cuda"]["routed_20_mean"]
    assert out["dreamerv3_routed_mean"] == models["dreamerv3_cuda"]["routed_20_mean"]

    best_hpwl = min(
        models["td3_cuda"]["hpwl_20_mean"],
        models["sac_cuda"]["hpwl_20_mean"],
        models["dreamerv3_cuda"]["hpwl_20_mean"],
    )
    best_routed = min(
        models["td3_cuda"]["routed_20_mean"],
        models["sac_cuda"]["routed_20_mean"],
        models["dreamerv3_cuda"]["routed_20_mean"],
    )
    assert out["benchmark_hpwl_target"] == best_hpwl
    assert out["benchmark_routed_target"] == best_routed
