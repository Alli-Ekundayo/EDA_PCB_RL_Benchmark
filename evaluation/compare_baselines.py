from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


def compare_baselines(actual_results: Optional[Dict[str, float]] = None, benchmark_path: str = "data/benchmarks/rl_pcb_model_benchmarks.json") -> Dict[str, float]:
    data = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    models = data["models"]

    td3_hpwl = float(models["td3_cuda"]["hpwl_20_mean"])
    sac_hpwl = float(models["sac_cuda"]["hpwl_20_mean"])
    dreamer_hpwl = float(models["dreamerv3_cuda"]["hpwl_20_mean"])
    td3_routed = float(models["td3_cuda"]["routed_20_mean"])
    sac_routed = float(models["sac_cuda"]["routed_20_mean"])
    dreamer_routed = float(models["dreamerv3_cuda"]["routed_20_mean"])

    ppo_hpwl = 0.0
    ppo_routed = 0.0
    if actual_results:
        ppo_hpwl = float(actual_results.get("ppo_hpwl_mean", 0.0))
        ppo_routed = float(actual_results.get("ppo_routed_mean", 0.0))

    hpwl_target = min(td3_hpwl, sac_hpwl, dreamer_hpwl)
    routed_target = min(td3_routed, sac_routed, dreamer_routed)

    return {
        "ppo_hpwl_mean": ppo_hpwl,
        "ppo_routed_mean": ppo_routed,
        "td3_hpwl_mean": td3_hpwl,
        "sac_hpwl_mean": sac_hpwl,
        "dreamerv3_hpwl_mean": dreamer_hpwl,
        "td3_routed_mean": td3_routed,
        "sac_routed_mean": sac_routed,
        "dreamerv3_routed_mean": dreamer_routed,
        "benchmark_hpwl_target": hpwl_target,
        "benchmark_routed_target": routed_target,
        "ppo_vs_benchmark_hpwl_delta": (ppo_hpwl - hpwl_target) if ppo_hpwl > 0 else 0.0,
        "ppo_vs_benchmark_routed_delta": (ppo_routed - routed_target) if ppo_routed > 0 else 0.0,
    }
