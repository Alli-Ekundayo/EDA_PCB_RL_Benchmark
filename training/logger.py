from __future__ import annotations
import os
from typing import Dict

LOG_FILE = None

def log_dict(payload: Dict[str, float]) -> None:
    pairs = " ".join([f"{k}={v:.4f}" for k, v in sorted(payload.items())])
    print(pairs)
    
    if LOG_FILE:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(pairs + "\n")
