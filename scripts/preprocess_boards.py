from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/boards/sample_board.json")
    args = parser.parse_args()

    payload = {
        "components": [
            {"ref": "U1", "class_id": 0, "nets": [1, 2]},
            {"ref": "R1", "class_id": 1, "nets": [1]},
            {"ref": "C1", "class_id": 1, "nets": [2]},
        ],
        "nets": {"1": ["U1", "R1"], "2": ["U1", "C1"]},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
