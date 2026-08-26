from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runtime H8 config from validation thresholds")
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=0.10)
    args = parser.parse_args()
    payload = json.loads(args.thresholds.read_text())
    values = payload["thresholds_by_layer_scheme_rate"]
    thresholds = {
        layer: schemes["dynamic_abs_w"][str(args.rate)]
        for layer, schemes in values.items()
    }
    config = {
        "scope": "Research sidecar DSA on DeepSeek-V2-Lite",
        "configs": [
            {
                "name": f"head_dynamic_abs_w_w8_b64_threshold_r{args.rate:g}",
                "path": "head",
                "width": 8,
                "block_size": 64,
                "rescue_fraction": args.rate,
                "strategy": "high_weight",
                "precision": "bf16",
                "promotion_policy": "validation_fixed_threshold",
                "promotion_threshold_by_layer": thresholds,
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
