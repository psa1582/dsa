from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluate_progressive_threshold import (
    build_policy,
    discover,
    encode,
    save_details,
    verifier_config,
)
from temporal_dsa.trace import load_trace
from temporal_dsa.verifier import replay_verifier_trace
from temporal_dsa.verifier_scoring import dynamic_head_indices, score_head_sparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Save dynamic-H8 threshold details for MLA")
    parser.add_argument("--heldout-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() != 2:
        raise RuntimeError("expected CUDA_VISIBLE_DEVICES=0,1")
    selection = json.loads(args.selection.read_text())
    policy_spec = next(
        row for row in selection["policies"] if row["selection_role"] == "Aggressive"
    )
    scales = {int(key): float(value) for key, value in selection["gamma_scale"].items()}
    thresholds = json.loads(args.thresholds.read_text())[
        "thresholds_by_layer_scheme_rate"
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    paths = discover(args.heldout_roots)
    for index, path in enumerate(paths, start=1):
        trace = load_trace(path)
        queries, weights, keys = encode(trace, args.checkpoint_root)
        ids = dynamic_head_indices(weights, 8, "high_weight")
        partial = score_head_sparse(queries, weights, keys, ids).cpu().numpy()
        policy = build_policy(policy_spec, trace.layer, scales)
        for target in (0.05, 0.10, 0.15, 0.20):
            name = f"head_dynamic_abs_w_w8_b64_threshold_r{target:g}"
            rows = {}

            def collect(step, values):
                rows[step] = values

            replay_verifier_trace(
                trace,
                partial,
                policy=policy,
                config=verifier_config(name),
                promotion_threshold=thresholds[str(trace.layer)]["dynamic_abs_w"][str(target)],
                detail_callback=collect,
            )
            save_details(
                args.output / f"{path.stem}__Aggressive__{name}.npz", rows
            )
        del queries, weights, keys
        torch.cuda.empty_cache()
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)


if __name__ == "__main__":
    main()
