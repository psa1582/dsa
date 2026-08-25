from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from temporal_dsa.approx import ApproxPolicy, replay_approx_trace
from temporal_dsa.trace import load_trace


def family_policy(family: str, *, name: str, gamma: float) -> ApproxPolicy:
    common: dict[str, Any] = {"name": name, "gamma": gamma}
    if family == "static":
        return ApproxPolicy(**common)
    if family == "dynamic_address":
        return ApproxPolicy(**common, dynamic_threshold=True)
    if family == "dynamic_previous_max":
        return ApproxPolicy(**common, dynamic_threshold=True, order="previous_max")
    if family in {"dynamic_bucket8", "dynamic_bucket16"}:
        return ApproxPolicy(
            **common, dynamic_threshold=True, order=family.removeprefix("dynamic_")
        )
    if family in {"streak2_bucket8", "streak4_bucket8"}:
        streak = int(family.removeprefix("streak").split("_", 1)[0])
        return ApproxPolicy(
            **common, dynamic_threshold=True, order="bucket8", cold_streak=streak
        )
    if family == "ema_bucket8_age8":
        return ApproxPolicy(
            **common,
            dynamic_threshold=True,
            order="bucket8",
            risk_model="ema",
            ema_alpha=0.9,
            volatility_lambda=2.0,
            age_cap=8,
        )
    raise ValueError(f"unknown family: {family}")


def build_policy(spec: dict[str, Any], layer: int, scales: dict[int, float]) -> ApproxPolicy:
    gamma = float(spec["multiplier"]) * scales[layer]
    policy = family_policy(str(spec["family"]), name=str(spec["config_id"]), gamma=gamma)
    return replace(policy, **spec.get("overrides", {}))


def worker(
    payload: tuple[str, list[dict[str, Any]], dict[int, float], int, int]
) -> pd.DataFrame:
    path_string, specs, scales, k, block_size = payload
    trace = load_trace(path_string)
    capture_path = Path(str(trace.metadata["source_capture"]))
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    teacher = capture["teacher_probabilities"].numpy().astype(np.float32)
    query_ids = [int(value) for value in trace.metadata.get("query_ids", range(teacher.shape[0]))]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        policy = build_policy(spec, trace.layer, scales)

        def collect(step: int, detail: dict[str, np.ndarray]) -> None:
            query_id = query_ids[step]
            length = int(trace.lengths[step])
            distribution = teacher[query_id, :length]
            dense_mass = float(distribution.sum(dtype=np.float64))
            base_mass = float(distribution[detail["baseline"]].sum(dtype=np.float64))
            approx_mass = float(distribution[detail["approximate"]].sum(dtype=np.float64))
            rows.append(
                {
                    "policy": spec["config_id"],
                    "selection_role": spec["selection_role"],
                    "step": step,
                    "layer": trace.layer,
                    "prompt_id": trace.prompt_id,
                    "workload": trace.workload,
                    "base_context_length": int(trace.lengths[0]) - 1,
                    "context_length": length,
                    "dense_teacher_mass": dense_mass,
                    "full_indexer_teacher_mass": base_mass,
                    "approx_teacher_mass": approx_mass,
                    "teacher_mass_delta": approx_mass - base_mass,
                    "teacher_mass_ratio": approx_mass / max(base_mass, 1e-30),
                }
            )

        replay_approx_trace(
            trace,
            policy=policy,
            k=k,
            block_size=block_size,
            history_mode="own",
            detail_callback=collect,
        )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby(["policy", "selection_role"], as_index=False).agg(
        transitions=("teacher_mass_ratio", "size"),
        full_indexer_teacher_mass_mean=("full_indexer_teacher_mass", "mean"),
        approx_teacher_mass_mean=("approx_teacher_mass", "mean"),
        teacher_mass_delta_mean=("teacher_mass_delta", "mean"),
        teacher_mass_delta_p1=("teacher_mass_delta", lambda x: x.quantile(0.01)),
        teacher_mass_delta_p5=("teacher_mass_delta", lambda x: x.quantile(0.05)),
        teacher_mass_delta_median=("teacher_mass_delta", "median"),
        teacher_mass_delta_worst=("teacher_mass_delta", "min"),
        teacher_mass_ratio_mean=("teacher_mass_ratio", "mean"),
        teacher_mass_ratio_p1=("teacher_mass_ratio", lambda x: x.quantile(0.01)),
        teacher_mass_ratio_p5=("teacher_mass_ratio", lambda x: x.quantile(0.05)),
        teacher_mass_ratio_median=("teacher_mass_ratio", "median"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse dense teacher captures for mass validation")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--trace-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    specs = selection["policies"]
    scales = {int(key): float(value) for key, value in selection["gamma_scale"].items()}
    k, block_size = int(selection["k"]), int(selection["block_size"])
    paths = sorted({path for root in args.trace_roots for path in root.rglob("*.npz")})
    payloads = [(str(path), specs, scales, k, block_size) for path in paths]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pieces = list(executor.map(worker, payloads))
    rows = pd.concat(pieces, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output / "teacher_mass.csv", index=False)
    summary = summarize(rows)
    summary.to_csv(args.output / "teacher_mass_summary.csv", index=False)
    rows.groupby(
        ["policy", "selection_role", "layer", "base_context_length", "workload"],
        as_index=False,
    ).agg(
        teacher_mass_delta_mean=("teacher_mass_delta", "mean"),
        teacher_mass_delta_p5=("teacher_mass_delta", lambda x: x.quantile(0.05)),
        teacher_mass_ratio_mean=("teacher_mass_ratio", "mean"),
        teacher_mass_ratio_p5=("teacher_mass_ratio", lambda x: x.quantile(0.05)),
    ).to_csv(args.output / "teacher_mass_breakdown.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
