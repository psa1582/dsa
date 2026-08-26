from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


CLOCK_HZ = 1_000_000_000
FULL_MACS_PER_CYCLE = 8192.0
FULL_KEY_BYTES_PER_TOKEN = 256.0


def median_rates(group: pd.DataFrame) -> dict[str, float]:
    length = group.context_length.astype(float)
    return {
        "temporal_exact": float((group.temporal_exact_tokens / length).median()),
        "cold": float((group.cold_tokens_scanned / length).median()),
        "rescue": float((group.rescue_exact_tokens / length).median()),
        "physical_bytes": float((1.0 - group.physical_b64_key_byte_reduction).median()),
        "sketch_bytes": float((group.sketch_bytes / length).median()),
        "metadata_bytes": float((group.metadata_bytes / length).median()),
        "rescue_blocks": float((group.rescue_blocks / length).median()),
    }


def simulate(
    *,
    context: int,
    path: str,
    width: int,
    rates: dict[str, float],
    bandwidth_gbps: int,
    sketch_memory: str,
    compute_ratio: int,
) -> dict[str, float]:
    bytes_per_cycle = bandwidth_gbps
    full_bytes = context * FULL_KEY_BYTES_PER_TOKEN
    full_macs = context * 64 * 128
    full_memory_cycles = full_bytes / bytes_per_cycle
    full_compute_cycles = full_macs / FULL_MACS_PER_CYCLE
    topk_cycles = 64.0
    baseline_cycles = max(full_memory_cycles, full_compute_cycles) + topk_cycles

    temporal_tokens = context * rates["temporal_exact"]
    cold_tokens = context * rates["cold"]
    rescue_tokens = context * rates["rescue"]
    exact_tokens = temporal_tokens + rescue_tokens
    verifier_macs = cold_tokens * width * (128 if path == "head" else 64)
    exact_macs = exact_tokens * 64 * 128
    query_transform_macs = 0.0 if path == "head" else 64 * 128 * 7
    verifier_cycles = (verifier_macs + query_transform_macs) / (
        FULL_MACS_PER_CYCLE * compute_ratio
    )
    rerank_cycles = exact_macs / FULL_MACS_PER_CYCLE

    sketch_bytes = context * rates["sketch_bytes"]
    metadata_bytes = context * rates["metadata_bytes"]
    physical_bytes = context * FULL_KEY_BYTES_PER_TOKEN * rates["physical_bytes"]
    full_key_bytes = max(0.0, physical_bytes - sketch_bytes - metadata_bytes)
    if path == "head" or sketch_memory == "shared_hbm":
        memory_cycles = physical_bytes / bytes_per_cycle
    elif sketch_memory == "separate_sram":
        memory_cycles = max(full_key_bytes / bytes_per_cycle, sketch_bytes / 4096.0)
    elif sketch_memory == "narrow_hbm":
        memory_cycles = max(full_key_bytes / bytes_per_cycle, sketch_bytes / 512.0)
    else:
        raise ValueError(f"unknown sketch memory: {sketch_memory}")

    block_count = math.ceil(context / 64)
    selector_cycles = math.ceil(block_count / 16) + 8
    # Optimistic streaming architecture: key/sketch scan, verifier MACs, and
    # candidate full reranks overlap; selector and final Top-K drain afterward.
    total_cycles = max(memory_cycles, verifier_cycles, rerank_cycles) + selector_cycles + topk_cycles
    mac_ratio = (verifier_macs + query_transform_macs + exact_macs) / full_macs
    byte_ratio = physical_bytes / full_bytes
    candidate_sram = rescue_tokens * FULL_KEY_BYTES_PER_TOKEN
    sketch_sram = sketch_bytes if path == "dim" and sketch_memory == "separate_sram" else 0.0
    return {
        "cycles_per_decode_step": total_cycles,
        "full_dsa_cycles_per_decode_step": baseline_cycles,
        "speedup_over_full_dsa": baseline_cycles / total_cycles,
        "effective_hbm_utilization": min(1.0, full_key_bytes / max(total_cycles * bytes_per_cycle, 1.0)),
        "verifier_utilization": min(1.0, verifier_cycles / total_cycles),
        "full_rerank_utilization": min(1.0, rerank_cycles / total_cycles),
        "rescue_fifo_occupancy_tokens": rescue_tokens,
        "physical_key_and_sketch_bytes": physical_bytes,
        "full_key_bytes": full_key_bytes,
        "sketch_bytes": sketch_bytes,
        "metadata_bytes": metadata_bytes,
        "candidate_sram_bytes": candidate_sram,
        "sketch_sram_bytes": sketch_sram,
        "total_sram_bytes": candidate_sram + sketch_sram,
        "normalized_energy_proxy": 0.8 * byte_ratio + 0.2 * mac_ratio,
        "net_qk_reduction_model": 1.0 - mac_ratio,
        "physical_byte_reduction_model": 1.0 - byte_ratio,
        "selector_cycles": selector_cycles,
        "model_is_optimistic_streaming": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analytical cycle model for two-path verifier")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", default=["Balanced", "Aggressive"])
    args = parser.parse_args()
    frame = pd.read_csv(args.rows)
    frame = frame[frame.policy_role.isin(args.roles)].copy()
    contexts = [8192, 16384, 32768, 65536, 131072]
    rows = []
    for (role, verifier, path, width, rescue_fraction), method in frame.groupby(
        ["policy_role", "verifier", "path", "width", "rescue_fraction"], sort=False
    ):
        available = sorted(method.base_context_length.unique())
        for context in contexts:
            source_context = context if context in available else max(available)
            rates = median_rates(method[method.base_context_length == source_context])
            memory_options = ["full_k_hbm"] if path == "head" else [
                "shared_hbm", "separate_sram", "narrow_hbm"
            ]
            for bandwidth in [256, 512, 1000, 2000]:
                for sketch_memory in memory_options:
                    normalized_memory = "shared_hbm" if path == "head" else sketch_memory
                    for compute_ratio in [1, 2, 4, 8]:
                        for rescue_scale in [0.5, 1.0, 1.5, 2.0]:
                            sensitivity_rates = dict(rates)
                            rescue_delta = rates["rescue"] * (rescue_scale - 1.0)
                            sensitivity_rates["rescue"] = max(
                                0.0, rates["rescue"] * rescue_scale
                            )
                            sensitivity_rates["physical_bytes"] = max(
                                0.0,
                                rates["physical_bytes"]
                                + rescue_delta,
                            )
                            result = simulate(
                                context=context,
                                path=str(path),
                                width=int(width),
                                rates=sensitivity_rates,
                                bandwidth_gbps=bandwidth,
                                sketch_memory=normalized_memory,
                                compute_ratio=compute_ratio,
                            )
                            result.update(
                                {
                                    "policy_role": role,
                                    "verifier": verifier,
                                    "path": path,
                                    "width": int(width),
                                    "measured_rescue_fraction": float(rescue_fraction),
                                    "rescue_rate_scale": rescue_scale,
                                    "modeled_rescue_fraction": float(rescue_fraction) * rescue_scale,
                                    "context_length": context,
                                    "rate_source_context": source_context,
                                    "full_k_bandwidth_gbps": bandwidth,
                                    "sketch_memory": sketch_memory,
                                    "verifier_compute_ratio": compute_ratio,
                                    "clock_hz": CLOCK_HZ,
                                    "full_engine_macs_per_cycle": FULL_MACS_PER_CYCLE,
                                }
                            )
                            rows.append(result)
    output = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "cycle_sim_results.csv", index=False)
    summary = output[output.rescue_rate_scale == 1.0].groupby(
        ["policy_role", "verifier", "path", "sketch_memory"], as_index=False
    ).agg(
        speedup_min=("speedup_over_full_dsa", "min"),
        speedup_median=("speedup_over_full_dsa", "median"),
        speedup_max=("speedup_over_full_dsa", "max"),
        max_sram_bytes=("total_sram_bytes", "max"),
        energy_proxy_median=("normalized_energy_proxy", "median"),
    )
    summary.to_csv(args.output / "cycle_sim_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
