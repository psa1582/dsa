from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

K = 2048
SCORE_BYTES = 4
INDEX_BYTES_GPU = 8
INDEX_BYTES_HW = 4
PAIR_BYTES_HW = SCORE_BYTES + INDEX_BYTES_HW
INDEXER_K_BYTES_PER_TOKEN = 128 * 2
MACS_PER_SCORE = 64 * 128
HEAP_DEPTH = int(math.log2(K))
CONTEXTS = (8192, 16384, 32768, 65536)


def load_gpu_baseline(progressive_root: Path) -> pd.DataFrame:
    timing = pd.read_csv(progressive_root / "bench" / "timing.csv")
    stage = pd.read_csv(progressive_root / "stages" / "stage_timing.csv")
    locked = timing[
        timing.method.eq("full_plus_topk")
        & timing.block_size.eq(64)
        & timing.promotion_target.eq(0.10)
        & timing.topk.eq(K)
    ][["context", "median_ms"]].drop_duplicates("context")
    score = timing[
        timing.method.eq("full_kernel")
        & timing.block_size.eq(64)
        & timing.promotion_target.eq(0.10)
        & timing.topk.eq(K)
    ][["context", "median_ms"]].drop_duplicates("context")
    topk = stage[
        stage.stage.eq("topk_full_vector") & stage.promotion_target.eq(0.10)
    ][["context", "median_ms"]].drop_duplicates("context")
    frame = locked.rename(columns={"median_ms": "combined_ms"}).merge(
        score.rename(columns={"median_ms": "score_ms"}), on="context"
    ).merge(topk.rename(columns={"median_ms": "topk_ms"}), on="context")
    frame["measurement_status"] = "LOCKED CUDA-EVENT MEASUREMENT"
    frame["gpu_ops"] = frame.context.map({8192: 2, 16384: 2, 32768: 18})
    # 64K is a transparent linear projection from the measured 32K isolated stages.
    source = frame[frame.context.eq(32768)].iloc[0]
    projected = {
        "context": 65536,
        "combined_ms": 2 * (source.score_ms + source.topk_ms),
        "score_ms": 2 * source.score_ms,
        "topk_ms": 2 * source.topk_ms,
        "measurement_status": "ANALYTICAL 2X PROJECTION FROM 32K; NO REAL 64K TRACE",
        "gpu_ops": np.nan,
    }
    frame = pd.concat([frame, pd.DataFrame([projected])], ignore_index=True)
    isolated = frame.score_ms + frame.topk_ms
    frame["score_fraction_of_isolated_sum"] = frame.score_ms / isolated
    frame["topk_fraction_of_isolated_sum"] = frame.topk_ms / isolated
    frame["score_fraction_of_combined"] = frame.score_ms / frame.combined_ms
    frame["topk_fraction_of_combined"] = frame.topk_ms / frame.combined_ms
    frame["nonadditivity_ratio"] = isolated / frame.combined_ms
    return frame


def current_repeat(root: Path) -> pd.DataFrame:
    frames = []
    for run in ["decomposition", "decomposition_repeat2"]:
        path = root / run / "full64_topk_decomposition_raw.csv"
        frame = pd.read_csv(path)
        frame["run"] = run
        frames.append(frame)
    values = pd.concat(frames, ignore_index=True)
    return (
        values.groupby(["context", "method"])
        .agg(
            repeat_count=("run", "nunique"),
            median_us_mean=("median_ms", lambda x: 1000 * x.mean()),
            median_us_min=("median_ms", lambda x: 1000 * x.min()),
            median_us_max=("median_ms", lambda x: 1000 * x.max()),
        )
        .reset_index()
    )


def architecture_a_sweep() -> pd.DataFrame:
    rows = []
    for context in CONTEXTS:
        for block in [32, 64, 128, 256]:
            chunks = math.ceil(context / block)
            log_b = int(math.log2(block))
            full_sort_comparators_per_chunk = block * log_b * (log_b + 1) // 4
            for requested_r in [8, 16, 32, 64]:
                retained = min(block, requested_r)
                exact = retained >= block
                candidate_count = min(context, chunks * retained)
                rows.append(
                    {
                        "context": context,
                        "chunk_size": block,
                        "requested_local_r": requested_r,
                        "effective_local_r": retained,
                        "worst_case_exact": exact,
                        "classification": "EXACT" if exact else "APPROXIMATE",
                        "candidate_count": candidate_count,
                        "candidate_fraction": candidate_count / context,
                        "local_buffer_bytes": block * PAIR_BYTES_HW,
                        "full_bitonic_sort_comparator_ops": chunks
                        * full_sort_comparators_per_chunk,
                        "exactness_reason": (
                            "r>=B retains every item in each chunk"
                            if exact
                            else "a chunk may contribute more than r members of global Top-2048"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def data_movement(gpu: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in gpu.itertuples():
        context = int(item.context)
        topk_upper_passes = 1 if context <= 16384 else 9
        dense_write = context * SCORE_BYTES
        dense_read_lower = context * SCORE_BYTES
        dense_read_upper = context * SCORE_BYTES * topk_upper_passes
        gpu_result = K * (SCORE_BYTES + INDEX_BYTES_GPU)
        hw_result = K * INDEX_BYTES_HW
        k_bytes = context * INDEXER_K_BYTES_PER_TOKEN
        for system in ["GPU dense score + stock Top-K", "Fused streaming exact Top-K"]:
            if system.startswith("GPU"):
                offchip_lower = k_bytes + dense_write + dense_read_lower + gpu_result
                offchip_upper = k_bytes + dense_write + dense_read_upper + gpu_result
                score_materialization = dense_write
                selection_read_lower = dense_read_lower
                selection_read_upper = dense_read_upper
                result_bytes = gpu_result
            else:
                offchip_lower = offchip_upper = k_bytes + hw_result
                score_materialization = 0
                selection_read_lower = selection_read_upper = 0
                result_bytes = hw_result
            rows.append(
                {
                    "context": context,
                    "system": system,
                    "indexer_k_read_bytes": k_bytes,
                    "dense_score_write_bytes": score_materialization,
                    "selection_score_read_bytes_lower": selection_read_lower,
                    "selection_score_read_bytes_upper": selection_read_upper,
                    "full_or_partial_score_passes_lower": 1 if system.startswith("GPU") else 0,
                    "full_or_partial_score_passes_signature_upper": (
                        topk_upper_passes if system.startswith("GPU") else 0
                    ),
                    "final_result_bytes": result_bytes,
                    "total_offchip_bytes_lower": offchip_lower,
                    "total_offchip_bytes_upper": offchip_upper,
                    "traffic_status": "CODE-LEVEL / ANALYTICAL ESTIMATE",
                    "intermediate_note": (
                        "Top-K workspace traffic not counter-measured; 32K upper bound counts four radix, four within-K count, and one gather score pass"
                        if system.startswith("GPU")
                        else "candidate heap/FIFO remains on chip"
                    ),
                }
            )
    return pd.DataFrame(rows)


def heap_lookup(
    heap_summary: pd.DataFrame,
    fifo: pd.DataFrame,
    context: int,
    mode: str,
    p: int,
    lanes: int,
) -> tuple[float, float, str]:
    if context <= 32768:
        row = heap_summary[heap_summary.context.eq(context)].iloc[0]
        admissions = float(row[f"{mode}_admissions_mean"])
        match = fifo[
            fifo.context.eq(context)
            & fifo.start_mode.eq(mode)
            & fifo.scores_per_cycle.eq(p)
            & fifo.admission_lanes.eq(lanes)
        ]
        final_p99 = float(match.final_fifo_depth_p99.iloc[0])
        return admissions, final_p99, "TRACE-MEASURED ADMISSION/FIFO"
    source = heap_summary[heap_summary.context.eq(32768)].iloc[0]
    admissions = float(source[f"{mode}_admissions_mean"]) * context / 32768
    return admissions, np.nan, "ANALYTICAL RATE EXTRAPOLATION FROM 32K"


def performance_sweep(
    gpu: pd.DataFrame, heap_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    heap = pd.read_csv(heap_root / "exact_heap_admission_summary.csv")
    fifo = pd.read_csv(heap_root / "exact_heap_fifo_summary.csv")
    gpu_latency = dict(zip(gpu.context.astype(int), gpu.combined_ms * 1000))
    gpu_topk_latency = dict(zip(gpu.context.astype(int), gpu.topk_ms * 1000))
    rows = []
    for context in CONTEXTS:
        for mode in ["cold", "warm"]:
            for p in [1, 2, 4, 8, 16]:
                for requested_lanes in [1, 2, 4]:
                    for banks in [2, 4, 8]:
                        effective_lanes = min(requested_lanes, max(1, banks // 2))
                        admissions, final_fifo_p99, source = heap_lookup(
                            heap, fifo, context, mode, p, effective_lanes
                        )
                        for frequency_mhz in [250, 400, 500, 1000]:
                            for bandwidth_gbps in [256, 512, 1024, 2048]:
                                compute_cycles = math.ceil(context / p)
                                memory_us = (
                                    context * INDEXER_K_BYTES_PER_TOKEN / (bandwidth_gbps * 1e3)
                                )
                                memory_cycles = math.ceil(memory_us * frequency_mhz)
                                admission_cycles = math.ceil(admissions / effective_lanes)
                                stream_cycles = max(compute_cycles, memory_cycles, admission_cycles)
                                if math.isnan(final_fifo_p99):
                                    drain_cycles = HEAP_DEPTH
                                else:
                                    drain_cycles = math.ceil(final_fifo_p99 / effective_lanes) + HEAP_DEPTH
                                total_cycles = stream_cycles + drain_cycles
                                latency_us = total_cycles / frequency_mhz
                                selector_cycles = max(compute_cycles, admission_cycles) + drain_cycles
                                selector_latency_us = selector_cycles / frequency_mhz
                                fifo_entries = (
                                    math.ceil(final_fifo_p99)
                                    if not math.isnan(final_fifo_p99)
                                    else max(256, math.ceil(admissions * 0.05))
                                )
                                candidate_sram = 2 * K * PAIR_BYTES_HW
                                previous_state = K * INDEX_BYTES_HW if mode == "warm" else 0
                                fifo_bytes = max(2048, fifo_entries * PAIR_BYTES_HW)
                                pipeline_bytes = effective_lanes * HEAP_DEPTH * PAIR_BYTES_HW * 2
                                total_sram = candidate_sram + previous_state + fifo_bytes + pipeline_bytes
                                rows.append(
                                    {
                                        "context": context,
                                        "mode": mode,
                                        "scores_per_cycle": p,
                                        "requested_admission_lanes": requested_lanes,
                                        "effective_admission_lanes": effective_lanes,
                                        "candidate_sram_banks": banks,
                                        "frequency_mhz": frequency_mhz,
                                        "memory_bandwidth_gbps": bandwidth_gbps,
                                        "mean_heap_admissions": admissions,
                                        "mean_candidate_admissions_per_cycle": admissions
                                        / max(1, compute_cycles),
                                        "admission_source": source,
                                        "score_compute_cycles": compute_cycles,
                                        "k_memory_cycles": memory_cycles,
                                        "heap_update_cycles": admission_cycles,
                                        "drain_cycles": drain_cycles,
                                        "total_cycles": total_cycles,
                                        "latency_us": latency_us,
                                        "speedup_vs_gpu": gpu_latency[context] / latency_us,
                                        "selector_latency_us": selector_latency_us,
                                        "topk_speedup_vs_gpu": gpu_topk_latency[context]
                                        / selector_latency_us,
                                        "stall_percentage": max(
                                            0.0, admission_cycles - compute_cycles
                                        )
                                        / max(1, stream_cycles),
                                        "sustained_scores_per_cycle": min(
                                            p,
                                            context / max(memory_cycles, compute_cycles),
                                        ),
                                        "threshold_comparators": p,
                                        "heap_pipeline_comparators": effective_lanes * HEAP_DEPTH,
                                        "total_comparator_units": p
                                        + effective_lanes * HEAP_DEPTH,
                                        "full64_bf16_mac_lanes": MACS_PER_SCORE * p,
                                        "dsp58_fraction_if_1_bf16_mac_per_dsp": (
                                            MACS_PER_SCORE * p / 14352
                                        ),
                                        "onchip_sram_bytes": total_sram,
                                        "required_fifo_entries_p99": fifo_entries,
                                        "pipeline_depth_cycles": HEAP_DEPTH,
                                        "offchip_indexer_k_bytes": context
                                        * INDEXER_K_BYTES_PER_TOKEN,
                                        "offchip_dense_score_bytes": 0,
                                        "exact": True,
                                        "model_limit": "one-admission/cycle/lane pipelined heap and SRAM hazard handling require RTL-level validation",
                                    }
                                )
    sweep = pd.DataFrame(rows)
    selections = [
        ("FPGA conservative HW-A", "cold", 1, 1, 2, 400, 256),
        ("FPGA conservative HW-B", "warm", 1, 1, 2, 400, 256),
        ("FPGA optimistic HW-B", "warm", 2, 2, 4, 400, 512),
        ("ASIC HW-A", "cold", 2, 2, 4, 1000, 1024),
        ("ASIC HW-B", "warm", 2, 2, 4, 1000, 1024),
    ]
    selected_rows = []
    for name, mode, p, lanes, banks, frequency, bandwidth in selections:
        picked = sweep[
            sweep["mode"].eq(mode)
            & sweep.scores_per_cycle.eq(p)
            & sweep.requested_admission_lanes.eq(lanes)
            & sweep.candidate_sram_banks.eq(banks)
            & sweep.frequency_mhz.eq(frequency)
            & sweep.memory_bandwidth_gbps.eq(bandwidth)
        ].copy()
        picked.insert(0, "system", name)
        selected_rows.append(picked)
    comparison = pd.concat(selected_rows, ignore_index=True)
    return sweep, comparison


def storage_cost(performance: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "component": "Top-2048 FP32 score + INT32 index",
            "instances": 1,
            "bytes": K * 8,
            "note": "theoretical minimum candidate state",
        },
        {
            "component": "Top-2048 FP16/BF16 score + INT32 index",
            "instances": 1,
            "bytes": K * 6,
            "note": "reduced-score option; ordering equivalence requires validation",
        },
        {
            "component": "Double-buffered FP32 candidate heap",
            "instances": 2,
            "bytes": K * 8 * 2,
            "note": "selected exact model",
        },
        {
            "component": "Previous Top-2048 IDs",
            "instances": 1,
            "bytes": K * 4,
            "note": "HW-B warm-start metadata",
        },
        {
            "component": "Previous Kth threshold",
            "instances": 1,
            "bytes": 4,
            "note": "ordering hint only; never used as an exact discard certificate",
        },
    ]
    selected = performance[
        performance.system.eq("FPGA conservative HW-B") & performance.context.eq(32768)
    ].iloc[0]
    rows.append(
        {
            "component": "Selected FPGA total selector SRAM",
            "instances": 1,
            "bytes": int(selected.onchip_sram_bytes),
            "note": "double buffer + previous IDs + FIFO + pipeline scratch",
        }
    )
    for context in CONTEXTS:
        rows.append(
            {
                "component": f"One-layer BF16 indexer K at {context // 1024}K",
                "instances": 1,
                "bytes": context * INDEXER_K_BYTES_PER_TOKEN,
                "note": "optional on-chip cache; primary model permits HBM streaming",
            }
        )
    return pd.DataFrame(rows)


def pcie_model(gpu: pd.DataFrame) -> pd.DataFrame:
    topk_gpu = dict(zip(gpu.context.astype(int), gpu.topk_ms * 1000))
    rows = []
    for context in CONTEXTS:
        input_bytes = context * SCORE_BYTES
        output_bytes = K * INDEX_BYTES_HW
        for generation, bandwidth in [("Gen4 x16", 25), ("Gen5 x16", 50)]:
            for fixed_roundtrip_us in [20, 40, 80]:
                transfer_us = (input_bytes + output_bytes) / (bandwidth * 1e3)
                selector_us = context / (16 * 400)
                total_us = fixed_roundtrip_us + transfer_us + selector_us
                rows.append(
                    {
                        "context": context,
                        "pcie": generation,
                        "effective_payload_bandwidth_gbps": bandwidth,
                        "fixed_roundtrip_latency_us": fixed_roundtrip_us,
                        "gpu_to_fpga_score_bytes": input_bytes,
                        "fpga_to_gpu_index_bytes": output_bytes,
                        "payload_transfer_us": transfer_us,
                        "fpga_selector_us_at_16_scores_cycle_400mhz": selector_us,
                        "modeled_total_us": total_us,
                        "gpu_topk_us": topk_gpu[context],
                        "topk_speedup": topk_gpu[context] / total_us,
                        "go_at_1_15x": topk_gpu[context] / total_us >= 1.15,
                        "verdict": "NO-GO",
                        "reason": "not consistently beneficial across contexts and preserves dense score materialization plus two synchronization boundaries",
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSA exact streaming Top-K hardware model")
    parser.add_argument("--progressive-root", type=Path, required=True)
    parser.add_argument("--decomposition-root", type=Path, required=True)
    parser.add_argument("--heap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = load_gpu_baseline(args.progressive_root)
    gpu.to_csv(args.output / "gpu_bottleneck_decomposition.csv", index=False)
    current_repeat(args.decomposition_root).to_csv(
        args.output / "gpu_decomposition_repeat_audit.csv", index=False
    )
    architecture_a_sweep().to_csv(args.output / "architecture_a_sweep.csv", index=False)
    traffic = data_movement(gpu)
    traffic.to_csv(args.output / "data_movement_analysis.csv", index=False)
    sweep, comparison = performance_sweep(gpu, args.heap_root)
    sweep.to_csv(args.output / "throughput_sensitivity.csv", index=False)
    comparison.to_csv(args.output / "hardware_performance_comparison.csv", index=False)
    storage_cost(comparison).to_csv(args.output / "storage_cost.csv", index=False)
    pcie_model(gpu).to_csv(args.output / "pcie_offload_analysis.csv", index=False)
    verdict = {
        "verdict": "PROMISING",
        "next_step": "BUILD MORE DETAILED CYCLE MODEL FIRST",
        "fpga_topk_only_offload": "NO-GO",
        "fused_fpga_indexer_topk": "NO-GO FOR RTL AT CONSERVATIVE 1-BF16-MAC/DSP MAPPING",
        "exact_asic_selector": "GO FOR FLOORPLAN/CYCLE MODEL; NOT RTL SIGN-OFF",
        "exact_primary": True,
        "approximate_h8_primary": False,
        "key_limits": [
            "64K GPU baseline is an analytical projection because no real 64K sidecar trace exists",
            "stock Top-K DRAM/L2 counters unavailable; traffic is code-level bounded",
            "pipelined heap SRAM hazards and comparator timing are not RTL-validated",
        ],
    }
    (args.output / "hardware_verdict.json").write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
