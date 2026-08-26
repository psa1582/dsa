#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/home/psa1582/temporal-dsa-pilot}
output=${2:-$repo/artifacts/fused_h8_sm89/run_20260826_sm89}
python=/home/psa1582/.venvs/l40-regime/bin/python
ncu=/opt/nvidia/nsight-compute/2025.3.1/ncu

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin
export PYTHONPATH=$repo/src

metrics=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sector_hit_rate.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio,gpu__time_duration.sum

for context in 16384 32768; do
  for method in K2 K3 K4 K6; do
    if [[ ! -f "$output/ncu_metrics_${method}_${context}.ncu-rep" ]]; then
      "$ncu" --target-processes all --nvtx \
        --nvtx-include "${method}_${context}_cold/" --metrics "$metrics" \
        -o "$output/ncu_metrics_${method}_${context}" \
        "$python" "$repo/scripts/profile_fused_h8_sm89.py" \
          --repo "$repo" --output "$output" --method "$method" \
          --context "$context" --iterations 1
    fi
    "$ncu" --import "$output/ncu_metrics_${method}_${context}.ncu-rep" \
      --csv --page raw > "$output/ncu_metrics_${method}_${context}_raw.csv"
  done
done
