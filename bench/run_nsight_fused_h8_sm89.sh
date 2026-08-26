#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/home/psa1582/temporal-dsa-pilot}
output=${2:-$repo/artifacts/fused_h8_sm89/run_20260826_sm89}
python=/home/psa1582/.venvs/l40-regime/bin/python
ncu=/opt/nvidia/nsight-compute/2025.3.1/ncu
nsys=/usr/local/bin/nsys

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin
export PYTHONPATH=$repo/src

for context in 16384 32768; do
  for method in K2 K3 K4 K6; do
    if [[ -f "$output/ncu_${method}_${context}.ncu-rep" ]]; then
      continue
    fi
    "$ncu" \
      --target-processes all \
      --nvtx --nvtx-include "${method}_${context}_cold/" \
      --section SpeedOfLight \
      --section MemoryWorkloadAnalysis \
      --section Occupancy \
      --section LaunchStats \
      --section WarpStateStats \
      --section InstructionStats \
      -o "$output/ncu_${method}_${context}" \
      "$python" "$repo/scripts/profile_fused_h8_sm89.py" \
        --repo "$repo" --output "$output" --method "$method" \
        --context "$context" --iterations 1
    "$ncu" --import "$output/ncu_${method}_${context}.ncu-rep" \
      --csv --page raw > "$output/ncu_${method}_${context}_raw.csv"
  done
done

for context in 16384 32768; do
  for method in K2 K3 K4 K6; do
    "$nsys" profile --trace=cuda,nvtx,osrt --sample=none \
      --force-overwrite=true -o "$output/nsys_${method}_${context}" \
      "$python" "$repo/scripts/profile_fused_h8_sm89.py" \
        --repo "$repo" --output "$output" --method "$method" \
        --context "$context" --iterations 100
    "$nsys" stats --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum \
      --format csv --output "$output/nsys_${method}_${context}_stats" \
      "$output/nsys_${method}_${context}.nsys-rep"
  done
done
