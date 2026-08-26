# L40S DSA measurement lock

Final verdict: **PROTOCOL-SENSITIVE-BUT-EXPLAINED**

This package locks the DeepSeek-V2-Lite research-sidecar DSA operator and existing quality evidence. It does not claim TensorRT-LLM integration or H100/SM120 silicon results.

## Environment and scope

- GPU: NVIDIA L40S ×4 physical; measurement isolated to physical GPU 0. GPUs 2/3 were protected because they had existing jobs.
- Measurement runtime: Python 3.10.12, torch 2.11.0+cu130, CUDA runtime 13.0, Triton 3.6.0, driver 580.173.02.
- Prior locked/runtime materialization environment: torch 2.9.1+cu128, CUDA 12.8.
- Authoritative local source base: commit `130148b4fe6e9a601a29cb40a8485bca77503158`, branch `exp/l40s-dsa-lock`; measurement host is a non-git source snapshot.
- `ncu` is unavailable. Nsight Systems 2025.3.2 was used; DRAM/L2/occupancy counters are therefore not invented.

Complete trace roots, result directories, kernel paths and Nsight report paths are in `environment.json`; exact prior artifact roles and hashes are in `prior_artifact_inventory.csv`.

## Locked shapes

Primary: B=1, Q=1, H=64, D=128, Top-K=2048, N=8K/16K/32K, BF16 Q/K. The real replay lengths include the decode offset (for example 8K near-threshold is 8,285 tokens). Synthetic 64K and 128K operator-only tests also passed without model inference; see `synthetic_shape_audit.json`.

## New fixed-address timing lock

CUDA Events, preallocated explicit outputs, fixed Q/K/w addresses, 200 warmups and 2,000 measurements were used. Flush is a preallocated 64 MiB device write outside the timed event. Warm-cache and cache-flush are not averaged.

| Method | 8K warm / flush µs | 16K warm / flush µs | 32K warm / flush µs |
|---|---:|---:|---:|
| L0 Full64 score | 26.688 / 15.360 | 27.616 / 14.336 | 26.624 / 15.360 |
| L1 precomputed score Top-K | 46.080 / 46.080 | 72.704 / 73.728 | 83.968 / 70.656 |
| L2 Full64 + Top-K | 48.128 / 48.128 | 61.440 / 61.440 | 112.640 / 100.480 |
| L3 dynamic H8 score | 28.672 / 17.408 | 28.672 / 16.544 | 29.440 / 17.280 |
| L4 two-pass progressive H8 | 66.560 / 56.320 | 68.608 / 57.344 | 67.552 / 57.120 |
| L5 fused progressive dense | 32.672 / 21.504 | 32.800 / 20.608 | 32.768 / 21.504 |
| L6 fused progressive compact | 42.880 / 29.888 | 42.848 / 29.888 | 43.008 / 30.560 |
| L7 fused progressive + Top-K | 55.296 / 49.152 | 63.488 / 63.488 | 118.880 / 107.472 |
| L8 T1 H8 + previous Full64 | 50.208 / 42.496 | 50.976 / 40.864 | 49.152 / 40.736 |

All mean/median/p5/p95/min/std samples, promotion rates and pointer-lock fields are in `latency_protocols.csv`.

## Prior audit references versus new replay

| Path | 8K prior → new flush µs | 16K prior → new flush µs | 32K prior → new flush µs |
|---|---:|---:|---:|
| Full64 + Top-K | 40.784 → 48.128 | 63.488 → 61.440 | 87.040 → 100.480 |
| fused progressive + Top-K | 62.464 → 49.152 | 70.656 → 63.488 | 104.608 → 107.472 |

The differences are explained, not normalized away:

1. The prior timing used torch 2.9.1/CUDA 12.8, 50/500, three rotating traces, stock Top-K output allocation, and a global-budget performance mask.
2. The new timing uses torch 2.11/CUDA 13, 200/2,000, one fixed real replay address, preallocated Top-K outputs, and the validation-fixed production threshold.
3. The flush write also raises the L40S clock state. Score-only kernels are faster under the nominal cache-flush protocol than warm-cache, so this protocol is not a pure cache-state intervention.
4. The near-threshold observations intentionally expose numerical sensitivity: reference/runtime promotion rates are 20.0/40.0% at 8K, 46.34/46.34% at 16K, and 1.047/1.047% at 32K. Aggregate quality remains the separately locked 29.36% actual promotion rate.

## Nsight Systems lock

At 16K the current path remains two GPU operations: `_full_score_kernel` and `gatherTopK`. At 32K the prior 18-operation path is **not** reproduced: the current torch 2.11/CUDA 13 path has 22 operations. It contains one score kernel, one fill, two memsets, four radix digit-histogram stages, four radix digit-cumulative-sum stages, four within-K count stages, one Kth-count stage, two scan initializers, two scans, and one gather. These are selection stages; they are not all called sorting kernels. Per-operation durations and launch gaps are in `kernel_inventory.csv` and the raw 100-iteration reports are under `nsight/`.

## Quality lock (no new expensive sweep)

The primary runtime-legal dynamic-H8 fixed-threshold result is: actual promotion 29.3597%, mean/median QK reduction 15.8913%/13.9827%, Top-128/512/2048 recall 99.9980%/99.9702%/99.5875%, newly-active token recall 54.9154%, MLA RelL2 p95/p99 2.6129%/9.1845%, logit KL 0.0024889, and teacher-forced PPL delta -0.0044188. Existing closed-loop agreement is 50.2083%, first divergence 6; NIAH success stayed 100% for both baseline and approximate paths.

The prior H8 + 10% H56 result is preserved separately (20.1252% QK reduction, Top-2048 99.5555%, MLA RelL2 p95 2.5958%, logit KL 0.0025926). T1 without H56 has ranking evidence (87.5% QK reduction, Top-128/512/2048 99.9402%/99.3853%/82.0076%) but still has no MLA/KL/PPL result. See `quality_lock.csv`.

## Portable replay contract

Three `.pt` files contain ten real observations across easy, newly-active, near-threshold, and 32K-tail categories. Each stores BF16 Q/K, FP32 w, stable dynamic H8 IDs, token- and B64-block temporal masks, promotion blocks/threshold, Full64/H8/T1 scores, previous Full64 and stable exact Top-2048 IDs. Source trace, capture, cache and checkpoint hashes are retained, but runtime replay requires no checkpoint.

The common runner is `cross_platform_runner/benchmark_dsa_replay.py` and supports `full64`, `topk_only`, `full64_topk`, `h8`, `progressive_h8`, `fused_dense`, `fused_compact`, `fused_topk`, and `t1`, with JSON and CSV output and no hard-coded platform path.

Example:

```bash
python benchmark_dsa_replay.py --bundle <bundle.pt> --method full64 --top-k 2048 --warmup 200 --iters 2000
```

Locked Full64 baseline: L2 prior 40.784/63.488/87.040 µs; new flush 48.128/61.440/100.480 µs
Locked Top-K baseline: L1 new flush 46.080/73.728/70.656 µs
Locked progressive H8 result: L7 prior 62.464/70.656/104.608 µs; new flush 49.152/63.488/107.472 µs
Locked quality result: fixed-threshold H8 mean QK reduction 15.8913%, Top-2048 recall 99.5875%, MLA RelL2 p95 2.6129%
Main protocol sensitivity: runtime/Top-K path, GPU clock effect of flush, fixed-address versus rotating input, and production threshold versus global budget
Portable replay bundle count: 3 files / 10 real observations
Ready for SM120: Yes for portable replay and compile-time validation; silicon measurement pending
Ready for H100: Yes for the common CUDA-event replay protocol; silicon measurement pending
Remaining missing artifact: T1 no-H56 MLA/KL/PPL replay and Nsight Compute DRAM/L2/SM counters
