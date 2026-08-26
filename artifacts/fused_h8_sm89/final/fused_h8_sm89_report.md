# L40S SM89 Fused Progressive H8 DSA CUDA Kernel Pilot

## Verdict — `NO-GO`

The single-load fused dataflow is real and removes the two-pass K reread, but the deployable online `cp.async` persistent kernel does **not** beat the fastest optimized Full64 kernel.  The requested software GO gate is therefore not met.

### First-page numbers

| Metric | 16K | 32K |
|---|---:|---:|
| K1 Full64 sync, cold | 25.600 µs | 33.792 µs |
| K2 Full64 async pipeline, cold | 29.696 µs | 45.056 µs |
| K3 two-pass precomputed, cold | 30.720 µs | 38.912 µs |
| K4 fused precomputed mask, cold | 27.648 µs | 35.840 µs |
| K6 fused online async, cold | 30.720 µs | 38.912 µs |
| K4 speedup vs K3 | 1.111× | 1.086× |
| K6 speedup vs K2 | 0.967× | 1.158× |
| K6 speedup vs fastest Full64 | 0.833× | 0.868× |

At 32K, measured DRAM read fell from **15.33 MB (K3)** to **9.29 MB (K4)**, a **39.4%** reduction.  This proves that fused same-tile continuation removed the physical reread.  It is a `DATAFLOW-ONLY / PRECOMPUTED-MASK` result, not a deployable policy claim.

K6 uses **255 registers/thread**, **33.2 KiB shared/CTA**, reaches only **16.5%** achieved occupancy and **4.5%** Tensor-pipe activity.  Its DRAM read is 9.45 MB, essentially the same one-load traffic as K2, so the loss is dominated by register pressure, synchronization and under-utilization rather than an unremoved K reread.

### Quality operating points

| Policy | Actual cold promotion | Net QK reduction | MLA RelL2 p95 | Top-128 recall | PPL delta |
|---|---:|---:|---:|---:|---:|
| P0 global top-10%, precomputed | 10% budget | 20.72% | 2.60% | 99.9966% | -0.495% |
| P1 validation-fixed local threshold | 29.36% | 13.98% | 2.61% | 99.9980% | -0.442% |

P1 maintained local answer-task success at 58.3%, matching the prior baseline aggregate, but long-code token accuracy remained a weak point.  These are locked prior replay results for the identical P0/P1 policy; this run revalidated mathematical kernel equivalence on actual sidecar traces (45/45 CUDA correctness rows passed).

## Required questions

1. **Was H8 mapped without padding?** Yes. Each compute warp executes exact `(16×128)·(128×8)` tiles with `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`; four warps cover B64.
2. **Did SASS generate Tensor Core instructions?** Yes. `cuobjdump` found 1,024 `HMMA` instructions in the extension; the correctness maximum absolute error stayed below 0.002.
3. **Was the K reread removed?** Yes for K4/K6. The 32K K3→K4 DRAM-read reduction was 39.4%.
4. **Did `cp.async` overlap help?** `LDGSTS` (72 SASS instructions) and double buffers are present, but the measured pipeline did not create a latency win. K6's resource/synchronization cost outweighed overlap.
5. **Was the same pipeline applied to Full64?** Yes. K2 uses the identical async copy primitive, 32 KiB K double buffer and persistent scheduling. K1 is also disclosed because it was faster than K2 and is the honest effective baseline.
6. **Did MAC reduction become latency reduction?** No. P1 reduced QK by 13.98% yet K6 was 13.2% slower than fastest Full64 at 32K.
7. **Why not?** K6 is register/occupancy/synchronization-bound: 255 registers/thread, 16.5% achieved occupancy, 4.5% Tensor-pipe activity, plus CTA-wide verifier barriers and irregular continuation.
8. **P0 vs P1 quality?** P1 preserved the P0 quality scale but drifted from the nominal 10% rescue to 29.36%, reducing net QK savings from 20.72% to 13.98%.
9. **Did speedup survive Top-K?** Not tested by design: K6 failed the required 1.05× scoring gate. No Top-K or TPOT speedup is claimed.
10. **What direction is supported?** Keep the single-load fusion idea, but move to a compact K-sketch/traffic-pruning front end or a dedicated progressive datapath. This particular SM89 persistent software kernel should not be integrated.

## Additional observations

- Dynamic Top-8 selection + packing alone costs 35.840 µs at 32K; fixed-head packing costs 26.624 µs. Prep makes the software case worse.
- Q-shared plus padded K stride was the best shared-layout ablation, but it does not change the K6 verdict.
- Hot-L2 results were measured separately; cold rotating traces and a 4×L2 flush were used for the verdict.
- 64K/128K were performance-only synthetic key repeats. Quality claims remain restricted to 8K/16K/32K research-sidecar traces.
- No TMA, WGMMA, thread-block clusters, SM90 PTX or Hopper-only kernels were used.
