# Temporal Progressive DSA Software Feasibility Pilot — L40S ×2

## Verdict

**ALGORITHM-PROMISING-BUT-SOFTWARE-SLOW**

단일-load progressive H8는 기존 two-pass보다 빨라졌지만 동일 Triton framework의 optimized full indexer를 이기지 못했다. 주 설정 B64/r10/K2048에서 full 대비 speedup은 8K **0.653×**, 16K **0.899×**, 32K **0.832×**다. 32K에서 fused는 two-pass보다 **1.282×** 빠르지만 full보다 느리다.

## 첫 페이지 핵심 수치

- Optimized full indexer+TopK: 8K/16K/32K = 40.78/63.49/87.04 µs
- H8 two-pass: 89.09/84.99/134.14 µs
- Fused progressive dense: 62.46/70.66/104.61 µs
- Nsight GPU ops/iteration (16K): full 2, two-pass 5, fused dense 2, compact 3
- Nsight GPU ops/iteration (32K, multi-kernel Top-K): full 18, two-pass 21, fused dense 18, compact 19
- Held-out threshold actual promotion rate: 29.36% (nominal 10%)
- Nearest actual-rate r20 sweep speedup vs full (8K/16K/32K): 0.656×/0.984×/0.825×
- Actual QK reduction median: 13.98%
- Actual threshold MLA RelL2 p95: 2.613%
- Actual threshold Top-128 recall: 99.9980%
- Teacher-forced PPL delta: -0.442%; logit KL mean 0.002489
- Local RULER-like task success: baseline 58.3% vs threshold H8 58.3% (LongBench 미실행)
- K source load: full와 fused 모두 1회 scan. two-pass는 promoted block을 reread한다.
- DRAM/L2/SM/achieved occupancy: **ncu 미설치로 unavailable**. code-level bytes를 hardware counter처럼 주장하지 않는다.

## Performance

| Method | Context | QK reduction | GPU ops/iter | Indexer+TopK | Speedup vs full |
| --- | --- | --- | --- | --- | --- |
| Full optimized | 8K | 0.00% | 2 logical | 40.78 µs | 1.000× |
| H8 two-pass | 8K | 4.10% | 5 logical | 89.09 µs | 0.458× |
| Fused progressive | 8K | 4.10% | 2 logical | 62.46 µs | 0.653× |
| Fused compact | 8K | 4.10% | 3 logical | 70.62 µs | 0.577× |
| Full optimized | 16K | 0.00% | 2 | 63.49 µs | 1.000× |
| H8 two-pass | 16K | 23.24% | 5 | 84.99 µs | 0.747× |
| Fused progressive | 16K | 23.24% | 2 | 70.66 µs | 0.899× |
| Fused compact | 16K | 23.24% | 3 | 68.11 µs | 0.932× |
| Full optimized | 32K | 0.00% | 18 | 87.04 µs | 1.000× |
| H8 two-pass | 32K | 31.96% | 21 | 134.14 µs | 0.649× |
| Fused progressive | 32K | 31.96% | 18 | 104.61 µs | 0.832× |
| Fused compact | 32K | 31.96% | 19 | 118.75 µs | 0.733× |

가장 좋은 단발점은 16K/B32/r5/K512에서 full 대비 약 1.109×였지만, 주 K=2048과 32K에서 재현되지 않았다. 따라서 software GO 근거가 아니다. compact는 16K 한 점에서 dense보다 소폭 빠르지만 8K/32K에서 느리고 full baseline도 이기지 못했다.

## 왜 two-pass가 느리고 fused도 full을 못 이겼는가

Two-pass는 H8 scan, compare, logical-or mask, masked rerank, Top-K의 5 launch와 verifier intermediate write, promoted K reread를 가진다. Fused는 CTA 안에서 K tile을 한 번 load하고 cold H8 뒤 같은 tile로 remaining 56 heads를 이어 계산하므로 source dataflow상 reread와 global H8 intermediate를 제거했다. 그 결과 two-pass보다 빨라졌다. 그러나 full도 정확히 한 번의 순차 K scan이며 커널이 단순하다. fused의 uniform CTA branch와 두 계산 경로가 만드는 제어·instruction/resource 비용이 줄어든 MAC보다 컸다. 이것은 Nsight Systems timeline과 1-kernel CUDA-event latency로 확인되지만, ncu가 없어 register pressure와 achieved occupancy는 인과 추정으로만 남긴다.

Top-K는 32K full path에서 전체 latency의 73.1%다. H8-only full-prefix도 full보다 빠르지 않았고, atomic compact는 일관된 해결책이 아니었다.

## Quality

Global top-10% 결과를 threshold 결과로 재사용하지 않았다. validation 24 traces에서 layer별 cutoff를 고정하고 held-out 144 traces를 own-trajectory로 replay했다. dynamic/fixed 세트도 validation에서만 선택했다. promotion 5/10/15/20%의 MLA curve와 r10 teacher-forced/closed-loop를 별도로 측정했다.

고정 head가 dynamic과 비슷한 경우 routing launch를 제거할 수 있지만, 어떤 head 방식도 주 fused latency가 full보다 느리다는 software verdict를 바꾸지는 않는다. task 표는 local NIAH single/multi, variable tracking, aggregation, code next-token ground truth를 포함한다. 로컬 LongBench harness가 없어 LongBench는 미실행이며 production quality pass를 선언하지 않는다.

## Profiling 및 memory accounting

- Full: K read ≈ L×128×2 bytes, dense score write L×4 bytes.
- Fused: K read source load는 동일한 1회이며 K traffic reduction을 주장하지 않는다. rejected token도 full-dimensional K tile을 읽는다.
- Two-pass: cold H8 K scan 뒤 accepted full rerank로 promoted block K를 재접근한다.
- Nsight Systems: kernel launch, CUDA API, NVTX critical path 측정.
- Nsight Compute: 서버에 `ncu`가 없어 DRAM/L2 byte, SM utilization, tensor utilization, occupancy, branch efficiency를 측정하지 못했다.

## E2E sidecar 및 다음 단계

8개 selected layer에 isolated kernel delta를 합산하면 fused는 TPOT를 개선하지 않고 오히려 context별 수십~백여 µs를 더한다. Python reference controller의 decode 시간은 production TRT-LLM TPOT가 아니며, 이 결과를 V3.2 FP8 production claim으로 일반화하지 않는다.

다음 단계는 production V3.2 port가 아니다. 이 dataflow는 hardware/ISA co-design 근거로 보존하되, software 방향은 optimized full을 유지한다. D32→H8는 추가 launch와 sketch traffic을 도입하며 primary fused가 이미 full보다 느린 원인을 해결하지 못하므로 구현하지 않았다. 새로운 접근은 persistent multi-CTA Top-K fusion 또는 full kernel 내부에서 register footprint를 증가시키지 않는 predication이 먼저다.

## 질문별 답

1. QK 감소가 latency 감소로 변환됐는가? **아니다.**
2. K reread를 없앴는가? **source dataflow상 그렇다.** 실제 DRAM byte counter는 ncu 부재로 미측정이다.
3. fused가 two-pass보다 빠른가? **그렇다**, 32K 주 설정에서 1.282×.
4. Top-K가 병목인가? **그렇다**, 특히 32K full path의 대부분을 차지한다.
5. dense와 compact 중 무엇이 유리한가? **dense가 주 결과**다. compact는 일관되지 않다.
6. dynamic H8가 필요한가? fixed quality 표를 보면 판단 가능하지만 routing 절감도 full 대비 열세를 뒤집지 못한다.
7. threshold promotion 품질은 유지되는가? PPL/task/MLA 표에 실제 측정값을 제시했으며 global budget과 분리했다.
8. 16K/32K에서 speedup이 커지는가? **아니다.** 32K도 full 대비 1× 미만이다.
9. 실제 task quality는 유지되는가? local 측정 범위는 표에 제시하지만 LongBench 미실행 때문에 포괄적 유지 주장은 하지 않는다.
10. 다음 단계는? **software 방향 종료, 알고리즘은 hardware 연구 근거로 보존**한다.
