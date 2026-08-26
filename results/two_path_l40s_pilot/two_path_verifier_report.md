# Two-Path Cheap Verifier DSA Hardware Feasibility Pilot

> Scope: **Research sidecar DSA on DeepSeek-V2-Lite**. L40S ×2에서 측정한 연구용 sidecar 결과이며 DeepSeek-V3.2 production 결과가 아니다.

## Overall Verdict

**`HEAD-SPARSE-SW-PROMISING`**

Aggressive temporal + `8H×128D`, B64, cold-block 10% rescue는 전체 144 trace에서 Quality Band A의 offline MLA 기준과 teacher-forced PPL 기준을 통과하고 median net QK reduction 20.72%를 남겼다. 그러나 full K scan과 rescue reread 때문에 physical B64 key traffic은 -2.75%(음수는 증가)이며, analytical simulator의 speedup 범위는 0.959×–1.571×로 대역폭 전 구간에서 일관되지 않았다. 따라서 TensorRT-LLM/H100용 BLASST 계열 **software candidate**로는 후속 profiling 가치가 있지만 dedicated hardware/RTL GO는 아니다.

closed-loop에서는 NIAH 3/3을 baseline과 동일하게 맞혔지만 greedy token agreement가 30.86%이고 first divergence가 10 step이었다. 이 결과를 production 안정성으로 해석하면 안 된다.

## Case A — Head-Sparse verdict

- Selected: Aggressive + dynamic high-|w| H8 + B64 + 10% rescue.
- MLA RelL2 p95/p99: 2.60% / 8.02%.
- cosine p5: 0.999666; Top-128/512: 99.9966% / 99.9681%.
- newly-active token/block recall: 52.43% / 45.52%; focused top-4 tail-block recall: 19.45%.
- net QK reduction: 20.72%; physical B64 bytes: -2.75%.
- teacher-forced: PPL delta -0.495%, logit KL mean 0.002593, Top-1 agreement 98.96%.
- Dynamic head routing은 검증했지만 validation-fixed/transition-aware/tail-aware head가 비슷한지는 이번 축소 sweep으로 확정하지 못했다.

## Case B — Dimension-Sparse verdict

**`NO-GO`**. BF16 후보 네 개 모두 Band B를 통과하지 못했다. 품질 경계에 가장 가까운 Balanced D32는 RelL2 p95 5.04%로 5% 기준을 0.039%p 초과했고, net QK/physical reduction도 15.95% / 15.93%에 그쳤다. 같은 12.5% MAC의 Aggressive D16은 physical bytes를 24.12% 줄였지만 RelL2 p95가 6.99%였다. BF16 gate 실패로 INT8/INT4/INT2와 learned projection은 조기 종료했다.

## 핵심 결과

| Path / policy | Band | RelL2 p95 | cosine p5 | Top-128 | Top-512 | net QK | physical B64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aggressive H8×128 | A + PPL pass | 2.60% | 0.999666 | 99.9966% | 99.9681% | 20.72% | -2.75% |
| Aggressive 64H×D16 | C | 6.99% | 0.997570 | 99.9105% | 99.6638% | 24.20% | 24.12% |
| Balanced 64H×D32 | C, B miss 0.039%p | 5.04% | 0.998744 | 99.9558% | 99.8090% | 15.95% | 15.93% |

## Oracle rescue ceiling

전체 9,216 observation/policy에서 Balanced temporal-only p95 6.60%는 oracle best-1/2/4/8 block rescue 시 4.35% / 3.62% / 3.01% / 2.44%가 됐다. Aggressive는 10.10%에서 best-4 4.80%로 내려갔다. 따라서 aggregate에서는 상위 4개 block rescue만으로 Band B가 가능하다.

반면 32K layer 8/17/21 집중 subset에서는 Balanced best-8도 p95 8.30%, Aggressive best-8도 11.11%였다. aggregate 개선과 worst-tail 복구를 구분해야 하며, 현재 cheap detector는 이 집중 tail을 Band B까지 복구하지 못했다.

## 같은 MAC 예산: 8H×128D vs 64H×16D

Aggressive 기준 H8은 D16보다 newly-active/tail event와 actual output을 훨씬 잘 보존했다. D16은 full K 대신 compact sketch를 읽어 physical traffic을 줄였지만 Band C에 머물렀다. 즉 이 데이터에서는 “모든 head를 조금씩 보면 rare transition을 더 잘 잡는다”는 가설이 성립하지 않았다. Hadamard는 일부 rank recall을 개선했지만 Band B를 만들 만큼 충분하지 않았다.

## 반드시 답할 질문

1. **일부 head가 놓친 event:** 32K 집중 tail에서 H8의 top-4 tail-critical block recall은 19.45%에 불과했고 MLA p95가 Band C에 머물렀다.
2. **all-head dim-sparse가 완화했는가:** 아니오. 같은 MAC의 D16은 H8보다 newly-active recall과 MLA tail이 더 나빴다.
3. **동일 12.5% MAC 승자:** 품질은 H8, physical traffic은 D16. 주 Band B 기준 때문에 최종 승자는 H8 software path다.
4. **필요 oracle block 수:** aggregate는 best-4로 B 통과, 집중 32K tail은 best-8로도 부족했다.
5. **candidate precision:** H8은 full replay에서 rescue precision이 52.54%로 유용하지만 tail-critical recall은 제한적이다.
6. **rerank 포함 net QK:** Aggressive H8만 주 후보 기준 20.72%로 20%를 넘겼다.
7. **physical memory:** dimension-sparse가 명확히 유리하다. head-sparse는 full K scan 때문에 오히려 증가했다.
8. **Hadamard 필요성:** random/even subset보다 도움을 주는 구간은 있으나 품질 gate를 바꾸지는 못했다.
9. **fixed 설정:** 이번 결과로는 답할 수 없다. 동적 H8만 최종 모델 검증했고 fixed transition/tail-aware calibration은 후속 항목이다.
10. **다음 단계:** H8을 TensorRT-LLM/BLASST software kernel로 구현해 Nsight HBM traffic과 end-to-end latency를 먼저 측정한다. dimension detector를 개선하기 전 accelerator RTL로 가지 않는다.

## Hardware simulator 해석

1 GHz, full engine 8,192 MAC/cycle, scan/verifier/rerank optimistic overlap의 analytical model이다. 실측 CUDA kernel timing이 아니다. H8은 256 GB/s corner에서 0.959×로 느려지고 최고 1.571×였다. full-K scan이 남아 bandwidth-sensitive하며 `>=1.3× across multiple bandwidth settings`를 안정적으로 충족하지 않는다. Dimension path는 속도/energy proxy는 좋지만 품질 gate 실패로 hardware GO에 사용할 수 없다.

## Closed-loop

| Benchmark | Context | First divergence | Token agreement | Task result |
|---|---:|---:|---:|---|
| ruler_niah_small | 8192 | 12 | 12.50% | NIAH baseline=pass, verifier=pass |
| long_code_completion | 8192 | 66 | 51.56% | agreement proxy only |
| ruler_niah_small | 16384 | 49 | 37.50% | NIAH baseline=pass, verifier=pass |
| long_code_completion | 16384 | 15 | 22.66% | agreement proxy only |
| ruler_niah_small | 32768 | 10 | 7.81% | NIAH baseline=pass, verifier=pass |
| long_code_completion | 32768 | 68 | 53.12% | agreement proxy only |


long-code에는 외부 정답 기반 task score가 없으므로 generated-token agreement만 보고한다. teacher-forced PPL 통과가 closed-loop 안정성을 보장하지 않는다는 반례로 해석한다.

## 실험 범위와 제한

- Full evaluation: 144 held-out traces, 9,216 transitions/observations per policy, layers 2/5/8/11/14/17/21/25, contexts 8K/16K/32K.
- 32K 집중 sweep: dynamic high-|w|/positive-weight heads와 original/Hadamard random/even dimensions, widths H2/4/8/16 및 D4/8/16/32, rescue 1/2/5/10%.
- 구현하지 않은 대규모 확장: validation-fixed energy/transition/tail-aware/held-out oracle head·dimension set, H1/D64, B32/B128, 20%/token rescue, verifier-only baseline. 집중 tail 조기 결과에 따라 확장하지 않았으며 해당 비교를 완료한 것으로 주장하지 않는다.
- Missed-token 개별 rank는 최종 detail artifact에 보존하지 않아 histogram은 결측 사유를 표시한 placeholder다.
- BF16 dimension path가 Band B를 통과하지 않아 low-bit precision/learned projection을 실행하지 않았다.
- Deterministic Top-K의 cutoff tie-break를 global lower-index 우선으로 수정했다. 이전 approximate pilot 수치는 보존했고 이번 baseline/후보는 수정된 동일 규칙을 사용한다.

## Reproducibility

- Model revision: `85864749cd611b4353ce1decdb286193298f64c7`
- GPUs: NVIDIA L40S 48GB ×2, physical GPU 0/1 only; GPU 2/3 untouched.
- Tests: 22 passed.
- Main machine-readable artifacts: `two_path_verdict.json`, `oracle_rescue_ceiling.csv`, `head_sparse_results.csv`, `dim_sparse_results.csv`, `equal_cost_comparison.csv`, `tail_critical_recall.csv`, `selection_quality.csv`, `mla_output_quality.csv`, `teacher_forced_quality.csv`, `closed_loop_quality.csv`, `hardware_cost_model.csv`, `cycle_sim_results.csv`, `selected_configs.json`, `reproducibility.json`.
- Graphs: 22 files under `graphs_two_path/`.
