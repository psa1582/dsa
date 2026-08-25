# Approximate Temporal DSA — L40S ×2 Quality and Hardware Feasibility Pilot

## Verdict

`NO-GO`

**Scope:** Research sidecar DSA on DeepSeek-V2-Lite. This is not a production DeepSeek-V3.2 FP8 Indexer result.

The Balanced point reduced held-out own-trajectory indexer QK by 28.46% median, but actual sparse MLA output relative-L2 p95 was 6.61%, above the pilot gate of 1%. The Safe point had 4.16% p95 output error and only 17.10% median reduction, below the 20% minimum. Therefore no candidate satisfies the pilot's quality-and-reduction decision rule.

## 핵심 숫자

- Safe / Balanced / Aggressive own median QK reduction: 17.10% / 28.46% / 41.00%
- Top-128 recall: 99.965% / 99.904% / 99.581%
- Top-512 recall: 99.842% / 99.643% / 98.807%
- Teacher attention mass mean ratio: 99.882% / 99.780% / 99.560%
- MLA output cosine median: 0.999996 / 0.999994 / 0.999982
- MLA output relative-L2 p95: 4.16% / 6.61% / 10.09%
- Teacher-forced logit KL mean: 0.003686 / 0.004310 / 0.005357
- Teacher-forced PPL delta: -0.03% / -0.47% / -0.25%
- Dense teacher → full-indexer sparse PPL delta: -0.24%
- Same-margin dynamic gain over static: 0.028 percentage point
- BF16 previous-TopK cache: 512 KiB/layer for K=2048, d=128
- Closed-loop/task/software approximate timing: skipped after the Phase-B NO-GO gate

## 최종 질문에 대한 답

1. **Exact match가 낮아도 output이 유지되는가?** Median은 유지되지만 tail은 아니다. Balanced cosine median은 0.999994지만 relative-L2 p95는 6.61%다.
2. **Miss가 cutoff에 집중되는가?** 대체로 그렇다. Safe Top-128 retention은 99.965%인 반면 Recall@2048은 99.034%다.
3. **High-score core가 안정적인가?** 평균적으로 매우 안정적이나 반복적인 rare miss가 있고 32K layer 17/21에서 큰 MLA output tail과 연결된다.
4. **25–35% reduction에서 model quality가 유지되는가?** Teacher-forced PPL은 유지됐지만 actual MLA output gate는 실패했다.
5. **40–50% reduction을 repair할 수 있는가?** R=8/age cap repair는 Balanced를 약 28%까지 복구했지만 40–50%에서 output gate를 만족하지 못했다.
6. **Teacher-forced가 closed loop에서도 유지되는가?** Phase-B gate 실패로 closed-loop는 실행하지 않았다. 이 항목은 미확인이다.
7. **Seed/metadata 포함 traffic이 줄어드는가?** Ideal token traffic은 줄지만 scattered seed가 32K block의 큰 비율을 touch해 B128 physical reduction이 크게 축소된다.
8. **Static CUDA-style filter로 대부분 얻는가?** 그렇다. 같은 gamma에서 static→dynamic median 추가 reduction은 0.028pp뿐이다.
9. **Dynamic HW가 정당화되는가?** 아니다. Quality gate 실패와 negligible dynamic gain 둘 다 dedicated scheduler/SRAM을 정당화하지 못한다.
10. **다음 단계는?** FPGA/HW가 아니라 tail-aware static/software policy 개선과 independent production-V3.2 validation이다.

## Algorithm quality

| Policy     | QK reduction | Exact match | Recall@K | Top-128 | Teacher mass | MLA cosine med/p5   | MLA RelL2 p95 |
| ---------- | ------------ | ----------- | -------- | ------- | ------------ | ------------------- | ------------- |
| Safe       | 17.10%       | 3.28%       | 99.03%   | 99.965% | 99.882%      | 0.999996 / 0.999142 | 4.16%         |
| Balanced   | 28.46%       | 5.98%       | 98.22%   | 99.904% | 99.780%      | 0.999994 / 0.997877 | 6.61%         |
| Aggressive | 41.00%       | 0.75%       | 96.01%   | 99.581% | 99.560%      | 0.999982 / 0.994987 | 10.09%        |

## Model quality

| Policy     | Logit KL | Top-1 agreement | PPL delta | Closed-loop | Task    |
| ---------- | -------- | --------------- | --------- | ----------- | ------- |
| Safe       | 0.003686 | 97.57%          | -0.03%    | skipped     | skipped |
| Balanced   | 0.004310 | 97.66%          | -0.47%    | skipped     | skipped |
| Aggressive | 0.005357 | 97.31%          | -0.25%    | skipped     | skipped |

## Hardware value at 32K

| Policy     | QK reduction | Ideal net bytes | Physical B64 | Physical B128 | Seed blocks touched | TopK cache   |
| ---------- | ------------ | --------------- | ------------ | ------------- | ------------------- | ------------ |
| Safe       | 17.10%       | 17.06%          | 17.01%       | 6.93%         | 63.42%              | 512 KiB BF16 |
| Balanced   | 28.46%       | 28.41%          | 28.36%       | 18.56%        | 61.01%              | 512 KiB BF16 |
| Aggressive | 41.00%       | 40.95%          | 40.89%       | 30.18%        | 54.86%              | 512 KiB BF16 |

## Baseline separation

Dense MLA teacher와 full-indexer research sidecar sparse baseline의 teacher-forced 비교는 logit KL mean 0.026216, Top-1 agreement 93.75%, PPL delta -0.24%였다. Approximate temporal 결과는 이 full-indexer sparse baseline에 대해 측정했으므로 sidecar 자체 오차와 temporal approximation 오차를 혼합하지 않았다.

## Phase A — runtime-legal replay

- Validation: 24 traces, held-out verdict: 144 traces / 18,288 transitions.
- Own-trajectory에서 seed는 이전 approximate Top-K이고, skip block은 stale max/age만 유지했다.
- Previous-TopK seed rescore를 QK cost에 포함했으며 current key 중복 score는 제거했다.
- 8K validation 절대 gamma가 16K/32K에서 과도하게 pruning되는 문제를 발견해 prespecified refresh/age sweeps를 적용했다.
- Safe는 age cap 2, Balanced는 periodic full refresh R=8, Aggressive는 no repair다.

## Phase B — actual teacher attention and MLA output

Teacher mass median은 세 정책 모두 거의 1이었지만 p5와 worst tail이 악화됐다. 실제 main MLA Q/K/V로 selection set만 바꾼 9,216 observations/policy에서 Safe/Balanced/Aggressive relative-L2 p95가 각각 4.16%, 6.61%, 10.09%였다. Worst concentration은 32K layers 17/21과 long-code layer 8에서 나타났다.

## Phase C — teacher-forced model validation

Text/code 각 3 prompts × 8K/16K/32K × 64 steps, selected 8 layers에서 dense, full-indexer sparse, 세 approximate 정책을 비교했다. Approximate 정책은 PPL delta ±1% gate를 통과했지만, 이 결과가 Phase-B attention-output tail을 무효화하지는 않는다. Residual path가 많은 오류를 흡수한다는 증거로 해석한다.

## Phase D early stop

Closed-loop generation, RULER/NIAH/LongBench/code task, query-change fallback, approximate CUDA-style timing은 명세의 early-stop 원칙에 따라 실행하지 않았다. 미실행 결과를 0이나 통과로 간주하지 않으며, 각각 CSV와 placeholder graph에 명시했다.

## Hardware interpretation

- Metadata는 last max + streak + age/bucket을 packed/aligned 8 B/block로 모델링했다.
- 32K에서 metadata는 B64 기준 4 KiB/layer이고 64K/128K extrapolation은 8/16 KiB/layer다.
- K=2048 seed-key cache는 BF16 512 KiB/layer, FP8 extrapolation 256 KiB/layer다.
- Seed Top-K가 scattered되어 physical B128 traffic reduction은 ideal token-level reduction보다 훨씬 작다.
- Same-margin static과 dynamic address order의 held-out median reduction 차이는 0.028pp로 hardware-specific feedback value가 관찰되지 않았다.

## Reproducibility and limitations

- Code commit: `a73eb97a`
- Model revision: `85864749cd611b4353ce1decdb286193298f64c7`
- GPUs: NVIDIA L40S, physical IDs 0/1 only; IDs 2/3 untouched.
- Unit tests: 16 passed.
- Main result files: `policy_pareto.csv`, `rank_recall.csv`, `teacher_mass.csv`, `mla_output_error.csv`, `teacher_forced_quality.csv`, `hardware_cost_model.csv`, `software_timing.csv`, `reproducibility.json`.
- Generated graphs: 20 under `graphs_approx/`.
- This result characterizes a BF16/FP32 research sidecar, not the production V3.2 FP8 Indexer or TensorRT-LLM kernel.

## References

- [DeepSeek-V3.2 paper](https://arxiv.org/abs/2512.02556)
- [Official DeepSeek-V3.2 experimental indexer reference](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)
- [DeepSeek-V2-Lite-Chat model](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)
