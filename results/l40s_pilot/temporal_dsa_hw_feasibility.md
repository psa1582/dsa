# Temporal DSA Indexer Pruning — L40S ×2 Feasibility Pilot

## Verdict: SOFTWARE-ONLY

전용 하드웨어의 근거로 삼을 수 있는 exact temporal pruning 이득은 확인되지 않았다. Validation에서 고정한 레이어·block별 최대 변화 margin을 held-out에 적용하면 exact-match 99.963%까지 회복하지만, median QK reduction은 0%, 평균은 0.468%에 불과하다. B64 metadata까지 포함한 median byte reduction은 -0.049%다.

반면 이전 Top-K를 먼저 계산하는 hot-first 순서는 final Top-K를 발견하는 시점을 전체 key scan의 99.97%에서 59.31%로 앞당겼다. 따라서 현재 증거는 전용 pruning datapath보다 software scheduling, cache locality, threshold warm-start 최적화를 후속 대상으로 지지한다.

## 한눈에 보는 판단 근거

| 정책 | QK reduction | exact-match | Recall@K | false-cold | 해석 |
| --- | ---: | ---: | ---: | ---: | --- |
| Full dense | 0% | 100% | 100% | 0% | 기준선 |
| Static, 1σ 분석 sweep | 17.60% | 93.39% | 99.890% | 0.441% | 부정확, 배포 불가 |
| Dynamic + previous-hot, 1σ 분석 sweep | 48.24% | 36.32% | 97.490% | 5.172% | 큰 ceiling이지만 부정확 |
| Static, validation-max margin | 0.023% mean | 99.999% | ~100% | 0.00027% | metadata 비용보다 작음 |
| Dynamic + previous-hot, validation-max margin | 0% median / 0.468% mean | 99.963% | 99.9998% | 0.00847% | 거의 정확하지만 이득 없음 |
| Oracle current-block max | 63.77% mean | 100% | 100% | 0% | 구현 불가능한 ceiling |

`1σ` 값은 held-out score 변화로 계산한 분석 sweep이며 online 정책이 아니다. 최종 verdict는 quality split에서 미리 고정한 `validation-max margin`을 기준으로 한다. 이 margin도 경험적일 뿐 수학적 certificate는 아니다.

## 실험 범위와 실제 하드웨어

- 서버는 사용자가 지정한 `10.201.135.16:7021`이다.
- 실제 GPU SKU는 요청서의 L40이 아니라 **NVIDIA L40S 48GB ×4**였다.
- GPU 0·1만 `CUDA_VISIBLE_DEVICES=0,1`로 사용했다. GPU 2·3의 기존 vLLM 작업은 건드리지 않았다.
- 모델은 `deepseek-ai/DeepSeek-V2-Lite-Chat` revision `85864749cd611b4353ce1decdb286193298f64c7`이다.
- PyTorch 2.9.1+cu128, CUDA runtime 12.8, Transformers 4.57.3, Python 3.12.12를 사용했다.
- 선택 레이어는 `{2, 5, 8, 11, 14, 17, 21, 25}`다.
- 4K·8K warm-up, 분리된 8K quality split, held-out 8K·16K·32K에서 각각 128 consecutive decode steps를 수집했다.
- 최종 temporal trace는 6 prompts × 3 lengths × 8 layers = 144개, 인접 전이는 18,288개다.

실제 trace 수집 시간은 다음과 같다.

| Context | Prompts | Decode steps | Prefill 합계 | Decode 합계 | Capture apparent size |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 6 | 768 | 7.52 s | 58.73 s | 1.7 GB |
| 16K | 6 | 768 | 17.94 s | 68.78 s | 3.3 GB |
| 32K | 6 | 768 | 52.05 s | 94.86 s | 6.5 GB |

## 연구 사이드카와 production Indexer의 차이

공식 V3.2 dense warm-up과 동일하게 teacher는 dense MLA의 head별 softmax probability를 head 방향으로 합산한 뒤 sequence 방향 L1-normalize했다. 학습 loss는 `D_KL(p_teacher || Softmax(I))`다.

사이드카 score는 다음 식을 사용했다.

`I_ts = Σ_j w_tj ReLU(q_tjᵀ k_s)`, `H_I=64`, `d_I=128`.

단, 이 결과는 production V3.2 Indexer가 아니다.

- V2-Lite에는 V3.2의 q-LoRA indexer input이 없어 layer input hidden state를 직접 projection했다.
- 학습은 FP32 master parameter/optimizer state와 BF16 autocast forward를 사용했다.
- V3.2의 FP8 quantization·Hadamard kernel 대신 BF16/FP32 reference score를 사용했다.
- backbone은 frozen이고 레이어별 sidecar만 학습했다.
- checkpoint와 capture hidden state로 q, w, K를 재구성할 수 있지만 production checkpoint와 호환되지 않는다.

## Indexer quality gate

Gate는 Recall@512의 normalized lift `(recall - 512/L) / (1 - 512/L)` median이 0.20 이상인 레이어 비율 75% 이상으로 정의했다. 이는 파일럿 정책이지 DeepSeek 공식 threshold가 아니다. 8개 레이어 모두 통과했다.

| Layer | Code Recall@512 median | Text Recall@512 median | 가장 낮은 median lift |
| ---: | ---: | ---: | ---: |
| 2 | 0.623 | 0.652 | 0.598 |
| 5 | 0.586 | 0.456 | 0.420 |
| 8 | 0.573 | 0.471 | 0.436 |
| 11 | 0.590 | 0.430 | 0.392 |
| 14 | 0.444 | 0.445 | 0.407 |
| 17 | 0.413 | 0.315 | 0.270 |
| 21 | 0.342 | 0.379 | 0.298 |
| 25 | 0.376 | 0.472 | 0.335 |

Gate 실패 레이어는 없다. 가장 약한 조합은 layer 17의 long-text였지만 median lift 0.270으로 정책 threshold를 넘었다.

## Temporal locality

전체 인접 Top-K overlap은 높았지만 context가 길어질수록 약해졌다.

| K | Mean overlap | Median overlap | Min observation |
| ---: | ---: | ---: | ---: |
| 128 | 0.62 | 0.63 | 0.05 |
| 256 | 0.64 | 0.65 | 0.05 |
| 512 | 0.66 | 0.67 | 0.11 |
| 1024 | 0.69 | 0.71 | 0.18 |
| 2048 | 0.74 | 0.75 | 0.27 |

Top-512의 context/workload별 평균 overlap은 다음과 같다.

| Context | Long code | Long text |
| ---: | ---: | ---: |
| 8K | 0.750 | 0.751 |
| 16K | 0.664 | 0.663 |
| 32K | 0.574 | 0.571 |

Adjacent score Pearson/Spearman 평균은 모든 context와 두 workload에서 대체로 0.89–0.92였다. 즉 score 자체는 안정적이지만, 극단값의 block upper bound로 exact pruning을 보장하기에는 tail 변화가 충분히 컸다.

## Block replay와 scan order

분석은 block size 16/32/64/128, K 128/256/512/1024/2048, gamma 0/0.25/0.5/1σ, validation-max layer margin을 포함한다. 비교군은 Full, Static, Static+hot-first, Dynamic running Top-K feedback, Oracle current-block max다.

Validation-max margin에서 previous-hot과 upper-hot은 QK 수를 거의 줄이지 않았지만 threshold discovery fraction을 0.9997에서 0.5931로 낮췄다. 이 40.66 percentage-point 차이가 `SOFTWARE-ONLY` 판단의 근거다. 다만 scan order만으로는 수학적 QK 수가 줄지 않는다.

모든 transition이 exact였던 non-oracle dynamic trace/config 행은 전체의 19.24%였다. 이 부분집합의 median QK reduction은 0%, 최대는 17.26%였다. 즉 일부 layer/context/K에서는 기회가 있지만 하나의 고정 정책으로 일반화되지 않았다.

## Oracle ceiling과 하드웨어 byte 관점

Oracle은 현재 step의 block max를 미리 안다는 가정이므로 구현 가능한 결과가 아니다. K=512, B64에서의 exact ceiling은 다음과 같다.

| Context | Oracle QK reduction |
| ---: | ---: |
| 8K | 46.56% |
| 16K | 66.41% |
| 32K | 73.94% |

반면 선택된 validation-max policy의 median QK reduction은 0%다. block metadata를 8 bytes/block으로 잡으면 B64에서 예상 net byte reduction은 32K/64K/128K 모두 약 -0.049%다. 따라서 현재 bound를 전용 SRAM metadata와 pruning control로 구현할 이유가 없다.

## L40S software baseline

한 개 frozen sidecar의 cached-key dense BF16 index score와 `torch.topk(K=2048)`를 측정했다. 각 길이마다 10 warm-up 후 CUDA synchronize를 포함한 50회 측정이다.

| Context | Median latency | p5 | p95 | BF16 key bytes |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 0.202 ms | 0.200 ms | 0.217 ms | 2.0 MiB |
| 16K | 0.226 ms | 0.224 ms | 0.241 ms | 4.0 MiB |
| 32K | 0.228 ms | 0.227 ms | 0.238 ms | 8.0 MiB |

이는 dense research-sidecar one-shot baseline이지 TensorRT-LLM sparse kernel이나 end-to-end decode latency가 아니다. 따라서 이 표에서 하드웨어 speedup을 주장하지 않는다.

## Certified Phase D

ReLU의 1-Lipschitz 성질과 Cauchy-Schwarz를 이용한 signed-weight-safe radius는 구현하고 unit test로 모든 key의 실제 score 변화를 덮는지 확인했다.

`|ΔI| ≤ Σ_j (|w_tj-w_pj| ||q_pj|| + |w_tj| ||q_tj-q_pj||) max_{s∈block} ||k_s||`.

하지만 empirical filter가 exact성과 유의미한 pruning을 동시에 달성하지 못해 certified Phase D 성능 실험으로 승격하지 않았다. 더 느슨할 가능성이 큰 certified bound의 하드웨어 효율을 주장하는 것은 부적절하다.

## 최종 권고

1. 전용 temporal-pruning hardware는 진행하지 않는다.
2. TensorRT-LLM software path에서는 previous Top-K seed를 먼저 계산해 running threshold를 빠르게 만드는 scheduling 실험을 진행할 가치가 있다.
3. 다음 하드웨어 재평가 조건은 per-block scalar `M_prev + γ`보다 훨씬 타이트하면서 online-computable한 predictor가 held-out에서 exact 100%와 최소 25% net QK reduction을 동시에 보이는 경우다.
4. production 판단 전에는 실제 V3.2 FP8 Indexer trace와 TensorRT-LLM kernel timing으로 재검증해야 한다.

## Reproducibility

- Code commit: `0a03bce8482c58ec9e7dbdb1f70afa112fa7b2d4` (short: `0a03bce`)
- Model revision: `85864749cd611b4353ce1decdb286193298f64c7`
- Seed: `1582`
- Unit tests: `11 passed`
- Full replay: 144 traces, 18,288 transitions, 16 CPU workers, 155.66 s with calibrated margin
- Text source: Project Gutenberg public-domain books with disjoint calibration/validation/heldout IDs
- Code source: disjoint Python 3.10 standard-library file groups

세부 수치는 [detailed_summary.json](detailed_summary.json), [verdict.json](verdict.json), [quality_gate.json](quality_gate.json), [calibrated_margins.json](calibrated_margins.json), [software_baseline.json](software_baseline.json)에 보존했다.

## Figures

1. [Adjacent Top-K overlap](graphs/01_adjacent_topk_overlap.png)
2. [Overlap by layer](graphs/02_overlap_layer_heatmap.png)
3. [Rank stability by workload](graphs/03_rank_stability_by_workload.png)
4. [Adjacent score correlation](graphs/04_score_correlation.png)
5. [Score delta distribution](graphs/05_score_delta_distribution.png)
6. [Block-max delta](graphs/06_block_max_delta.png)
7. [Cold persistence](graphs/07_cold_persistence.png)
8. [Seed vs final threshold](graphs/08_seed_vs_final_threshold.png)
9. [Pruning–recall frontier](graphs/09_pruning_recall_frontier.png)
10. [Method/hardware comparison](graphs/10_method_hardware_comparison.png)

## Primary references

- [DeepSeek-V3.2 paper](https://arxiv.org/abs/2512.02556)
- [Official DeepSeek-V3.2 inference Indexer](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)
- [DeepSeek-V2-Lite-Chat model](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)
