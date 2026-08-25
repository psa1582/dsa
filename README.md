# Temporal DSA Indexer Pruning Pilot

Reproducible feasibility pipeline for using consecutive decode-step history to reduce Lightning Indexer QK/key traffic. The target experiment is DeepSeek-V2-Lite-Chat with eight frozen-backbone, per-layer research sidecars on two L40-class GPUs.

The repository does **not** claim a result until the indexer quality gate and real traces pass. DeepSeek-V2-Lite has no V3.2 q-LoRA indexer input, so the sidecar projects the layer input directly. It also uses BF16/FP32 rather than the production FP8 kernel. Results therefore characterize the temporal hypothesis, not production DeepSeek-V3.2 or TensorRT-LLM.

## Pipeline

1. Capture selected-layer hidden states and the official dense warm-up target: per-head dense MLA probabilities summed across heads, then L1-normalized.
2. Warm up one 64×128 Lightning Indexer sidecar per layer with the backbone frozen.
3. Stop with `INCONCLUSIVE` if the held-out Recall@K gate fails.
4. Save 8K/16K/32K consecutive decode score traces.
5. Replay block sizes 16/32/64/128 with previous Top-K seeding, hot-first order, running Top-K feedback, and an oracle ceiling.
6. Generate ten required figures, hardware byte extrapolations, and `temporal_dsa_hw_feasibility.md`.

## Install and test

```bash
python -m pip install -e '.[gpu,dev]'
pytest -q
```

## L40/L40S guard and smoke capture

Only GPUs 0 and 1 are made visible. The collector refuses non-L40 names unless the explicit smoke-only override is passed.

```bash
bash scripts/remote_preflight.sh
export CUDA_VISIBLE_DEVICES=0,1
MODEL="$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
python -m temporal_dsa.v2_collect \
  --model "$MODEL" --local-files-only \
  --prompts examples/smoke_prompts.jsonl --allow-repeat \
  --context-length 4096 --decode-steps 32 \
  --layers 2 --output artifacts/smoke/captures
```

The repeated smoke prompts validate plumbing only and are deliberately labeled so they cannot enter a final verdict.

## Warm-up, quality gate, and trace scoring

Training and evaluation roots should be disjoint prompt sets. The gate's normalized lift is `(Recall@K - K/L) / (1 - K/L)`; its threshold is configurable pilot policy, not an official DeepSeek threshold.

```bash
python -m temporal_dsa.training \
  --train-roots artifacts/warmup_4k artifacts/warmup_8k \
  --eval-roots artifacts/heldout_8k \
  --output artifacts/indexers
```

Exit code 42 means the quality gate failed. Temporal results from that run must be reported as `INCONCLUSIVE`.

## Replay and report

```bash
temporal-dsa analyze \
  --trace-roots artifacts/indexers/traces \
  --quality-gate artifacts/indexers/quality_gate.json \
  --config configs/pilot.yaml \
  --output artifacts/report
```

Static and dynamic `M_previous + gamma` filters are empirical and may miss keys. `oracle_current` is never presented as implementable. The signed-weight ReLU Lipschitz/Cauchy radius in `certified.py` is a separate safe Phase-D primitive and is covered by unit tests.

