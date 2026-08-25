from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .metrics import normalized_recall_lift, stable_topk, topk_recall
from .quality import apply_quality_gate
from .sidecar import LightningIndexerSidecar, indexer_kl_loss
from .trace import ScoreTrace, save_trace


@dataclass(frozen=True)
class CaptureSample:
    path: Path
    hidden: torch.Tensor
    teacher: torch.Tensor
    lengths: torch.Tensor
    context_length: int
    metadata: dict[str, Any]


def load_capture(path: Path) -> CaptureSample:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"hidden", "teacher_probabilities", "lengths", "context_length", "metadata"}
    if not required.issubset(payload):
        raise ValueError(f"{path} is missing {sorted(required - set(payload))}")
    return CaptureSample(
        path=path,
        hidden=payload["hidden"],
        teacher=payload["teacher_probabilities"],
        lengths=payload["lengths"],
        context_length=int(payload["context_length"]),
        metadata=dict(payload["metadata"]),
    )


def discover_captures(roots: list[Path]) -> dict[int, list[CaptureSample]]:
    result: dict[int, list[CaptureSample]] = {}
    paths = sorted({path for root in roots for path in root.rglob("layer_*.pt")})
    for path in paths:
        sample = load_capture(path)
        layer = int(sample.metadata["layer"])
        result.setdefault(layer, []).append(sample)
    if not result:
        raise FileNotFoundError(f"no layer_*.pt captures found under {roots}")
    return result


def _new_indexer(args: argparse.Namespace, device: torch.device) -> LightningIndexerSidecar:
    indexer = LightningIndexerSidecar(
        hidden_size=args.hidden_size,
        heads=args.heads,
        head_dim=args.head_dim,
        rope_dim=args.rope_dim,
        rope_theta=args.rope_theta,
    ).to(device=device, dtype=torch.float32)
    return indexer


def _score_query(
    indexer: LightningIndexerSidecar,
    sample: CaptureSample,
    query_id: int,
    device: torch.device,
    key_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = int(sample.lengths[query_id])
    query_position = length - 1
    key_hidden = sample.hidden[:length].to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    query_hidden = sample.hidden[query_position : query_position + 1].to(
        device=device, dtype=torch.bfloat16
    ).unsqueeze(0)
    key_positions = torch.arange(length, device=device).view(1, -1)
    query_positions = torch.tensor([[query_position]], device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        scores = indexer(
            query_hidden,
            key_hidden,
            query_positions,
            key_positions,
            key_chunk_size=key_chunk_size,
        )
    teacher = sample.teacher[query_id, :length].to(device=device, dtype=torch.float32)
    return scores, teacher.view(1, 1, -1)


def train_layer(
    layer: int,
    samples: list[CaptureSample],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[LightningIndexerSidecar, list[dict[str, float]]]:
    indexer = _new_indexer(args, device)
    optimizer = torch.optim.AdamW(indexer.parameters(), lr=args.learning_rate, weight_decay=0.0)
    generator = random.Random(args.seed + layer)
    history = []
    indexer.train()
    for step in range(args.train_steps):
        sample = generator.choice(samples)
        query_id = generator.randrange(sample.teacher.shape[0])
        optimizer.zero_grad(set_to_none=True)
        scores, teacher = _score_query(indexer, sample, query_id, device, args.key_chunk_size)
        loss = indexer_kl_loss(scores, teacher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(indexer.parameters(), args.max_grad_norm)
        optimizer.step()
        if step % args.log_every == 0 or step + 1 == args.train_steps:
            history.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    return indexer.eval(), history


@torch.no_grad()
def evaluate_and_trace(
    indexer: LightningIndexerSidecar,
    samples: list[CaptureSample],
    layer: int,
    args: argparse.Namespace,
    device: torch.device,
    trace_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        query_ids = np.linspace(
            0,
            sample.teacher.shape[0] - 1,
            min(args.eval_queries, sample.teacher.shape[0]),
            dtype=int,
        )
        score_rows: list[np.ndarray] = []
        lengths: list[int] = []
        all_hidden = sample.hidden.to(device=device, dtype=torch.bfloat16).unsqueeze(0)
        all_positions = torch.arange(all_hidden.shape[1], device=device).view(1, -1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded_keys = indexer.encode_keys(all_hidden, all_positions)
        for query_id in query_ids:
            length = int(sample.lengths[int(query_id)])
            query_position = length - 1
            query_hidden = all_hidden[:, query_position : query_position + 1]
            query_position_tensor = torch.tensor([[query_position]], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                queries, weights = indexer.encode_queries(query_hidden, query_position_tensor)
                scores = indexer.score_encoded(
                    queries,
                    weights,
                    encoded_keys[:, :length],
                    key_chunk_size=args.key_chunk_size,
                )
            index_scores = scores[0, 0].float().cpu().numpy()
            teacher = sample.teacher[int(query_id), : index_scores.size].float().numpy()
            score_rows.append(index_scores.astype(np.float16))
            lengths.append(index_scores.size)
            for k in args.k_values:
                if k > index_scores.size:
                    continue
                recall = topk_recall(stable_topk(index_scores, k), stable_topk(teacher, k))
                rows.append(
                    {
                        "layer": layer,
                        "workload": sample.metadata["workload"],
                        "prompt_id": sample.metadata["prompt_id"],
                        "query": int(query_id),
                        "context_length": index_scores.size,
                        "k": k,
                        "recall": recall,
                        "normalized_lift": normalized_recall_lift(recall, k, index_scores.size),
                    }
                )
        width = max(lengths)
        padded = np.full((len(score_rows), width), -np.inf, dtype=np.float16)
        for row_id, score_row in enumerate(score_rows):
            padded[row_id, : score_row.size] = score_row
        trace = ScoreTrace(
            scores=padded,
            lengths=np.asarray(lengths, dtype=np.int32),
            layer=layer,
            workload=str(sample.metadata["workload"]),
            prompt_id=str(sample.metadata["prompt_id"]),
            metadata={
                "source_capture": str(sample.path),
                "model_revision": sample.metadata.get("model_revision"),
                "source": sample.metadata.get("source"),
                "split": sample.metadata.get("split"),
                "query_ids": query_ids.tolist(),
                "research_sidecar": True,
            },
        )
        save_trace(
            trace_root
            / f"{sample.metadata['prompt_id']}_c{sample.context_length}_layer_{layer:02d}.npz",
            trace,
        )
        del all_hidden, all_positions, encoded_keys
    return rows


def _save_indexer(indexer: LightningIndexerSidecar, path: Path, metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file

        save_file({key: value.detach().cpu() for key, value in indexer.state_dict().items()}, path, metadata)
    except ImportError:
        torch.save({"state_dict": indexer.state_dict(), "metadata": metadata}, path.with_suffix(".pt"))


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("sidecar warm-up requires CUDA")
    device = torch.device(args.device)
    train = discover_captures(args.train_roots)
    evaluate = discover_captures(args.eval_roots)
    if args.layers:
        selected = set(args.layers)
        train = {layer: samples for layer, samples in train.items() if layer in selected}
        evaluate = {layer: samples for layer, samples in evaluate.items() if layer in selected}
        missing = selected - set(train)
        if missing:
            raise ValueError(f"requested layers have no training captures: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    all_quality: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for layer in sorted(train):
        if layer not in evaluate:
            raise ValueError(f"layer {layer} has no evaluation capture")
        indexer, history = train_layer(layer, train[layer], args, device)
        histories[str(layer)] = history
        _save_indexer(
            indexer,
            args.output / "checkpoints" / f"layer_{layer:02d}.safetensors",
            {
                "layer": str(layer),
                "architecture": "v2-lite-direct-hidden-research-sidecar",
                "seed": str(args.seed),
            },
        )
        all_quality.extend(
            evaluate_and_trace(
                indexer,
                evaluate[layer],
                layer,
                args,
                device,
                args.output / "traces",
            )
        )
        del indexer
        torch.cuda.empty_cache()

    quality = pd.DataFrame(all_quality)
    quality.to_csv(args.output / "quality_rows.csv", index=False)
    gate = apply_quality_gate(
        quality,
        gate_k=args.gate_k,
        min_median_normalized_lift=args.min_median_normalized_lift,
        min_layer_pass_fraction=args.min_layer_pass_fraction,
    )
    summary = {
        "passed": gate.passed,
        "layer_pass_fraction": gate.layer_pass_fraction,
        "failed_layers": gate.failed_layers,
        "reason": gate.reason,
        "gate_is_pilot_policy_not_official_deepseek_threshold": True,
        "train_roots": [str(path) for path in args.train_roots],
        "eval_roots": [str(path) for path in args.eval_roots],
        "seed": args.seed,
        "history": histories,
    }
    (args.output / "quality_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not gate.passed:
        raise SystemExit(42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm up and evaluate per-layer Lightning Indexer sidecars")
    parser.add_argument("--train-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--key-chunk-size", type=int, default=4096)
    parser.add_argument("--eval-queries", type=int, default=256)
    parser.add_argument("--k-values", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--gate-k", type=int, default=512)
    parser.add_argument("--min-median-normalized-lift", type=float, default=0.20)
    parser.add_argument("--min-layer-pass-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=1582)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
