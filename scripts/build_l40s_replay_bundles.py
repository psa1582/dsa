from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from temporal_dsa.approx import ApproxPolicy
from temporal_dsa.trace import load_trace
from temporal_dsa.verifier import VerifierConfig, replay_verifier_trace
from temporal_dsa.verifier_scoring import load_sidecar_encoded


CONTEXTS = (8192, 16384, 32768)
BLOCK_SIZE = 64
TOPK = 2048
WIDTH = 8
POLICY = ApproxPolicy(
    name="streak2_bucket8_m0",
    gamma=0.0,
    dynamic_threshold=True,
    order="bucket8",
    cold_streak=2,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_mask(indices: np.ndarray, length: int) -> np.ndarray:
    result = np.zeros(math.ceil(length / BLOCK_SIZE), dtype=bool)
    if indices.size:
        result[np.unique(np.asarray(indices, dtype=np.int64) // BLOCK_SIZE)] = True
    return result


def choose_observations(
    details: dict[int, dict[str, np.ndarray]],
    h8: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
    context: int,
) -> list[tuple[str, int, dict[str, float]]]:
    metrics: list[dict[str, float | int | bool]] = []
    for step, item in sorted(details.items()):
        length = int(lengths[step])
        baseline = item["baseline"]
        approximate = item["approximate"]
        newly = item["newly_active"]
        cold_blocks = np.unique(item["cold"] // BLOCK_SIZE)
        if cold_blocks.size:
            maxima = np.asarray(
                [
                    np.max(h8[step, block * BLOCK_SIZE : min(length, (block + 1) * BLOCK_SIZE)])
                    for block in cold_blocks
                ],
                dtype=np.float32,
            )
            near_gap = float(np.min(np.abs(maxima - threshold)))
        else:
            near_gap = math.inf
        tail_start = int(length * 0.75)
        metrics.append(
            {
                "step": step,
                "exact": bool(np.array_equal(np.sort(approximate), np.sort(baseline))),
                "recall": float(np.isin(baseline, approximate, assume_unique=False).mean()),
                "newly": int(newly.size),
                "tail_newly": int(np.count_nonzero(newly >= tail_start)),
                "near_gap": near_gap,
            }
        )

    rankings: list[tuple[str, list[dict[str, float | int | bool]]]] = [
        (
            "easy",
            sorted(
                metrics,
                key=lambda row: (
                    not bool(row["exact"]),
                    -float(row["recall"]),
                    int(row["newly"]),
                    int(row["step"]),
                ),
            ),
        ),
        (
            "newly_active",
            sorted(
                metrics,
                key=lambda row: (
                    -int(row["newly"]),
                    -int(row["tail_newly"]),
                    -float(row["recall"]),
                ),
            ),
        ),
        (
            "near_threshold",
            sorted(metrics, key=lambda row: (float(row["near_gap"]), -int(row["newly"]))),
        ),
    ]
    if context == 32768:
        rankings.append(
            (
                "32k_tail",
                sorted(
                    metrics,
                    key=lambda row: (
                        -int(row["tail_newly"]),
                        -int(row["newly"]),
                        -float(row["recall"]),
                    ),
                ),
            )
        )

    used: set[int] = set()
    selected: list[tuple[str, int, dict[str, float]]] = []
    for category, ranking in rankings:
        row = next((candidate for candidate in ranking if int(candidate["step"]) not in used), ranking[0])
        step = int(row["step"])
        used.add(step)
        selected.append(
            (
                category,
                step,
                {
                    "exact": float(bool(row["exact"])),
                    "recall": float(row["recall"]),
                    "newly_active_tokens": float(row["newly"]),
                    "tail_newly_active_tokens": float(row["tail_newly"]),
                    "nearest_promotion_threshold_gap": float(row["near_gap"]),
                },
            )
        )
    return selected


def pad_rows(rows: list[np.ndarray], width: int, fill: float | bool = 0) -> np.ndarray:
    dtype = rows[0].dtype
    output = np.full((len(rows), width), fill, dtype=dtype)
    for index, row in enumerate(rows):
        output[index, : row.size] = row
    return output


def build_context(
    repo: Path,
    output: Path,
    *,
    context: int,
    prompt: str,
    layer: int,
    device: str,
    thresholds: dict[str, float],
    t1: dict[str, list[float]],
) -> tuple[Path, list[dict[str, Any]]]:
    trace_path = repo / "artifacts" / "pilot" / "scores_b" / "traces" / (
        f"{prompt}_c{context}_layer_{layer}.npz"
    )
    cache_path = (
        repo / "artifacts" / "h8_reconstruction" / "cache" / "test" / trace_path.name
    )
    checkpoint_path = (
        repo
        / "artifacts"
        / "pilot"
        / "indexers_1000"
        / "checkpoints"
        / f"layer_{layer:02d}.safetensors"
    )
    trace = load_trace(trace_path)
    with np.load(trace_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
    with np.load(cache_path, allow_pickle=False) as payload:
        h8 = payload["h8"].astype(np.float32)
        head_ids = payload["head_ids"].astype(np.int32)
        cached_lengths = payload["lengths"].astype(np.int32)
    if not np.array_equal(cached_lengths, trace.lengths):
        raise RuntimeError(f"H8 cache length mismatch: {cache_path}")

    verifier = VerifierConfig(
        name="head_dynamic_abs_w_w8_b64_threshold_r0.1",
        path="head",
        width=WIDTH,
        score_ratio=WIDTH / 64,
        block_size=BLOCK_SIZE,
        rescue_fraction=0.1,
        retain_candidate_keys=True,
    )
    detail_rows: dict[int, dict[str, np.ndarray]] = {}

    def collect(step: int, values: dict[str, np.ndarray]) -> None:
        detail_rows[step] = values

    threshold = float(thresholds[str(layer)])
    replay_verifier_trace(
        trace,
        h8,
        policy=POLICY,
        config=verifier,
        promotion_threshold=threshold,
        detail_callback=collect,
    )
    selected = choose_observations(detail_rows, h8, trace.lengths, threshold, context)
    steps = np.asarray([step for _, step, _ in selected], dtype=np.int32)
    lengths = trace.lengths[steps].astype(np.int32)
    previous_lengths = trace.lengths[steps - 1].astype(np.int32)
    capture_path = repo / metadata["source_capture"]
    queries, weights, keys, _ = load_sidecar_encoded(
        capture_path, checkpoint_path, lengths, device=device
    )
    max_length = int(lengths.max())
    keys = keys[:max_length].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    queries = queries.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    weights = weights.detach().to(device="cpu", dtype=torch.float32).contiguous()

    temporal_token_rows: list[np.ndarray] = []
    temporal_block_rows: list[np.ndarray] = []
    promotion_block_rows: list[np.ndarray] = []
    exact_candidate_rows: list[np.ndarray] = []
    full_rows: list[np.ndarray] = []
    h8_rows: list[np.ndarray] = []
    t1_rows: list[np.ndarray] = []
    previous_rows: list[np.ndarray] = []
    topk_rows: list[np.ndarray] = []
    observation_metadata: list[dict[str, Any]] = []
    block_width = math.ceil(max_length / BLOCK_SIZE)
    t1_coefficients = np.asarray(t1[f"t1_layer_{layer}"], dtype=np.float32)

    for category, step, selection_metrics in selected:
        length = int(trace.lengths[step])
        previous_length = int(trace.lengths[step - 1])
        details = detail_rows[step]
        temporal_tokens = np.zeros(length, dtype=bool)
        temporal_tokens[details["temporal_evaluated"]] = True
        exact_candidates = np.zeros(length, dtype=bool)
        exact_candidates[details["exact_candidates"]] = True
        promotions = np.zeros(math.ceil(length / BLOCK_SIZE), dtype=bool)
        promotions[np.asarray(details["rescued_blocks"], dtype=np.int64)] = True
        temporal_blocks = block_mask(details["temporal_evaluated"], length)
        temporal_token_rows.append(temporal_tokens)
        temporal_block_rows.append(temporal_blocks)
        promotion_block_rows.append(promotions)
        exact_candidate_rows.append(exact_candidates)
        full_rows.append(trace.row(step).astype(np.float32, copy=True))
        current_h8 = h8[step, :length].astype(np.float32, copy=True)
        previous_full = trace.row(step - 1).astype(np.float32, copy=True)
        h8_rows.append(current_h8)
        previous_rows.append(previous_full)
        t1_row = np.full(length, -math.inf, dtype=np.float32)
        common = min(length, previous_full.size)
        t1_row[:common] = (
            t1_coefficients[0] * current_h8[:common]
            + t1_coefficients[1] * previous_full[:common]
            + t1_coefficients[2]
        )
        t1_rows.append(t1_row)
        topk_rows.append(np.asarray(details["baseline"], dtype=np.int32))
        observation_metadata.append(
            {
                "category": category,
                "step": int(step),
                "length": length,
                "previous_length": previous_length,
                "temporal_exact_tokens": int(temporal_tokens.sum()),
                "temporal_block_any_count": int(temporal_blocks.sum()),
                "promoted_block_count": int(promotions.sum()),
                "exact_candidate_count": int(exact_candidates.sum()),
                **selection_metrics,
            }
        )

    bundle: dict[str, Any] = {
        "schema_version": "dsa-replay-v1",
        "q_indexer": queries,
        "k_indexer": keys,
        "head_weight": weights,
        "lengths": torch.from_numpy(lengths),
        "previous_lengths": torch.from_numpy(previous_lengths),
        "steps": torch.from_numpy(steps),
        "dynamic_h8_ids": torch.from_numpy(head_ids[steps]).to(torch.int32).contiguous(),
        "temporal_token_mask": torch.from_numpy(
            pad_rows(temporal_token_rows, max_length, False)
        ).contiguous(),
        "temporal_block_mask": torch.from_numpy(
            pad_rows(temporal_block_rows, block_width, False)
        ).contiguous(),
        "promotion_block_mask": torch.from_numpy(
            pad_rows(promotion_block_rows, block_width, False)
        ).contiguous(),
        "exact_candidate_mask": torch.from_numpy(
            pad_rows(exact_candidate_rows, max_length, False)
        ).contiguous(),
        "promotion_threshold": torch.full((len(selected),), threshold, dtype=torch.float32),
        "full64_score": torch.from_numpy(pad_rows(full_rows, max_length, -math.inf)),
        "h8_score": torch.from_numpy(pad_rows(h8_rows, max_length, -math.inf)),
        "previous_full64_score": torch.from_numpy(
            pad_rows(previous_rows, max_length, -math.inf)
        ),
        "t1_score": torch.from_numpy(pad_rows(t1_rows, max_length, -math.inf)),
        "exact_topk_ids": torch.from_numpy(np.stack(topk_rows)).to(torch.int32),
        "t1_coefficients": torch.from_numpy(
            np.repeat(t1_coefficients[None, :], len(selected), axis=0)
        ),
        "metadata": {
            "model": "deepseek-ai/DeepSeek-V2-Lite-Chat research sidecar",
            "model_revision": metadata["model_revision"],
            "prompt_id": metadata["prompt_id"],
            "workload": metadata["workload"],
            "split": metadata["split"],
            "source": metadata["source"],
            "layer": layer,
            "nominal_context": context,
            "block_size": BLOCK_SIZE,
            "topk": TOPK,
            "policy": POLICY.name,
            "policy_gamma": POLICY.gamma,
            "policy_order": POLICY.order,
            "policy_cold_streak": POLICY.cold_streak,
            "verifier": verifier.name,
            "h8_width": WIDTH,
            "promotion_policy": "validation_fixed_threshold",
            "promotion_threshold": threshold,
            "t1_formula": "a * current_h8 + b * previous_full64 + c",
            "t1_coefficients": t1_coefficients.tolist(),
            "score_scaling": "sum_h w_h * ReLU(q_h dot k); no 1/sqrt(D) factor",
            "stable_topk_tie_break": "descending score, then ascending token ID",
            "checkpoint_independent_replay": True,
            "source_materialization": {
                "trace": str(trace_path.relative_to(repo)),
                "trace_sha256": sha256(trace_path),
                "h8_cache": str(cache_path.relative_to(repo)),
                "h8_cache_sha256": sha256(cache_path),
                "capture": str(capture_path.relative_to(repo)),
                "capture_sha256": sha256(capture_path),
                "checkpoint": str(checkpoint_path.relative_to(repo)),
                "checkpoint_sha256": sha256(checkpoint_path),
            },
            "observations": observation_metadata,
            "tensor_contract": {
                "q_indexer": "[O,64,128] BF16 contiguous",
                "k_indexer": "[max_length,128] BF16 contiguous shared prefix",
                "head_weight": "[O,64] FP32 contiguous",
                "dynamic_h8_ids": "[O,8] INT32 stable |w| routing",
                "temporal_token_mask": "[O,max_length] BOOL, padded false",
                "temporal_block_mask": "[O,ceil(max_length/64)] BOOL; any temporal token",
                "promotion_block_mask": "[O,ceil(max_length/64)] BOOL",
                "full64_score": "[O,max_length] FP32, padded -inf",
                "h8_score": "[O,max_length] FP32, padded -inf",
                "previous_full64_score": "[O,max_length] FP32, padded -inf",
                "t1_score": "[O,max_length] FP32, padded -inf",
                "exact_topk_ids": "[O,2048] INT32 stable reference",
            },
            "block_expansion_note": (
                "The fused SM89 prototype consumes temporal_block_mask, which expands any "
                "token-granular temporal hit to a full B64 block; exact_candidate_mask preserves "
                "the production-legal token-granular reference separately."
            ),
        },
    }
    layout_keys = (
        "q_indexer",
        "k_indexer",
        "head_weight",
        "dynamic_h8_ids",
        "temporal_token_mask",
        "temporal_block_mask",
        "promotion_block_mask",
        "full64_score",
        "h8_score",
        "previous_full64_score",
        "t1_score",
        "exact_topk_ids",
    )
    bundle["metadata"]["tensor_layouts"] = {
        key: {
            "shape": list(bundle[key].shape),
            "dtype": str(bundle[key].dtype).removeprefix("torch."),
            "stride": list(bundle[key].stride()),
            "contiguous": bundle[key].is_contiguous(),
        }
        for key in layout_keys
    }
    destination = output / f"dsa_replay_c{context}_layer{layer}_{prompt}.pt"
    torch.save(bundle, destination)
    bundle_hash = sha256(destination)
    manifest_rows: list[dict[str, Any]] = []
    for index, observation in enumerate(observation_metadata):
        manifest_rows.append(
            {
                "bundle": destination.name,
                "bundle_sha256": bundle_hash,
                "schema_version": bundle["schema_version"],
                "model_revision": metadata["model_revision"],
                "prompt_id": prompt,
                "workload": metadata["workload"],
                "split": metadata["split"],
                "layer": layer,
                "nominal_context": context,
                "observation_index": index,
                **observation,
            }
        )
    return destination, manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize checkpoint-independent real-trace DSA replay bundles"
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="code_heldout_3")
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = json.loads(
        (args.repo / "artifacts" / "progressive_sw" / "threshold" / "runtime_config.json")
        .read_text(encoding="utf-8")
    )
    thresholds = runtime["configs"][0]["promotion_threshold_by_layer"]
    t1 = json.loads(
        (args.repo / "artifacts" / "h8_reconstruction" / "final" / "coefficients.json")
        .read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    for context in CONTEXTS:
        path, context_rows = build_context(
            args.repo,
            args.output,
            context=context,
            prompt=args.prompt,
            layer=args.layer,
            device=args.device,
            thresholds=thresholds,
            t1=t1,
        )
        rows.extend(context_rows)
        print(json.dumps({"bundle": str(path), "observations": len(context_rows)}), flush=True)
    manifest = args.output.parent / "replay_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"manifest": str(manifest), "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
