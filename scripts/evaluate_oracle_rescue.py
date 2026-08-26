from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from evaluate_mla_output import prepare_main_attention, selection_trajectories, vector_metrics
from temporal_dsa.metrics import stable_topk
from temporal_dsa.trace import load_trace
from temporal_dsa.v2_collect import MODEL_REVISION, fixed_device_map


def discover(files: list[Path] | None, roots: list[Path] | None) -> list[Path]:
    paths = list(files or [])
    for root in roots or []:
        paths.extend(sorted(root.rglob("*.npz")))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("no score traces")
    return paths


@torch.no_grad()
def attention_output(module, logits, values, ids: np.ndarray) -> torch.Tensor:
    selected = torch.as_tensor(ids, device=logits.device, dtype=torch.long)
    probabilities = torch.softmax(logits[:, selected].float(), dim=-1).to(values.dtype)
    heads = torch.matmul(probabilities.unsqueeze(1), values[:, selected]).squeeze(1)
    return module.o_proj(heads.reshape(1, 1, -1)).squeeze()


def rerank_with_blocks(
    scores: np.ndarray,
    evaluated: np.ndarray,
    blocks: np.ndarray,
    *,
    block_size: int,
    k: int,
) -> np.ndarray:
    mask = np.zeros(scores.size, dtype=bool)
    mask[evaluated] = True
    for block_id in blocks:
        start = int(block_id) * block_size
        mask[start : min(scores.size, start + block_size)] = True
    candidates = np.flatnonzero(mask)
    return candidates[stable_topk(scores[candidates], k)]


@torch.no_grad()
def evaluate_trace(
    model,
    trace_path: Path,
    specs: list[dict[str, Any]],
    scales: dict[int, float],
    k: int,
    block_size: int,
    max_transitions: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace = load_trace(trace_path)
    details, _ = selection_trajectories(
        trace, specs, scales, k, block_size, max_transitions
    )
    capture = torch.load(trace.metadata["source_capture"], map_location="cpu", weights_only=True)
    hidden = capture["hidden"]
    steps = sorted(next(iter(details.values())))
    positions = np.asarray([int(trace.lengths[step]) - 1 for step in steps], dtype=np.int64)
    module = model.model.layers[trace.layer].self_attn
    apply_rotary = module.__class__.forward.__globals__["apply_rotary_pos_emb"]
    queries, keys, values = prepare_main_attention(module, hidden, positions, apply_rotary)
    output_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for local_step, step in enumerate(steps):
        length = int(trace.lengths[step])
        logits = torch.einsum("hd,hkd->hk", queries[:, local_step], keys[:, :length])
        logits = logits * float(module.softmax_scale)
        current_scores = trace.row(step)
        for spec in specs:
            name = str(spec["config_id"])
            detail = details[name][step]
            baseline = detail["baseline"]
            temporal = detail["approximate"]
            evaluated = detail["evaluated"]
            missed = np.setdiff1d(baseline, temporal, assume_unique=False)
            missed_blocks = np.unique(missed // block_size)
            base_output = attention_output(module, logits, values[:, :length], baseline)
            temporal_output = attention_output(module, logits, values[:, :length], temporal)
            _, temporal_l2, _ = vector_metrics(base_output, temporal_output)

            individual: list[tuple[float, int, float]] = []
            for block_id in missed_blocks:
                rescued = rerank_with_blocks(
                    current_scores,
                    evaluated,
                    np.asarray([block_id]),
                    block_size=block_size,
                    k=k,
                )
                rescued_output = attention_output(module, logits, values[:, :length], rescued)
                _, rescued_l2, _ = vector_metrics(base_output, rescued_output)
                individual.append((temporal_l2 - rescued_l2, int(block_id), rescued_l2))
            individual.sort(key=lambda item: (-item[0], item[1]))
            ranked_blocks = np.asarray([item[1] for item in individual], dtype=np.int64)
            for rank, (gain, block_id, individual_l2) in enumerate(individual, start=1):
                label_rows.append(
                    {
                        "policy": name,
                        "policy_role": spec["selection_role"],
                        "step": step,
                        "layer": trace.layer,
                        "prompt_id": trace.prompt_id,
                        "workload": trace.workload,
                        "base_context_length": int(trace.lengths[0]) - 1,
                        "block_id": block_id,
                        "gain_rank": rank,
                        "individual_gain": gain,
                        "individual_relative_l2": individual_l2,
                        "gain_at_least_1pp": gain >= 0.01,
                        "gain_at_least_25pct": gain >= 0.25 * temporal_l2,
                        "top1": rank <= 1,
                        "top2": rank <= 2,
                        "top4": rank <= 4,
                    }
                )
            budgets: list[tuple[str, np.ndarray]] = [("0", np.empty(0, dtype=np.int64))]
            for count in (1, 2, 4, 8):
                budgets.append((f"best{count}", ranked_blocks[:count]))
            budgets.append(("all", ranked_blocks))
            for rescue_name, block_ids in budgets:
                selected = (
                    temporal
                    if rescue_name == "0"
                    else rerank_with_blocks(
                        current_scores,
                        evaluated,
                        block_ids,
                        block_size=block_size,
                        k=k,
                    )
                )
                output = attention_output(module, logits, values[:, :length], selected)
                cosine, relative_l2, max_abs = vector_metrics(base_output, output)
                output_rows.append(
                    {
                        "policy": name,
                        "policy_role": spec["selection_role"],
                        "rescue": rescue_name,
                        "rescue_blocks": len(block_ids),
                        "missed_blocks": len(missed_blocks),
                        "step": step,
                        "layer": trace.layer,
                        "prompt_id": trace.prompt_id,
                        "workload": trace.workload,
                        "base_context_length": int(trace.lengths[0]) - 1,
                        "context_length": length,
                        "output_cosine": cosine,
                        "output_relative_l2": relative_l2,
                        "output_max_abs": max_abs,
                        "top128_recall": np.isin(baseline[:128], selected).mean(),
                        "top512_recall": np.isin(baseline[:512], selected).mean(),
                        "top2048_recall": np.isin(baseline, selected).mean(),
                    }
                )
    del hidden, queries, keys, values
    torch.cuda.empty_cache()
    return output_rows, label_rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["policy_role", "policy", "rescue"], as_index=False).agg(
        observations=("output_relative_l2", "size"),
        rescue_blocks_mean=("rescue_blocks", "mean"),
        missed_blocks_mean=("missed_blocks", "mean"),
        output_cosine_p5=("output_cosine", lambda x: x.quantile(0.05)),
        output_relative_l2_p50=("output_relative_l2", "median"),
        output_relative_l2_p90=("output_relative_l2", lambda x: x.quantile(0.90)),
        output_relative_l2_p95=("output_relative_l2", lambda x: x.quantile(0.95)),
        output_relative_l2_p99=("output_relative_l2", lambda x: x.quantile(0.99)),
        output_relative_l2_max=("output_relative_l2", "max"),
        top128_recall=("top128_recall", "mean"),
        top512_recall=("top512_recall", "mean"),
        top2048_recall=("top2048_recall", "mean"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle block rescue ceiling")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--trace-files", type=Path, nargs="+")
    parser.add_argument("--trace-roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-roles", nargs="+", default=["Balanced", "Aggressive"])
    parser.add_argument("--max-transitions", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM

    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected two visible GPUs, found {torch.cuda.device_count()}")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("L40" in value for value in gpu_names):
        raise RuntimeError(f"L40 guard rejected {gpu_names}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    specs = [
        value for value in selection["policies"] if value["selection_role"] in args.policy_roles
    ]
    scales = {int(key): float(value) for key, value in selection["gamma_scale"].items()}
    paths = discover(args.trace_files, args.trace_roots)
    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=fixed_device_map(config.num_hidden_layers),
        attn_implementation="eager",
    ).eval()
    started = time.perf_counter()
    output_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        outputs, labels = evaluate_trace(
            model,
            path,
            specs,
            scales,
            int(selection["k"]),
            int(selection["block_size"]),
            args.max_transitions,
        )
        output_rows.extend(outputs)
        label_rows.extend(labels)
        print(f"[{index}/{len(paths)}] {path}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(output_rows)
    labels = pd.DataFrame(label_rows)
    frame.to_csv(args.output / "oracle_rescue_rows.csv", index=False)
    labels.to_csv(args.output / "tail_critical_labels.csv", index=False)
    summarize(frame).to_csv(args.output / "oracle_rescue_ceiling.csv", index=False)
    audit = {
        "command": " ".join(__import__("sys").argv),
        "elapsed_seconds": time.perf_counter() - started,
        "gpus": gpu_names,
        "visible_gpu_ids": [0, 1],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "trace_count": len(paths),
        "max_transitions": args.max_transitions,
        "model_revision": args.revision,
    }
    (args.output / "oracle_rescue_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summarize(frame).to_string(index=False))


if __name__ == "__main__":
    main()
