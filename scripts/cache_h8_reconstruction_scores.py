from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from temporal_dsa.metrics import stable_topk
from temporal_dsa.verifier_scoring import (
    dynamic_head_indices,
    load_sidecar_encoded,
    score_head_sparse,
)


def cache_name(output: Path, split: str, trace_file: str) -> Path:
    return output / split / Path(trace_file).name


def reconstruct(trace_file: Path, checkpoint: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(trace_file, allow_pickle=False) as payload:
        lengths = payload["lengths"].astype(np.int32)
        metadata = json.loads(str(payload["metadata"].item()))
    queries, weights, keys, _ = load_sidecar_encoded(
        metadata["source_capture"], checkpoint, lengths, device="cpu"
    )
    head_ids = dynamic_head_indices(weights, 8, "high_weight")
    h8 = score_head_sparse(
        queries,
        weights,
        keys,
        head_ids,
        query_chunk_size=1,
        key_chunk_size=4096,
    )
    selected_mass = torch.gather(weights.abs(), 1, head_ids).sum(dim=1)
    rho = selected_mass / weights.abs().sum(dim=1).clamp_min(1e-12)
    result = (
        h8.cpu().numpy().astype(np.float32),
        rho.cpu().numpy().astype(np.float32),
        head_ids.cpu().numpy().astype(np.int16),
        metadata,
    )
    del queries, weights, keys, h8, rho, head_ids
    return result


def numeric_audit(
    trace_file: Path,
    checkpoint: Path,
    steps: list[int],
) -> list[dict]:
    with np.load(trace_file, allow_pickle=False) as payload:
        lengths = payload["lengths"].astype(np.int32)
        reference = payload["scores"].astype(np.float32)
        metadata = json.loads(str(payload["metadata"].item()))
    queries, weights, keys, _ = load_sidecar_encoded(
        metadata["source_capture"], checkpoint, lengths, device="cpu"
    )
    rows = []
    all_heads = torch.arange(64, dtype=torch.long)
    for step in steps:
        length = int(lengths[step])
        replay = score_head_sparse(
            queries[step : step + 1],
            weights[step : step + 1],
            keys[:length],
            all_heads,
            query_chunk_size=1,
            key_chunk_size=4096,
        )[0].cpu().numpy()
        target = reference[step, :length]
        expected_ids = stable_topk(target, min(2048, length))
        actual_ids = stable_topk(replay, min(2048, length))
        rows.append(
            {
                "trace_file": str(trace_file),
                "layer": int(metadata["layer"]),
                "step": int(step),
                "length": length,
                "max_abs_error": float(np.max(np.abs(replay - target))),
                "mean_abs_error": float(np.mean(np.abs(replay - target))),
                "pearson": float(np.corrcoef(replay, target)[0, 1]),
                "top2048_recall": float(np.isin(expected_ids, actual_ids).mean()),
            }
        )
    del queries, weights, keys
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only exact dynamic-H8 score reconstruction")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be hidden for the strict offline cache pass")
    torch.set_num_threads(args.threads)
    inventory = pd.read_csv(args.inventory)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    audited_layers: set[int] = set()
    audit_rows: list[dict] = []
    for index, row in enumerate(inventory.itertuples(), start=1):
        destination = cache_name(args.output, row.split, row.trace_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        trace_file = Path(row.trace_file)
        checkpoint = Path(row.checkpoint)
        item_started = time.perf_counter()
        if destination.exists() and not args.force:
            status = "reused"
        else:
            h8, rho, head_ids, metadata = reconstruct(trace_file, checkpoint)
            np.savez_compressed(
                destination,
                h8=h8,
                rho=rho,
                head_ids=head_ids,
                lengths=np.asarray(
                    np.load(trace_file, allow_pickle=False)["lengths"], dtype=np.int32
                ),
                metadata=np.asarray(
                    json.dumps(
                        {
                            "trace_file": str(trace_file),
                            "source_capture": metadata["source_capture"],
                            "checkpoint": str(checkpoint),
                            "model_revision": metadata["model_revision"],
                            "formula": "sum dynamic Top8-|w| heads of w*ReLU(q dot k), D=128",
                            "runtime_oracle_inputs": False,
                        },
                        sort_keys=True,
                    )
                ),
            )
            status = "created"
        if row.split == "calibration" and int(row.layer) not in audited_layers:
            with np.load(trace_file, allow_pickle=False) as payload:
                last = int(payload["lengths"].shape[0]) - 1
            audit_rows.extend(numeric_audit(trace_file, checkpoint, [0, last]))
            audited_layers.add(int(row.layer))
        rows.append(
            {
                "trace_file": str(trace_file),
                "cache_file": str(destination),
                "split": row.split,
                "layer": int(row.layer),
                "status": status,
                "seconds": time.perf_counter() - item_started,
            }
        )
        print(f"[{index}/{len(inventory)}] {status} {trace_file}", flush=True)

    pd.DataFrame(rows).to_csv(args.output / "cache_manifest.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.output / "cpu_full64_numeric_audit.csv", index=False)
    audit = {
        "command": " ".join(__import__("sys").argv),
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "gpu_used": False,
        "model_inference_run": False,
        "threads": args.threads,
        "trace_count": len(inventory),
    }
    (args.output / "cache_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
