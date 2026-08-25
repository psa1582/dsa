from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch

from .training import _new_indexer, discover_captures, evaluate_and_trace


def _layer_from_name(path: Path) -> int:
    match = re.fullmatch(r"layer_(\d+)\.safetensors", path.name)
    if not match:
        raise ValueError(f"unexpected checkpoint name: {path}")
    return int(match.group(1))


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("trace scoring requires CUDA")
    from safetensors.torch import load_file

    device = torch.device(args.device)
    captures = discover_captures(args.capture_roots)
    checkpoints = sorted(args.checkpoint_root.glob("layer_*.safetensors"))
    if not checkpoints:
        raise FileNotFoundError(f"no sidecar checkpoints in {args.checkpoint_root}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    scored_layers = []
    for checkpoint in checkpoints:
        layer = _layer_from_name(checkpoint)
        if args.layers and layer not in args.layers:
            continue
        if layer not in captures:
            raise ValueError(f"layer {layer} has no captures")
        indexer = _new_indexer(args, device)
        indexer.load_state_dict(load_file(checkpoint, device=str(device)))
        indexer.eval()
        rows.extend(
            evaluate_and_trace(
                indexer,
                captures[layer],
                layer,
                args,
                device,
                args.output / "traces",
            )
        )
        scored_layers.append(layer)
        del indexer
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(args.output / "teacher_recall_rows.csv", index=False)
    summary = {
        "checkpoint_root": str(args.checkpoint_root),
        "capture_roots": [str(path) for path in args.capture_roots],
        "layers": scored_layers,
        "eval_queries": args.eval_queries,
        "score_dtype": "float16-on-disk",
        "forward_dtype": "bfloat16-autocast-with-fp32-master-weights",
    }
    (args.output / "scoring_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score held-out captures with frozen sidecars")
    parser.add_argument("--capture-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--key-chunk-size", type=int, default=4096)
    parser.add_argument("--eval-queries", type=int, default=256)
    parser.add_argument("--k-values", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

