from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from .replay import block_temporal_stats, replay_trace, temporal_stats
from .reporting import create_required_graphs, hardware_extrapolation, choose_verdict, write_report
from .trace import load_trace


def _process_trace(path: Path, replay_config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trace = load_trace(path)
    temporal = temporal_stats(trace, replay_config["k_values"])
    blocks = block_temporal_stats(trace, replay_config["block_sizes"], replay_config["k_values"])
    replay = replay_trace(
        trace,
        k_values=replay_config["k_values"],
        block_sizes=replay_config["block_sizes"],
        gamma_sigma_values=replay_config["gamma_sigma"],
        key_bytes=replay_config["key_bytes"],
        metadata_bytes_per_block=replay_config["metadata_bytes_per_block"],
        absolute_margins=replay_config.get("calibrated_margins", {}).get(str(trace.layer)),
    )
    return temporal, blocks, replay


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def analyze(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.margin_file:
        margin_payload = json.loads(args.margin_file.read_text(encoding="utf-8"))
        config["replay"]["calibrated_margins"] = {
            str(layer): {int(block): value for block, value in blocks.items()}
            for layer, blocks in margin_payload["margins"].items()
        }
    trace_paths = []
    for root in args.trace_roots:
        trace_paths.extend([root] if root.is_file() and root.suffix == ".npz" else sorted(root.rglob("*.npz")))
    if not trace_paths:
        raise FileNotFoundError("no .npz traces found")
    args.output.mkdir(parents=True, exist_ok=True)
    csv_paths = [
        args.output / "temporal_stats.csv",
        args.output / "block_stats.csv",
        args.output / "replay_rows.csv",
    ]
    if args.reuse_csv and all(path.exists() for path in csv_paths):
        temporal, blocks, replay = (pd.read_csv(path) for path in csv_paths)
    else:
        if args.workers == 1:
            pieces = [_process_trace(path, config["replay"]) for path in trace_paths]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                pieces = list(
                    executor.map(
                        _process_trace,
                        trace_paths,
                        [config["replay"]] * len(trace_paths),
                    )
                )
        temporal = pd.concat([piece[0] for piece in pieces], ignore_index=True)
        blocks = pd.concat([piece[1] for piece in pieces], ignore_index=True)
        replay = pd.concat([piece[2] for piece in pieces], ignore_index=True)
        temporal.to_csv(csv_paths[0], index=False)
        blocks.to_csv(csv_paths[1], index=False)
        replay.to_csv(csv_paths[2], index=False)
    quality = json.loads(args.quality_gate.read_text(encoding="utf-8"))
    verdict, numbers = choose_verdict(replay, quality, config["verdict"])
    graphs = create_required_graphs(temporal, blocks, replay, args.output / "graphs")
    hardware = hardware_extrapolation(
        replay,
        [32768, 65536, 131072],
        config["replay"]["block_sizes"],
        config["replay"]["key_bytes"],
        config["replay"]["metadata_bytes_per_block"],
    )
    hardware.to_csv(args.output / "hardware_extrapolation.csv", index=False)
    audit = {
        "git_commit": _git_revision(),
        "config": str(args.config),
        "trace_roots": [str(path) for path in args.trace_roots],
        "quality_gate": str(args.quality_gate),
        "trace_count": len(trace_paths),
    }
    write_report(args.output, verdict, numbers, quality, graphs, hardware, audit)
    (args.output / "verdict.json").write_text(
        json.dumps({"verdict": verdict, "numbers": numbers}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="temporal-dsa")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze", help="replay traces and produce the report")
    analyze_parser.add_argument("--trace-roots", type=Path, nargs="+", required=True)
    analyze_parser.add_argument("--quality-gate", type=Path, required=True)
    analyze_parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--reuse-csv", action="store_true")
    analyze_parser.add_argument("--workers", type=int, default=1)
    analyze_parser.add_argument("--margin-file", type=Path)
    analyze_parser.set_defaults(function=analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
