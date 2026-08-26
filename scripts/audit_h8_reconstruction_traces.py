from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VALIDATION_PROMPTS = {"code_heldout_4", "text_heldout_27454"}


def trace_metadata(path: Path) -> tuple[dict[str, Any], tuple[int, ...], tuple[int, ...]]:
    with np.load(path, allow_pickle=False) as payload:
        if not {"scores", "lengths", "metadata"}.issubset(payload.files):
            raise ValueError(f"not a score trace: {path}")
        metadata = json.loads(str(payload["metadata"].item()))
        return metadata, tuple(payload["scores"].shape), tuple(payload["lengths"].shape)


def discover(root: Path) -> list[Path]:
    return sorted(root.rglob("*.npz"))


def split_name(metadata: dict[str, Any]) -> str:
    if metadata.get("split") == "validation":
        return "calibration"
    return "validation" if metadata["prompt_id"] in VALIDATION_PROMPTS else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the locked H8 reconstruction traces")
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--heldout-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = discover(args.calibration_root)
    for root in args.heldout_roots:
        paths.extend(discover(root))
    paths = sorted(set(paths))
    rows: list[dict[str, Any]] = []
    for path in paths:
        metadata, score_shape, length_shape = trace_metadata(path)
        source_capture = Path(metadata["source_capture"])
        checkpoint = args.checkpoint_root / f"layer_{int(metadata['layer']):02d}.safetensors"
        if not source_capture.is_file():
            raise FileNotFoundError(source_capture)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        with np.load(path, allow_pickle=False) as payload:
            lengths = payload["lengths"]
        rows.append(
            {
                "split": split_name(metadata),
                "trace_file": str(path),
                "source_capture": str(source_capture),
                "checkpoint": str(checkpoint),
                "prompt_id": metadata["prompt_id"],
                "workload": metadata["workload"],
                "layer": int(metadata["layer"]),
                "model_revision": metadata["model_revision"],
                "base_context_length": int(lengths[0]) - 1,
                "decode_steps": int(score_shape[0]),
                "score_rows": int(score_shape[0]),
                "score_columns": int(score_shape[1]),
                "length_rows": int(length_shape[0]),
                "score_dtype": "float16",
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        ["split", "prompt_id", "base_context_length", "layer"]
    )
    expected_layers = [2, 5, 8, 11, 14, 17, 21, 25]
    if sorted(frame.layer.unique().tolist()) != expected_layers:
        raise RuntimeError("layer coverage changed")
    if frame.model_revision.nunique() != 1:
        raise RuntimeError("multiple model revisions found")
    expected_counts = {"calibration": 24, "validation": 48, "test": 96}
    actual_counts = frame.groupby("split").size().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError(f"unexpected split counts: {actual_counts}")
    heldout_contexts = sorted(
        frame.loc[frame.split.ne("calibration"), "base_context_length"].unique().tolist()
    )
    if heldout_contexts != [8192, 16384, 32768]:
        raise RuntimeError(f"heldout contexts changed: {heldout_contexts}")

    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "trace_inventory.csv", index=False)
    split_manifest = {
        split: sorted(group.prompt_id.unique().tolist())
        for split, group in frame.groupby("split", sort=False)
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    setup = {
        "experiment": "strict_offline_h8_full_score_reconstruction",
        "model": "deepseek-ai/DeepSeek-V2-Lite-Chat research sidecar",
        "model_revision": frame.model_revision.iloc[0],
        "topk": 2048,
        "block_size": 64,
        "indexer_heads": 64,
        "selected_heads": 8,
        "head_dimension": 128,
        "layers": expected_layers,
        "heldout_context_lengths": heldout_contexts,
        "calibration_context_lengths": sorted(
            frame.loc[frame.split.eq("calibration"), "base_context_length"].unique().tolist()
        ),
        "decode_steps_by_split": {
            split: sorted(group.decode_steps.unique().tolist())
            for split, group in frame.groupby("split")
        },
        "trace_counts": {key: int(value) for key, value in actual_counts.items()},
        "observation_shapes": sorted(
            {
                (int(row.score_rows), int(row.score_columns))
                for row in frame.itertuples()
            }
        ),
        "split_method": (
            "locked prior validation traces for calibration; sequence-level heldout split: "
            "code_heldout_4/text_heldout_27454 for model selection, all other heldout "
            "sequences for final test"
        ),
        "model_inference_run": False,
        "gpu_used": False,
    }
    (args.output / "trace_setup.json").write_text(
        json.dumps(setup, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(setup, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
