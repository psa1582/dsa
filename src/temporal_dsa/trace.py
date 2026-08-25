from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScoreTrace:
    """One layer/prompt trace; row t contains scores for keys [0, lengths[t])."""

    scores: np.ndarray
    lengths: np.ndarray
    layer: int
    workload: str
    prompt_id: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.scores.ndim != 2:
            raise ValueError(f"scores must be [steps,max_length], got {self.scores.shape}")
        if self.lengths.shape != (self.scores.shape[0],):
            raise ValueError("lengths must have one entry per step")
        if np.any(self.lengths <= 0) or np.any(self.lengths > self.scores.shape[1]):
            raise ValueError("lengths are outside the score matrix")

    def row(self, step: int) -> np.ndarray:
        return np.asarray(self.scores[step, : self.lengths[step]], dtype=np.float32)


def save_trace(path: str | Path, trace: ScoreTrace) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **trace.metadata,
        "layer": trace.layer,
        "workload": trace.workload,
        "prompt_id": trace.prompt_id,
    }
    np.savez_compressed(
        path,
        scores=trace.scores,
        lengths=trace.lengths,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_trace(path: str | Path) -> ScoreTrace:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        return ScoreTrace(
            scores=data["scores"],
            lengths=data["lengths"],
            layer=int(metadata.pop("layer")),
            workload=str(metadata.pop("workload")),
            prompt_id=str(metadata.pop("prompt_id")),
            metadata=metadata,
        )

