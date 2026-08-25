from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    rows = pd.read_csv(args.csv)
    summary = (
        rows.groupby(["layer", "workload", "k"])[["recall", "normalized_lift"]]
        .agg(["mean", "median", "min"])
        .reset_index()
    )
    summary.columns = ["_".join(str(part) for part in column if part) for column in summary.columns]
    print(json.dumps(summary.to_dict(orient="records"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

