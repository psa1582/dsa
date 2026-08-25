from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    args = parser.parse_args()
    rows = []
    for root in args.roots:
        payload = json.loads((root / "collection_summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "root": str(root),
                "prompts": len(payload),
                "prefill_seconds": sum(row["prefill_seconds"] for row in payload),
                "decode_seconds": sum(row["decode_seconds"] for row in payload),
                "decode_steps": sum(len(row["generated_token_ids"]) for row in payload),
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

