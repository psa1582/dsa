from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    rows = []
    for path in sorted(args.root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                prompt = json.loads(line)
                tokens = len(tokenizer(prompt["text"], add_special_tokens=True)["input_ids"])
                rows.append(
                    {
                        "file": path.name,
                        "id": prompt["id"],
                        "workload": prompt["workload"],
                        "tokens": tokens,
                    }
                )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

