from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


BOOKS = [
    (10321, "calibration", "Dragon's Blood"),
    (10356, "calibration", "Travels in Morocco Volume 2"),
    (10762, "calibration", "Impressions of Theophrastus Such"),
    (22424, "calibration", "Frank Merriwell Down South"),
    (2544, "validation", "From Sand Hill to Pine"),
    (25773, "validation", "Portraits of English Authors on Gardening"),
    (26183, "validation", "Laurence Sterne in Germany"),
    (26239, "validation", "The Forester's Daughter"),
    (26493, "heldout", "The Life of Gordon Volume II"),
    (27454, "heldout", "In Her Own Right"),
    (28444, "heldout", "Turn About Eleanor"),
    (28988, "heldout", "Jennie Gerhardt"),
]


def download_books(root: Path) -> dict[int, str]:
    root.mkdir(parents=True, exist_ok=True)
    texts = {}
    for book_id, _, _ in BOOKS:
        path = root / f"gutenberg_{book_id}.txt"
        if not path.exists():
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        texts[book_id] = path.read_text(encoding="utf-8-sig", errors="replace")
    return texts


def code_documents(code_root: Path, count: int, target_chars: int) -> list[tuple[str, str]]:
    paths = sorted(
        path
        for path in code_root.rglob("*.py")
        if "site-packages" not in path.parts and path.stat().st_size < 1_000_000
    )
    buckets = [[] for _ in range(count)]
    sizes = [0] * count
    used = set()
    for path in paths:
        bucket = min(range(count), key=sizes.__getitem__)
        if sizes[bucket] >= target_chars:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        relative = str(path.relative_to(code_root))
        buckets[bucket].append(f"\n# ===== FILE: {relative} =====\n{content}")
        sizes[bucket] += len(content)
        used.add(path)
        if all(size >= target_chars for size in sizes):
            break
    if any(size < target_chars for size in sizes):
        raise RuntimeError(f"not enough code under {code_root}: bucket sizes={sizes}")
    return [(f"{len(bucket)} disjoint Python stdlib files", "".join(bucket)) for bucket in buckets]


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    books = download_books(args.output / "source_text")
    code = code_documents(args.code_root, count=6, target_chars=args.code_chars)
    by_split = {split: [] for split in ("calibration", "validation", "heldout")}
    for book_id, split, title in BOOKS:
        by_split[split].append(
            {
                "id": f"text_{split}_{book_id}",
                "workload": "long-text",
                "text": books[book_id],
                "source": f"Project Gutenberg ebook {book_id}: {title}",
                "split": split,
            }
        )
    code_rows = []
    code_splits = ["calibration", "calibration", "validation", "heldout", "heldout", "heldout"]
    for code_id, ((source, text), split) in enumerate(zip(code, code_splits, strict=True)):
        code_rows.append(
            {
                "id": f"code_{split}_{code_id}",
                "workload": "long-code",
                "text": text,
                "source": f"Python stdlib: {source}",
                "split": split,
            }
        )
    warmup = by_split["calibration"] + [row for row in code_rows if row["split"] == "calibration"]
    quality = by_split["validation"][:2] + [row for row in code_rows if row["split"] == "validation"]
    trace = by_split["heldout"][:3] + [row for row in code_rows if row["split"] == "heldout"]
    write_jsonl(args.output / "warmup.jsonl", warmup)
    write_jsonl(args.output / "quality.jsonl", quality)
    write_jsonl(args.output / "trace.jsonl", trace)
    manifest = {
        "warmup_prompts": len(warmup),
        "quality_prompts": len(quality),
        "trace_prompts": len(trace),
        "book_ids": [book_id for book_id, _, _ in BOOKS],
        "code_root": str(args.code_root),
        "code_target_chars": args.code_chars,
        "split_overlap": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path("/usr/lib/python3.10"))
    parser.add_argument("--code-chars", type=int, default=180_000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

