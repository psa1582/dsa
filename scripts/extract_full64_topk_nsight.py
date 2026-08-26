from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def short_stage(name: str) -> tuple[str, str]:
    if name == "_full_score_kernel":
        return "score", "Full64 score"
    if name == "fill":
        return "topk_aux", "initialize Top-K workspace"
    if name == "radixFindKthValues":
        return "topk_radix", "radix threshold refinement"
    if name == "computeBlockDigitCounts":
        return "topk_radix", "radix digit histogram"
    if name == "computeDigitCumSum":
        return "topk_radix", "radix digit cumulative sum"
    if name == "computeBlockwiseWithinKCounts":
        return "topk_count", "blockwise within-K count"
    if name == "computeBlockwiseKthCounts":
        return "topk_count", "blockwise Kth count"
    if name == "DeviceScanByKeyInitKernel":
        return "topk_scan", "scan-by-key initialization"
    if name == "DeviceScanByKeyKernel":
        return "topk_scan", "scan-by-key"
    if name == "gatherTopK":
        return "topk_gather", "final Top-K gather"
    if name == "cudaMemset":
        return "memset", "Top-K workspace memset"
    return "other", name


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def extract(path: Path, context: int) -> tuple[list[dict], list[dict]]:
    connection = sqlite3.connect(path)
    range_names = (f"L2_full64_topk_c{context}", f"optimized_full_c{context}")
    range_row = None
    range_name = ""
    for candidate in range_names:
        range_row = connection.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE text=?", (candidate,)
        ).fetchone()
        if range_row is not None:
            range_name = candidate
            break
    if range_row is None:
        raise RuntimeError(f"none of the expected NVTX ranges {range_names} exist in {path}")
    start, end = range_row
    score_starts = [
        row[0]
        for row in connection.execute(
            """
            SELECT k.start
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id=k.shortName
            WHERE s.value='_full_score_kernel' AND k.start>=? AND k.end<=?
            ORDER BY k.start
            """,
            (start, end),
        )
    ]
    iterations = len(score_starts)
    if iterations < 2:
        raise RuntimeError(f"expected repeated score launches in {path}")
    first_start, second_start = score_starts[:2]
    events = list(
        connection.execute(
            """
            SELECT k.start, k.end, s.value, 'kernel' AS operation_type, NULL AS bytes
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id=k.shortName
            WHERE k.start>=? AND k.start<?
            ORDER BY k.start
            """,
            (first_start, second_start),
        )
    )
    if table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMSET"):
        events.extend(
            connection.execute(
                """
                SELECT start, end, 'cudaMemset', 'memset', bytes
                FROM CUPTI_ACTIVITY_KIND_MEMSET
                WHERE start>=? AND start<?
                """,
                (first_start, second_start),
            )
        )
    if table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        events.extend(
            connection.execute(
                """
                SELECT start, end, 'cudaMemcpy', 'memcpy', bytes
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                WHERE start>=? AND start<?
                """,
                (first_start, second_start),
            )
        )
    events.sort(key=lambda row: row[0])
    timeline = []
    for order, (event_start, event_end, name, operation_type, byte_count) in enumerate(
        events, start=1
    ):
        category, stage = short_stage(name)
        timeline.append(
            {
                "context": context,
                "operation_order": order,
                "operation_type": operation_type,
                "category": category,
                "stage": stage,
                "short_name": name,
                "duration_us_first_iteration": (event_end - event_start) / 1000.0,
                "gap_from_previous_us_first_iteration": (
                    0.0
                    if order == 1
                    else (event_start - events[order - 2][1]) / 1000.0
                ),
                "bytes": byte_count,
                "source": str(path),
                "nvtx_range": range_name,
            }
        )

    aggregate_rows = []
    kernel_query = """
        SELECT s.value, COUNT(*), SUM(k.end-k.start)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id=k.shortName
        WHERE k.start>=? AND k.end<=?
        GROUP BY s.value
    """
    aggregates = list(connection.execute(kernel_query, (start, end)))
    if table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMSET"):
        row = connection.execute(
            """
            SELECT 'cudaMemset', COUNT(*), SUM(end-start)
            FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start>=? AND end<=?
            """,
            (start, end),
        ).fetchone()
        if row[1]:
            aggregates.append(row)
    if table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        row = connection.execute(
            """
            SELECT 'cudaMemcpy', COUNT(*), SUM(end-start)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=?
            """,
            (start, end),
        ).fetchone()
        if row[1]:
            aggregates.append(row)
    for name, count, total_ns in aggregates:
        category, stage = short_stage(name)
        aggregate_rows.append(
            {
                "context": context,
                "iterations": iterations,
                "operation_type": "memset" if name == "cudaMemset" else (
                    "memcpy" if name == "cudaMemcpy" else "kernel"
                ),
                "category": category,
                "stage": stage,
                "short_name": name,
                "launch_count_total": count,
                "launches_per_iteration": count / iterations,
                "active_duration_us_per_iteration": total_ns / iterations / 1000.0,
                "nvtx_wall_us_per_iteration": (end - start) / iterations / 1000.0,
                "source": str(path),
                "nvtx_range": range_name,
            }
        )
    connection.close()
    return timeline, aggregate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract exact Full64+Top-K Nsight timeline")
    parser.add_argument("--sqlite-16k", type=Path, required=True)
    parser.add_argument("--sqlite-32k", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timeline_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for path, context in [(args.sqlite_16k, 16384), (args.sqlite_32k, 32768)]:
        timeline, aggregate = extract(path, context)
        timeline_rows.extend(timeline)
        aggregate_rows.extend(aggregate)
    pd.DataFrame(timeline_rows).to_csv(args.output / "gpu_topk_timeline.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(
        args.output / "gpu_topk_operation_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
