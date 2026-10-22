#!/usr/bin/env python3
"""Analyze a HiResFlow run-dir summary and emit CSV plus Markdown reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered))
    return ordered[min(len(ordered) - 1, max(0, rank - 1))]


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def index_path(mask_path: Path) -> Path:
    suffix = ".mask.png"
    text = str(mask_path)
    return Path(text[: -len(suffix)] + ".index.png") if text.endswith(suffix) else mask_path


def tile_bucket(tiles: int) -> str:
    if tiles == 1:
        return "1"
    if tiles <= 4:
        return "2-4"
    if tiles <= 9:
        return "5-9"
    if tiles <= 16:
        return "10-16"
    return "17+"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    document: dict[str, Any] = json.loads(args.summary.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for image in document["images"]:
        if not image["success"]:
            continue
        input_path = Path(image["input"])
        output_path = Path(image["output"])
        with Image.open(input_path) as source:
            width, height = source.size
        classes = ""
        dominant_fraction = ""
        mask_path = index_path(output_path)
        if mask_path.is_file():
            with Image.open(mask_path) as mask:
                counts = Counter(mask.getdata())
                classes = len(counts)
                dominant_fraction = round(max(counts.values()) / (mask.width * mask.height), 6)
        tiles = int(image["tiles"])
        rows.append(
            {
                **image,
                "width": width,
                "height": height,
                "classes_present": classes,
                "dominant_fraction": dominant_fraction,
                "total_per_tile_ms": image["total_ms"] / max(1, tiles),
                "inference_per_tile_ms": image["inference_ms"] / max(1, tiles),
            }
        )

    columns = [
        "input",
        "width",
        "height",
        "tiles",
        "batch_size",
        "batch_retries",
        "classes_present",
        "dominant_fraction",
        "load_ms",
        "preprocess_ms",
        "inference_ms",
        "merge_ms",
        "write_ms",
        "total_ms",
        "total_per_tile_ms",
        "inference_per_tile_ms",
        "gpu_peak_used_bytes",
    ]
    csv_path = Path(str(args.output_prefix) + ".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    total_values = [float(row["total_ms"]) for row in rows]
    inference_values = [float(row["inference_ms"]) for row in rows]
    preprocess_values = [float(row["preprocess_ms"]) for row in rows]
    merge_values = [float(row["merge_ms"]) for row in rows]
    write_values = [float(row["write_ms"]) for row in rows]
    tile_values = [float(row["tiles"]) for row in rows]
    correlation = pearson(tile_values, total_values)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tile_bucket(int(row["tiles"]))].append(row)

    slowest = sorted(rows, key=lambda row: row["total_ms"], reverse=True)[:10]
    per_tile_outliers = sorted(
        rows,
        key=lambda row: row["total_per_tile_ms"],
        reverse=True,
    )[:10]
    retries = sum(int(row["batch_retries"]) for row in rows)
    single_class = sum(
        1 for row in rows if str(row["classes_present"]) == "1"
    )
    class_counts = Counter(str(row["classes_present"]) for row in rows)

    markdown: list[str] = [
        "# Screenshot batch analysis",
        "",
        f"- Images: {document['total_images']}",
        f"- Successful: {document['successful_images']}",
        f"- Failed: {document['failed_images']}",
        f"- Warmup runs: {document.get('warmup_runs', 0)}",
        f"- Tiles: {document['total_tiles']}",
        f"- Wall time: {document['wall_time_ms']:.3f} ms",
        f"- Image throughput: {len(rows) / (document['wall_time_ms'] / 1000):.2f} images/s",
        f"- Tile throughput: {document['total_tiles'] / (document['wall_time_ms'] / 1000):.2f} tiles/s",
        f"- Total latency P50/P95: {percentile(total_values, 0.50):.3f} / "
        f"{percentile(total_values, 0.95):.3f} ms",
        f"- Inference P50/P95: {percentile(inference_values, 0.50):.3f} / "
        f"{percentile(inference_values, 0.95):.3f} ms",
        f"- Pearson correlation, tile count vs total latency: {correlation:.4f}",
        f"- Batch retries: {retries}",
        f"- Single-class masks: {single_class}",
        f"- Mean preprocessing: {statistics.fmean(preprocess_values):.3f} ms",
        f"- Mean merge: {statistics.fmean(merge_values):.3f} ms",
        f"- Mean output write: {statistics.fmean(write_values):.3f} ms",
        "",
        "## Tile-count buckets",
        "",
        "| Tiles | Images | Mean total ms | P95 total ms | Mean inference ms | Mean merge ms | Mean write ms | Mean classes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["1", "2-4", "5-9", "10-16", "17+"]:
        bucket = buckets.get(name, [])
        if not bucket:
            continue
        bucket_classes = [
            float(row["classes_present"])
            for row in bucket
            if row["classes_present"] != ""
        ]
        markdown.append(
            f"| {name} | {len(bucket)} | "
            f"{statistics.fmean(float(row['total_ms']) for row in bucket):.3f} | "
            f"{percentile([float(row['total_ms']) for row in bucket], 0.95):.3f} | "
            f"{statistics.fmean(float(row['inference_ms']) for row in bucket):.3f} | "
            f"{statistics.fmean(float(row['merge_ms']) for row in bucket):.3f} | "
            f"{statistics.fmean(float(row['write_ms']) for row in bucket):.3f} | "
            f"{(statistics.fmean(bucket_classes) if bucket_classes else 0.0):.2f} |"
        )

    def append_table(title: str, values: list[dict[str, Any]], per_tile: bool = False) -> None:
        markdown.extend(
            [
                "",
                f"## {title}",
                "",
                "| Image | Size | Tiles | Total ms | Inference ms | ms/tile | Classes |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in values:
            markdown.append(
                f"| {Path(row['input']).name} | {row['width']}x{row['height']} | "
                f"{row['tiles']} | {row['total_ms']:.3f} | "
                f"{row['inference_ms']:.3f} | "
                f"{row['total_per_tile_ms']:.3f} | {row['classes_present']} |"
            )

    append_table("Slowest images", slowest)
    append_table("Highest per-tile latency", per_tile_outliers, per_tile=True)
    markdown.extend(
        [
            "",
            "## Class distribution",
            "",
            "| Classes present | Images |",
            "|---:|---:|",
        ]
    )
    for classes, count in sorted(class_counts.items(), key=lambda item: int(item[0] or 0)):
        markdown.append(f"| {classes} | {count} |")

    markdown_path = Path(str(args.output_prefix) + ".md")
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(markdown_path)
    print(csv_path)


if __name__ == "__main__":
    main()
