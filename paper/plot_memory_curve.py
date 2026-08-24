#!/usr/bin/env python3
"""Render the paper's memory/throughput and cache-miss figure as vector PDF."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


BLUE = (0.13, 0.40, 0.67)
ORANGE = (0.72, 0.34, 0.04)
GRAY = (0.45, 0.45, 0.45)
LIGHT_GRAY = (0.82, 0.82, 0.82)
BLACK = (0.08, 0.08, 0.08)


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    return start + (value - low) * (end - start) / (high - low)


def _summary(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    grouped = defaultdict(list)
    for run in payload["runs"]:
        report = run["report"]
        grouped[run["requested_cache_budget_gb"]].append({
            "tokens_per_second": report["generated_tokens"] / report["elapsed_seconds"],
            "peak_gb": report["memory"]["peak_bytes"] / 1_000_000_000,
            "misses": report["expert_cache"]["misses"],
            "slots": report["expert_cache"]["capacity"],
        })
    return [
        {
            "budget_gb": budget,
            "peak_gb": statistics.median(item["peak_gb"] for item in items),
            "median_tps": statistics.median(
                item["tokens_per_second"] for item in items
            ),
            "min_tps": min(item["tokens_per_second"] for item in items),
            "max_tps": max(item["tokens_per_second"] for item in items),
            "all_tps": [item["tokens_per_second"] for item in items],
            "misses": int(statistics.median(item["misses"] for item in items)),
            "slots": items[0]["slots"],
            "n": len(items),
        }
        for budget, items in sorted(grouped.items())
    ]


def _text(c, x, y, value, *, size=7, align="center", color=BLACK):
    c.setFillColorRGB(*color)
    c.setFont("Helvetica", size)
    width = stringWidth(value, "Helvetica", size)
    if align == "center":
        x -= width / 2
    elif align == "right":
        x -= width
    c.drawString(x, y, value)


def _axes(c, box, x_ticks, y_ticks, x_domain, y_domain, y_label, panel):
    left, bottom, right, top = box
    c.setStrokeColorRGB(*BLACK)
    c.setLineWidth(0.6)
    c.rect(left, bottom, right - left, top - bottom, stroke=1, fill=0)
    for value in x_ticks:
        x = _scale(value, *x_domain, left, right)
        c.line(x, bottom, x, bottom - 3)
        _text(c, x, bottom - 12, f"{value:.1f}", size=6.5)
    for value in y_ticks:
        y = _scale(value, *y_domain, bottom, top)
        c.setStrokeColorRGB(*LIGHT_GRAY)
        c.line(left, y, right, y)
        c.setStrokeColorRGB(*BLACK)
        _text(c, left - 5, y - 2.3, f"{value:g}", size=6.5, align="right")
    _text(c, (left + right) / 2, bottom - 24, "Peak MLX memory (GB)", size=7)
    c.saveState()
    c.translate(left - 30, (bottom + top) / 2)
    c.rotate(90)
    _text(c, 0, 0, y_label, size=7)
    c.restoreState()
    _text(c, left, top + 7, panel, size=7, align="left")


def render(input_path: Path, output_path: Path, summary_path: Path | None) -> None:
    points = _summary(input_path)
    if len(points) < 3 or any(point["n"] < 3 for point in points):
        raise ValueError("the paper figure requires at least three budgets and repeats")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 504, 150
    c = canvas.Canvas(str(output_path), pagesize=(width, height))
    left_box = (42, 35, 244, 128)
    right_box = (298, 35, 500, 128)
    x_values = [point["peak_gb"] for point in points]
    x_domain = (min(x_values) - 0.08, max(x_values) + 0.08)
    x_ticks = [5.4, 5.8, 6.2, 6.4]

    all_tps = [value for point in points for value in point["all_tps"]]
    tps_domain = (min(all_tps) - 0.12, max(all_tps) + 0.12)
    tps_ticks = [2.0, 2.4, 2.8, 3.2]
    _axes(
        c, left_box, x_ticks, tps_ticks, x_domain, tps_domain,
        "Generation throughput (tok/s)", "(a) Throughput: median and all runs",
    )
    c.setStrokeColorRGB(*BLUE)
    c.setLineWidth(1.2)
    median_path = c.beginPath()
    for index, point in enumerate(points):
        x = _scale(point["peak_gb"], *x_domain, left_box[0], left_box[2])
        y = _scale(point["median_tps"], *tps_domain, left_box[1], left_box[3])
        if index == 0:
            median_path.moveTo(x, y)
        else:
            median_path.lineTo(x, y)
        c.setStrokeColorRGB(*GRAY)
        for offset, value in zip((-2.2, 0, 2.2), sorted(point["all_tps"])):
            observed_y = _scale(value, *tps_domain, left_box[1], left_box[3])
            c.circle(x + offset, observed_y, 1.7, stroke=1, fill=0)
        c.setStrokeColorRGB(*BLUE)
        c.setFillColorRGB(*BLUE)
        c.circle(x, y, 2.7, stroke=1, fill=1)
    c.drawPath(median_path, stroke=1, fill=0)
    gain = 100 * (points[-1]["median_tps"] / points[0]["median_tps"] - 1)
    _text(
        c, left_box[2] - 3, left_box[3] - 10,
        f"median: +{gain:.1f}%", size=6.5, align="right", color=BLUE,
    )

    miss_values = [point["misses"] for point in points]
    miss_domain = (min(miss_values) - 300, max(miss_values) + 300)
    miss_ticks = [4500, 5000, 5500, 6000]
    _axes(
        c, right_box, x_ticks, miss_ticks, x_domain, miss_domain,
        "Expert-cache misses / 64 tokens", "(b) Mechanism: fewer cache misses",
    )
    c.setStrokeColorRGB(*ORANGE)
    c.setFillColorRGB(*ORANGE)
    c.setLineWidth(1.2)
    miss_path = c.beginPath()
    for index, point in enumerate(points):
        x = _scale(point["peak_gb"], *x_domain, right_box[0], right_box[2])
        y = _scale(point["misses"], *miss_domain, right_box[1], right_box[3])
        if index == 0:
            miss_path.moveTo(x, y)
        else:
            miss_path.lineTo(x, y)
        c.circle(x, y, 2.7, stroke=1, fill=1)
        _text(c, x, y + 6, str(point["slots"]), size=6, color=ORANGE)
    c.drawPath(miss_path, stroke=1, fill=0)
    reduction = 100 * (1 - points[-1]["misses"] / points[0]["misses"])
    _text(
        c, right_box[2] - 3, right_box[3] - 10,
        f"misses: -{reduction:.1f}%", size=6.5, align="right", color=ORANGE,
    )
    _text(
        c, (right_box[0] + right_box[2]) / 2, right_box[1] + 4,
        "labels show resident expert slots", size=6, color=GRAY,
    )
    c.showPage()
    c.save()
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(points, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    render(args.input, args.output, args.summary)


if __name__ == "__main__":
    main()
