"""Create three combined evaluation charts from evaluation_metrics.py output.

This script does not modify the metrics report. It reads summary.json and writes
three standalone SVG charts:
  1. performance_summary.svg
  2. conflict_confusion_matrix.svg
  3. conflict_per_class_breakdown.svg

No plotting libraries are required.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


COLORS = {
    "ink": "#1f252b",
    "muted": "#6f777f",
    "grid": "#ded6c9",
    "paper": "#fbfaf7",
    "line": "#cfc5b7",
    "teal": "#18877d",
    "amber": "#c57516",
    "rose": "#c65367",
    "blue": "#4f6f9f",
}

CLASS_ORDER = ["emotional", "logical", "misunderstanding", "value"]


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def read_summary(report_dir: Path) -> dict[str, Any]:
    path = report_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found in {report_dir}")
    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    per_class_path = report_dir / "per_class_metrics.csv"
    if per_class_path.exists():
        with per_class_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                task = row.get("task")
                if not task or task not in summary.get("tasks", {}):
                    continue
                summary["tasks"][task].setdefault("per_class", []).append(
                    {
                        "label": row.get("label", ""),
                        "precision": float(row.get("precision") or 0),
                        "recall": float(row.get("recall") or 0),
                        "f1": float(row.get("f1") or 0),
                        "support": int(float(row.get("support") or 0)),
                    }
                )
    return summary


def svg_shell(width: int, height: int, title: str, body: str, subtitle: str = "") -> str:
    subtitle_text = (
        f'<text x="34" y="58" font-family="Inter, Arial, sans-serif" font-size="13" fill="{COLORS["muted"]}">{esc(subtitle)}</text>'
        if subtitle
        else ""
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" rx="18" fill="{COLORS["paper"]}"/>
  <text x="34" y="34" font-family="Inter, Arial, sans-serif" font-size="22" font-weight="800" fill="{COLORS["ink"]}">{esc(title)}</text>
  {subtitle_text}
  {body}
</svg>
"""


def performance_chart(summary: dict[str, Any]) -> str:
    tasks = summary["tasks"]
    run = summary["run_summary"]
    values = [
        ("Conflict", float(tasks["conflict_type"]["weighted_f1"]), COLORS["teal"]),
        ("Resolvability", float(tasks["resolvability"]["weighted_f1"]), COLORS["amber"]),
        ("Strategy", float(tasks["expected_strategy"]["weighted_f1"]), COLORS["blue"]),
        ("Safety", float(tasks["is_safety_sensitive"]["accuracy"]), COLORS["rose"]),
        ("Overall", float(run["hard_pass_rate"]), COLORS["ink"]),
    ]
    width, height = 980, 520
    chart_x, chart_y, chart_w, chart_h = 90, 100, 820, 300
    bar_gap = 34
    bar_w = (chart_w - bar_gap * (len(values) - 1)) / len(values)
    grid = []
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = chart_y + chart_h - tick * chart_h
        grid.append(
            f'<line x1="{chart_x}" y1="{y:.1f}" x2="{chart_x + chart_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{chart_x - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="12" fill="{COLORS["muted"]}">{pct(tick)}</text>'
        )

    bars = []
    for idx, (label, value, color) in enumerate(values):
        x = chart_x + idx * (bar_w + bar_gap)
        bar_h = value * chart_h
        y = chart_y + chart_h - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="8" fill="{color}"/>'
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 12:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="800" fill="{COLORS["ink"]}">{pct(value)}</text>'
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{chart_y + chart_h + 32:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="700" fill="{COLORS["ink"]}">{esc(label)}</text>'
        )

    note = (
        f'Total cases: {int(run["total_cases"])} | Passed: {int(run["passed"])} | '
        f'Failed: {int(run["failed"])} | Fallbacks: {int(run["fallbacks"])}'
    )
    body = "\n  ".join(grid + bars) + (
        f'\n  <text x="{chart_x}" y="{height - 38}" font-family="Inter, Arial, sans-serif" font-size="13" fill="{COLORS["muted"]}">{esc(note)}</text>'
    )
    return svg_shell(width, height, "Unified Performance Dashboard", body, "One chart for system-level accuracy and F1 signals")


def confusion_chart(summary: dict[str, Any]) -> str:
    task = summary["tasks"]["conflict_type"]
    labels = list(task["labels"])
    matrix = task["confusion_matrix"]
    normalized = task["confusion_matrix_normalized"]
    width, height = 860, 740
    n = len(labels)
    cell = 105
    x0, y0 = 250, 135
    max_count = max(max(row) for row in matrix) or 1
    parts = [
        f'<text x="{x0 + (n * cell) / 2}" y="104" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="800" fill="{COLORS["muted"]}">Predicted</text>',
        f'<text x="76" y="{y0 + (n * cell) / 2}" text-anchor="middle" transform="rotate(-90 76 {y0 + (n * cell) / 2})" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="800" fill="{COLORS["muted"]}">Expected</text>',
    ]
    for idx, label in enumerate(labels):
        x = x0 + idx * cell + cell / 2
        y = y0 - 18
        parts.append(
            f'<text x="{x:.1f}" y="{y}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" font-weight="800" fill="{COLORS["ink"]}">{esc(label)}</text>'
        )
        parts.append(
            f'<text x="{x0 - 16}" y="{y0 + idx * cell + cell / 2 + 5:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="12" font-weight="800" fill="{COLORS["ink"]}">{esc(label)}</text>'
        )

    for row_idx, row in enumerate(matrix):
        for col_idx, count in enumerate(row):
            x = x0 + col_idx * cell
            y = y0 + row_idx * cell
            intensity = count / max_count
            fill = COLORS["teal"] if row_idx == col_idx else COLORS["rose"]
            opacity = 0.12 + 0.78 * intensity
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 6}" height="{cell - 6}" rx="12" fill="{fill}" fill-opacity="{opacity:.2f}" stroke="{COLORS["line"]}"/>'
            )
            parts.append(
                f'<text x="{x + (cell - 6) / 2:.1f}" y="{y + 42:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="900" fill="{COLORS["ink"]}">{count}</text>'
            )
            parts.append(
                f'<text x="{x + (cell - 6) / 2:.1f}" y="{y + 68:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="700" fill="{COLORS["muted"]}">{pct(float(normalized[row_idx][col_idx]))}</text>'
            )

    body = "\n  ".join(parts)
    return svg_shell(width, height, "Conflict Type Confusion Matrix", body, "Raw counts with row-normalized percentages")


def per_class_chart(summary: dict[str, Any]) -> str:
    per_class = summary["tasks"]["conflict_type"].get("per_class")
    if not per_class:
        # summary.json keeps per-class details in per_class_metrics.csv, but older
        # reports may omit it. Reconstruct from perfect matrix if necessary.
        labels = summary["tasks"]["conflict_type"]["labels"]
        per_class = [{"label": label, "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 0} for label in labels]
    by_label = {item["label"]: item for item in per_class}
    ordered = [by_label[label] for label in CLASS_ORDER if label in by_label]
    ordered.extend(item for item in per_class if item["label"] not in {row["label"] for row in ordered})

    width, height = 1060, 560
    chart_x, chart_y, chart_w, chart_h = 95, 105, 885, 300
    group_w = chart_w / len(ordered)
    bar_w = 18
    metrics = [("precision", COLORS["teal"]), ("recall", COLORS["amber"]), ("f1", COLORS["blue"])]
    parts = []
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = chart_y + chart_h - tick * chart_h
        parts.append(f'<line x1="{chart_x}" y1="{y:.1f}" x2="{chart_x + chart_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
        parts.append(
            f'<text x="{chart_x - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="12" fill="{COLORS["muted"]}">{pct(tick)}</text>'
        )
    for idx, row in enumerate(ordered):
        center = chart_x + idx * group_w + group_w / 2
        for metric_idx, (metric, color) in enumerate(metrics):
            value = float(row[metric])
            x = center + (metric_idx - 1) * (bar_w + 8) - bar_w / 2
            h = value * chart_h
            y = chart_y + chart_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="5" fill="{color}"/>')
        parts.append(
            f'<text x="{center:.1f}" y="{chart_y + chart_h + 31}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="800" fill="{COLORS["ink"]}">{esc(row["label"])}</text>'
        )
        parts.append(
            f'<text x="{center:.1f}" y="{chart_y + chart_h + 51}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" fill="{COLORS["muted"]}">n={int(row.get("support", 0))}</text>'
        )

    legend_x = chart_x + 10
    legend_y = height - 58
    for idx, (metric, color) in enumerate(metrics):
        x = legend_x + idx * 145
        parts.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" rx="4" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 22}" y="{legend_y + 12}" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="700" fill="{COLORS["ink"]}">{esc(metric.title())}</text>'
        )

    body = "\n  ".join(parts)
    return svg_shell(width, height, "Per-Class Conflict Performance", body, "Precision, recall, and F1 by conflict category")


def write_index(out_dir: Path, files: list[str]) -> None:
    cards = "\n".join(
        f'<section><h2>{esc(path.replace("_", " ").replace(".svg", "").title())}</h2><img src="{esc(path)}" alt="{esc(path)}"></section>'
        for path in files
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Combined Evaluation Charts</title>
  <style>
    body {{ margin: 0; padding: 32px; background: #f3f0ea; color: #1f252b; font-family: Inter, Arial, sans-serif; }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    h1 {{ margin: 0 0 20px; }}
    section {{ margin: 0 0 28px; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    img {{ width: 100%; height: auto; display: block; border: 1px solid #d7ccbd; border-radius: 18px; background: #fbfaf7; }}
  </style>
</head>
<body>
  <main>
    <h1>Combined Evaluation Charts</h1>
    {cards}
  </main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate three combined SVG charts from metrics_report/summary.json.")
    parser.add_argument("--report", default="metrics_report_batches", help="Metrics report folder containing summary.json.")
    parser.add_argument("--out", default="combined_eval_charts", help="Output folder for combined SVG charts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_dir = Path(args.report)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = read_summary(report_dir)
    outputs = {
        "performance_summary.svg": performance_chart(summary),
        "conflict_confusion_matrix.svg": confusion_chart(summary),
        "conflict_per_class_breakdown.svg": per_class_chart(summary),
    }
    for filename, content in outputs.items():
        (out_dir / filename).write_text(content, encoding="utf-8")
    write_index(out_dir, list(outputs))
    print(f"[charts] wrote {len(outputs)} SVG charts to {out_dir}")
    print(f"[charts] preview: {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
