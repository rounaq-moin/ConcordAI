"""Leakage-aware audit for saved backend evaluation results.

This is intentionally separate from evaluation_metrics.py. It answers a
different question: which metrics are truly measured from model/system output,
which are proxies, and which are not independently measurable from the saved
result files.

The goal is to prevent suspicious "everything is 100%" reports caused by
comparing labels against derived labels instead of independent predictions.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OBSERVED_TASKS = ("conflict_type", "resolvability", "should_use_fallback")
PROXY_TASKS = ("output_safety_toxicity",)
ANNOTATION_TASKS = ("expected_strategy", "is_safety_sensitive", "intensity", "is_repeat_pattern")
HARD_CHECKS = (
    "conflict_type",
    "resolvability",
    "distinct_responses",
    "specific_a",
    "specific_b",
    "empathy_a",
    "empathy_b",
    "pov_correct",
    "safety",
)
CHART_COLORS = {
    "paper": "#fbfaf7",
    "ink": "#1f252b",
    "muted": "#69727c",
    "grid": "#ded6c9",
    "border": "#d4cabb",
    "teal": "#18877d",
    "amber": "#c57516",
    "rose": "#c65367",
    "blue": "#4f6f9f",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def as_list(payload: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    raise ValueError(f"{source} must be a JSON list or contain a 'results' list.")


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "unknown"
    return str(value)


def case_sort_key(case_id: str) -> tuple[str, int, str]:
    match = re.match(r"^([A-Za-z]+)(\d+)$", case_id)
    if not match:
        return (case_id, -1, case_id)
    return (match.group(1), int(match.group(2)), case_id)


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    cases = as_list(load_json(path), path)
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for case in cases:
        case_id = str(case.get("id", "")).strip()
        if not case_id:
            raise ValueError(f"{path} contains a case without an id.")
        if case_id in by_id:
            duplicates.append(case_id)
        by_id[case_id] = case
    if duplicates:
        raise ValueError(f"Duplicate case ids in {path}: {', '.join(sorted(set(duplicates)))}")
    return by_id


def load_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}
    duplicates: list[str] = []
    for path in paths:
        results = as_list(load_json(path), path)
        for result in results:
            result_id = str(result.get("id", "")).strip()
            if not result_id:
                raise ValueError(f"{path} contains a result without an id.")
            if result_id in by_id:
                duplicates.append(f"{result_id} ({origins[result_id]} and {path})")
            by_id[result_id] = result
            origins[result_id] = str(path)
    if duplicates:
        raise ValueError("Duplicate result ids found: " + "; ".join(duplicates))
    return by_id


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def classification_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    total = len(pairs)
    correct = sum(1 for expected, actual in pairs if expected == actual)
    labels = sorted({value for pair in pairs for value in pair})
    per_class: list[dict[str, Any]] = []
    for item in labels:
        tp = sum(1 for expected, actual in pairs if expected == item and actual == item)
        fp = sum(1 for expected, actual in pairs if expected != item and actual == item)
        fn = sum(1 for expected, actual in pairs if expected == item and actual != item)
        support = sum(1 for expected, _ in pairs if expected == item)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class.append(
            {
                "label": item,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    accuracy = safe_div(correct, total)
    macro_f1 = safe_div(sum(row["f1"] for row in per_class), len(per_class))
    weighted_f1 = safe_div(sum(row["f1"] * row["support"] for row in per_class), total)
    return {
        "total": total,
        "accuracy": accuracy,
        "micro_f1": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mismatch_count": total - correct,
        "per_class": per_class,
    }


def build_rows(cases: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    unknown_ids = sorted(set(results) - set(cases), key=case_sort_key)
    if unknown_ids:
        raise ValueError("Result ids missing from test_cases.json: " + ", ".join(unknown_ids))
    rows: list[dict[str, Any]] = []
    for case_id in sorted(results, key=case_sort_key):
        case = cases[case_id]
        result = results[case_id]
        rows.append(
            {
                "id": case_id,
                "category": case.get("category", ""),
                "description": case.get("description", ""),
                "case": case,
                "result": result,
            }
        )
    return rows


def observed_pairs(rows: list[dict[str, Any]], task: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for row in rows:
        case = row["case"]
        result = row["result"]
        if task == "conflict_type":
            pairs.append((label(case.get("expected_conflict_type")), label(result.get("conflict_type"))))
        elif task == "resolvability":
            pairs.append((label(case.get("expected_resolvability")), label(result.get("resolvability"))))
        elif task == "should_use_fallback":
            pairs.append((label(boolish(case.get("should_use_fallback", False))), label(boolish(result.get("fallback_used", False)))))
        elif task == "output_safety_toxicity":
            # This is intentionally named as a proxy: the saved result contains
            # output toxicity score, not a direct input safety-sensitive classifier.
            score = result.get("safety_score")
            try:
                actual_unsafe = float(score) >= 0.4 if score is not None else False
            except (TypeError, ValueError):
                actual_unsafe = False
            pairs.append((label(False), label(actual_unsafe)))
    return pairs


def soft_check_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for task in ANNOTATION_TASKS:
        total = len(rows)
        present = 0
        matches = 0
        mismatches = 0
        for row in rows:
            soft = row["result"].get("soft_checks")
            item = soft.get(task) if isinstance(soft, dict) else None
            if isinstance(item, dict) and "actual" in item and "expected" in item:
                present += 1
                if item.get("match"):
                    matches += 1
                else:
                    mismatches += 1
        coverage[task] = {
            "total": total,
            "present": present,
            "missing": total - present,
            "matches": matches,
            "mismatches": mismatches,
            "status": "not_measured" if present == 0 else "derived_soft_check",
            "warning": (
                "No independent actual prediction exists in saved results."
                if present == 0
                else "Actual comes from backend_eval soft-check logic, not a separate model output."
            ),
        }
    return coverage


def hard_run_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if boolish(row["result"].get("passed")))
    failed = total - passed
    errors = sum(1 for row in rows if row["result"].get("error"))
    fallbacks = sum(1 for row in rows if boolish(row["result"].get("fallback_used")))
    return {
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "fallbacks": fallbacks,
        "hard_pass_rate": safe_div(passed, total),
    }


def hard_check_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for check in HARD_CHECKS:
        values: list[Any] = []
        for row in rows:
            checks = row["result"].get("checks")
            if isinstance(checks, dict) and check in checks:
                values.append(checks[check].get("passed"))
        total = len(values)
        passed = sum(1 for value in values if value is True)
        warnings = sum(1 for value in values if value is None)
        failed = sum(1 for value in values if value is False)
        out.append(
            {
                "check": check,
                "total": total,
                "passed": passed,
                "warnings": warnings,
                "failed": failed,
                "strict_pass_rate": safe_div(passed, total),
                "weighted_pass_rate": safe_div(passed + 0.5 * warnings, total),
            }
        )
    return out


def dataset_distribution(cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fields = {
        "conflict_type": "expected_conflict_type",
        "resolvability": "expected_resolvability",
        "strategy": "expected_strategy",
        "intensity": "intensity",
        "safety_sensitive": "is_safety_sensitive",
        "repeat_pattern": "is_repeat_pattern",
    }
    out: list[dict[str, Any]] = []
    for field, key in fields.items():
        counts = Counter(label(case.get(key, "unknown")) for case in cases.values())
        total = sum(counts.values())
        for value, count in sorted(counts.items()):
            out.append({"field": field, "label": value, "count": count, "percent": safe_div(count, total)})
    return out


def mismatches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in OBSERVED_TASKS:
        for row, (expected, actual) in zip(rows, observed_pairs(rows, task)):
            if expected != actual:
                result = row["result"]
                out.append(
                    {
                        "task": task,
                        "id": row["id"],
                        "expected": expected,
                        "actual": actual,
                        "confidence": result.get("confidence"),
                        "fallback_used": result.get("fallback_used"),
                        "passed": result.get("passed"),
                        "category": row["category"],
                        "description": row["description"],
                        "error": result.get("error"),
                    }
                )
    return out


def suspicious_metrics(rows: list[dict[str, Any]], observed_metrics: dict[str, dict[str, Any]], soft_coverage: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for task, metric in observed_metrics.items():
        if metric["accuracy"] == 1.0 and task != "should_use_fallback":
            flags.append(
                {
                    "task": task,
                    "severity": "review",
                    "reason": "100% observed score. This can be real, but should be checked against mismatches and source fields.",
                    "source": "independent_result_field",
                }
            )
    for task, info in soft_coverage.items():
        flags.append(
            {
                "task": task,
                "severity": "high" if info["status"] == "not_measured" else "medium",
                "reason": info["warning"],
                "source": info["status"],
            }
        )
    return flags


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    return value


def svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pct_text(value: float) -> str:
    return f"{value * 100:.0f}%"


def svg_shell(width: int, height: int, title: str, subtitle: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" rx="18" fill="{CHART_COLORS["paper"]}"/>
  <text x="34" y="36" font-family="Inter, Arial, sans-serif" font-size="22" font-weight="900" fill="{CHART_COLORS["ink"]}">{svg_escape(title)}</text>
  <text x="34" y="60" font-family="Inter, Arial, sans-serif" font-size="13" fill="{CHART_COLORS["muted"]}">{svg_escape(subtitle)}</text>
  {body}
</svg>
"""


def honest_summary_chart(summary: dict[str, Any], hard_checks: list[dict[str, Any]]) -> str:
    run = summary["run_summary"]
    observed = summary["observed_metrics"]
    proxy = summary["proxy_metrics"]
    check_map = {item["check"]: item for item in hard_checks}
    values = [
        ("Hard pass", run["hard_pass_rate"], CHART_COLORS["ink"], "true run result"),
        ("Fallback ok", observed["should_use_fallback"]["accuracy"], CHART_COLORS["amber"], "operational"),
        ("Empathy A", check_map.get("empathy_a", {}).get("strict_pass_rate", 0), CHART_COLORS["teal"], "hard check"),
        ("Empathy B", check_map.get("empathy_b", {}).get("strict_pass_rate", 0), CHART_COLORS["teal"], "hard check"),
        ("Safety", proxy["output_safety_toxicity"]["accuracy"], CHART_COLORS["blue"], "toxicity proxy"),
        ("Conflict", observed["conflict_type"]["accuracy"], CHART_COLORS["rose"], "review 100%"),
        ("Resolve", observed["resolvability"]["accuracy"], CHART_COLORS["rose"], "review 100%"),
    ]
    width, height = 1120, 560
    x0, y0, chart_w, chart_h = 90, 116, 960, 300
    gap = 24
    bar_w = (chart_w - gap * (len(values) - 1)) / len(values)
    parts: list[str] = []
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y0 + chart_h - tick * chart_h
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + chart_w}" y2="{y:.1f}" stroke="{CHART_COLORS["grid"]}"/>')
        parts.append(f'<text x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="12" fill="{CHART_COLORS["muted"]}">{pct_text(tick)}</text>')
    for idx, (label_name, value, color, note) in enumerate(values):
        x = x0 + idx * (bar_w + gap)
        h = value * chart_h
        y = y0 + chart_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="8" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 12:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="15" font-weight="900" fill="{CHART_COLORS["ink"]}">{pct_text(value)}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y0 + chart_h + 30:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="800" fill="{CHART_COLORS["ink"]}">{svg_escape(label_name)}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y0 + chart_h + 49:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="10" fill="{CHART_COLORS["muted"]}">{svg_escape(note)}</text>')
    footer = f"Cases: {run['total_cases']} | Passed: {run['passed']} | Failed: {run['failed']} | Fallbacks: {run['fallbacks']}"
    parts.append(f'<text x="{x0}" y="{height - 36}" font-family="Inter, Arial, sans-serif" font-size="13" fill="{CHART_COLORS["muted"]}">{svg_escape(footer)}</text>')
    return svg_shell(width, height, "Honest Performance Summary", "Shows real run quality plus review-flagged perfect metrics", "\n  ".join(parts))


def hard_checks_chart(hard_checks: list[dict[str, Any]]) -> str:
    width, height = 1120, 600
    x0, y0, chart_w, row_h = 270, 104, 780, 42
    parts: list[str] = []
    for idx, item in enumerate(hard_checks):
        y = y0 + idx * row_h
        rate = float(item["strict_pass_rate"])
        failed = int(item["failed"])
        color = CHART_COLORS["teal"] if rate >= 0.98 else (CHART_COLORS["amber"] if rate >= 0.9 else CHART_COLORS["rose"])
        parts.append(f'<text x="34" y="{y + 21}" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="800" fill="{CHART_COLORS["ink"]}">{svg_escape(item["check"])}</text>')
        parts.append(f'<rect x="{x0}" y="{y}" width="{chart_w}" height="24" rx="8" fill="#eee8df"/>')
        parts.append(f'<rect x="{x0}" y="{y}" width="{chart_w * rate:.1f}" height="24" rx="8" fill="{color}"/>')
        parts.append(f'<text x="{x0 + chart_w + 14}" y="{y + 18}" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="900" fill="{CHART_COLORS["ink"]}">{pct_text(rate)}</text>')
        if failed:
            parts.append(f'<text x="{x0 + chart_w + 70}" y="{y + 18}" font-family="Inter, Arial, sans-serif" font-size="12" fill="{CHART_COLORS["rose"]}">{failed} fail</text>')
    return svg_shell(width, height, "Hard Check Breakdown", "What actually passed or failed in the saved backend eval output", "\n  ".join(parts))


def provenance_chart(summary: dict[str, Any]) -> str:
    observed_count = len(summary["observed_metrics"])
    proxy_count = len(summary["proxy_metrics"])
    not_measured_count = sum(1 for item in summary["annotation_metric_coverage"].values() if item["status"] == "not_measured")
    review_count = sum(1 for item in summary["suspicious_or_not_measured"] if item["severity"] == "review")
    cards = [
        ("Observed", observed_count, CHART_COLORS["teal"], "Actual fields exist in eval result JSON"),
        ("Proxy", proxy_count, CHART_COLORS["blue"], "Measured indirectly, useful but limited"),
        ("Not measured", not_measured_count, CHART_COLORS["rose"], "No independent actual prediction saved"),
        ("Review", review_count, CHART_COLORS["amber"], "100% scores that deserve source checking"),
    ]
    width, height = 1060, 430
    card_w, card_h = 220, 180
    start_x, y = 48, 116
    parts: list[str] = []
    for idx, (title, count, color, note) in enumerate(cards):
        x = start_x + idx * (card_w + 28)
        parts.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="18" fill="#fffdf9" stroke="{CHART_COLORS["border"]}"/>')
        parts.append(f'<circle cx="{x + 38}" cy="{y + 42}" r="13" fill="{color}"/>')
        parts.append(f'<text x="{x + 62}" y="{y + 47}" font-family="Inter, Arial, sans-serif" font-size="17" font-weight="900" fill="{CHART_COLORS["ink"]}">{svg_escape(title)}</text>')
        parts.append(f'<text x="{x + 34}" y="{y + 112}" font-family="Inter, Arial, sans-serif" font-size="54" font-weight="900" fill="{color}">{count}</text>')
        wrapped = wrap_svg_text(note, 28)
        for line_idx, line in enumerate(wrapped):
            parts.append(f'<text x="{x + 34}" y="{y + 142 + line_idx * 17}" font-family="Inter, Arial, sans-serif" font-size="12" fill="{CHART_COLORS["muted"]}">{svg_escape(line)}</text>')
    parts.append(f'<text x="48" y="358" font-family="Inter, Arial, sans-serif" font-size="13" fill="{CHART_COLORS["muted"]}">This chart is the anti-fake-perfect guard: it separates real metrics from labels the result files cannot independently prove.</text>')
    return svg_shell(width, height, "Metric Provenance / Leakage Risk", "Why not every annotation should become a 100% score", "\n  ".join(parts))


def weighted_class_metric(metric: dict[str, Any], key: str) -> float:
    per_class = metric.get("per_class") or []
    total = sum(float(item.get("support", 0)) for item in per_class)
    return safe_div(sum(float(item.get(key, 0)) * float(item.get("support", 0)) for item in per_class), total)


def line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    first, *rest = points
    commands = [f"M {first[0]:.1f} {first[1]:.1f}"]
    commands.extend(f"L {x:.1f} {y:.1f}" for x, y in rest)
    return " ".join(commands)


def metric_xy(
    labels: list[str],
    values: list[float],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> list[tuple[float, float]]:
    step = width / max(1, len(labels) - 1)
    return [(x0 + idx * step, y0 + height - max(0, min(1, value)) * height) for idx, value in enumerate(values)]


def chart_axes(x0: float, y0: float, width: float, height: float) -> list[str]:
    parts: list[str] = []
    for tick in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = y0 + height - tick * height
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" stroke="#ebedf0" stroke-width="1"/>')
        parts.append(f'<text x="{x0 - 8}" y="{y + 3:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#717780">{tick:.1f}</text>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#d8dde4"/>')
    parts.append(f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#d8dde4"/>')
    return parts


def audit_dashboard_chart(summary: dict[str, Any], hard_checks: list[dict[str, Any]]) -> str:
    """A single academic-style dashboard matching the requested layout."""
    observed = summary["observed_metrics"]
    proxy = summary["proxy_metrics"]
    run = summary["run_summary"]
    check_map = {item["check"]: item for item in hard_checks}

    task_labels = ["Conflict", "Resolve", "Fallback", "Safety", "Empathy A", "Empathy B", "POV", "Overall"]
    accuracy_values = [
        observed["conflict_type"]["accuracy"],
        observed["resolvability"]["accuracy"],
        observed["should_use_fallback"]["accuracy"],
        proxy["output_safety_toxicity"]["accuracy"],
        check_map.get("empathy_a", {}).get("strict_pass_rate", 0),
        check_map.get("empathy_b", {}).get("strict_pass_rate", 0),
        check_map.get("pov_correct", {}).get("strict_pass_rate", 0),
        run["hard_pass_rate"],
    ]
    macro_values = [
        observed["conflict_type"]["macro_f1"],
        observed["resolvability"]["macro_f1"],
        observed["should_use_fallback"]["macro_f1"],
        proxy["output_safety_toxicity"]["macro_f1"],
        check_map.get("empathy_a", {}).get("weighted_pass_rate", 0),
        check_map.get("empathy_b", {}).get("weighted_pass_rate", 0),
        check_map.get("pov_correct", {}).get("weighted_pass_rate", 0),
        run["hard_pass_rate"],
    ]
    weighted_values = [
        observed["conflict_type"]["weighted_f1"],
        observed["resolvability"]["weighted_f1"],
        observed["should_use_fallback"]["weighted_f1"],
        proxy["output_safety_toxicity"]["weighted_f1"],
        check_map.get("empathy_a", {}).get("strict_pass_rate", 0),
        check_map.get("empathy_b", {}).get("strict_pass_rate", 0),
        check_map.get("pov_correct", {}).get("strict_pass_rate", 0),
        run["hard_pass_rate"],
    ]

    width, height = 1500, 960
    parts = [
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="750" y="32" text-anchor="middle" font-family="Georgia, serif" font-size="18" font-weight="700" fill="#222">ConcordAI Evaluation Results</text>',
    ]

    # Top-left line chart.
    x0, y0, w, h = 90, 72, 610, 245
    parts.append('<text x="395" y="55" text-anchor="middle" font-family="Georgia, serif" font-size="13" font-weight="700" fill="#111">Comparison of Evaluation Metrics Across Audit Tasks</text>')
    parts.extend(chart_axes(x0, y0, w, h))
    series = [
        ("Accuracy", accuracy_values, "#1f77b4"),
        ("Macro F1", macro_values, "#ff7f0e"),
        ("Weighted F1", weighted_values, "#2ca02c"),
    ]
    for name, values, color in series:
        points = metric_xy(task_labels, values, x0=x0, y0=y0, width=w, height=h)
        parts.append(f'<path d="{line_path(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#fff" stroke-width="1"/>')
    for idx, label_name in enumerate(task_labels):
        x = x0 + idx * (w / max(1, len(task_labels) - 1))
        parts.append(f'<text x="{x:.1f}" y="{y0 + h + 22}" text-anchor="end" transform="rotate(-35 {x:.1f} {y0 + h + 22})" font-family="Arial, sans-serif" font-size="10" fill="#333">{svg_escape(label_name)}</text>')
    for idx, (name, _, color) in enumerate(series):
        lx = x0 + w - 90
        ly = y0 + 16 + idx * 18
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 20}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 26}" y="{ly + 4}" font-family="Arial, sans-serif" font-size="10" fill="#222">{svg_escape(name)}</text>')

    # Bottom-left grouped bar chart.
    bx0, by0, bw, bh = 90, 405, 610, 245
    parts.append('<text x="395" y="388" text-anchor="middle" font-family="Georgia, serif" font-size="13" font-weight="700" fill="#111">Audit-wise Comparison of Evaluation Metrics</text>')
    parts.extend(chart_axes(bx0, by0, bw, bh))
    group_w = bw / len(task_labels)
    bar_w = 14
    for idx, label_name in enumerate(task_labels):
        center = bx0 + idx * group_w + group_w / 2
        values = [accuracy_values[idx], macro_values[idx], weighted_values[idx]]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for bidx, value in enumerate(values):
            x = center + (bidx - 1) * (bar_w + 3) - bar_w / 2
            bar_h = max(0, min(1, value)) * bh
            y = by0 + bh - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{colors[bidx]}"/>')
        parts.append(f'<text x="{center:.1f}" y="{by0 + bh + 22}" text-anchor="end" transform="rotate(-35 {center:.1f} {by0 + bh + 22})" font-family="Arial, sans-serif" font-size="10" fill="#333">{svg_escape(label_name)}</text>')
    for idx, (name, _, color) in enumerate(series):
        lx = bx0 + bw - 100
        ly = by0 + 16 + idx * 18
        parts.append(f'<rect x="{lx}" y="{ly - 8}" width="14" height="10" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{ly + 1}" font-family="Arial, sans-serif" font-size="10" fill="#222">{svg_escape(name)}</text>')

    # Top-right precision/recall/F1 chart for independently observed tasks.
    rx0, ry0, rw, rh = 835, 80, 520, 330
    parts.append('<text x="1095" y="55" text-anchor="middle" font-family="Georgia, serif" font-size="16" font-weight="700" fill="#111">Precision, Recall, and F1-Score Across Observed Tasks</text>')
    parts.extend(chart_axes(rx0, ry0, rw, rh))
    right_tasks = [
        ("Conflict", observed["conflict_type"]),
        ("Resolve", observed["resolvability"]),
        ("Fallback", observed["should_use_fallback"]),
    ]
    right_metrics = [("Precision", "precision", "#1f77b4"), ("Recall", "recall", "#ff7f0e"), ("F1-Score", "f1", "#2ca02c")]
    right_group_w = rw / len(right_tasks)
    right_bar_w = 36
    for idx, (task_label, metric) in enumerate(right_tasks):
        center = rx0 + idx * right_group_w + right_group_w / 2
        weighted_precision = weighted_class_metric(metric, "precision")
        weighted_recall = weighted_class_metric(metric, "recall")
        weighted_f1 = metric["weighted_f1"]
        values = [weighted_precision, weighted_recall, weighted_f1]
        for midx, (_, _, color) in enumerate(right_metrics):
            value = values[midx]
            x = center + (midx - 1) * (right_bar_w + 8) - right_bar_w / 2
            bar_h = max(0, min(1, value)) * rh
            y = ry0 + rh - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{right_bar_w}" height="{bar_h:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x + right_bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#222">{pct_text(value)}</text>')
        parts.append(f'<text x="{center:.1f}" y="{ry0 + rh + 26}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#222">{svg_escape(task_label)}</text>')
    for idx, (name, _, color) in enumerate(right_metrics):
        lx = rx0 + rw - 100
        ly = ry0 + 18 + idx * 18
        parts.append(f'<rect x="{lx}" y="{ly - 8}" width="14" height="10" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{ly + 1}" font-family="Arial, sans-serif" font-size="10" fill="#222">{svg_escape(name)}</text>')

    # Bottom result cards.
    card_y = 690
    card_w, card_h = 330, 210
    cards = [
        (
            180,
            "#fff6ef",
            "#e9a071",
            "CORE CLASSIFICATION - AUDIT RESULTS",
            [
                ("Task Type", "Observed output fields"),
                ("Conflict", pct_text(observed["conflict_type"]["accuracy"])),
                ("Resolvability", pct_text(observed["resolvability"]["accuracy"])),
                ("Review Flag", "Perfect scores need source check"),
                ("Assessment", "Strong, but verify against fresh runs"),
            ],
        ),
        (
            585,
            "#f2f7ff",
            "#7da8e8",
            "RESPONSE QUALITY - AUDIT RESULTS",
            [
                ("Hard pass", pct_text(run["hard_pass_rate"])),
                ("Empathy A", pct_text(check_map.get("empathy_a", {}).get("strict_pass_rate", 0))),
                ("Empathy B", pct_text(check_map.get("empathy_b", {}).get("strict_pass_rate", 0))),
                ("POV", pct_text(check_map.get("pov_correct", {}).get("strict_pass_rate", 0))),
                ("Assessment", "Good quality; empathy is the main gap"),
            ],
        ),
        (
            990,
            "#f1fbf3",
            "#84c995",
            "OPERATIONAL RELIABILITY - AUDIT RESULTS",
            [
                ("Fallbacks", f"{run['fallbacks']} / {run['total_cases']}"),
                ("Fallback expected", pct_text(observed["should_use_fallback"]["accuracy"])),
                ("Errors", str(run["errors"])),
                ("Safety proxy", pct_text(proxy["output_safety_toxicity"]["accuracy"])),
                ("Assessment", "Stable, but fallback usage is visible"),
            ],
        ),
    ]
    for x, fill, stroke, title, rows in cards:
        parts.append(f'<rect x="{x}" y="{card_y}" width="{card_w}" height="{card_h}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<text x="{x + 18}" y="{card_y + 26}" font-family="Consolas, monospace" font-size="12" font-weight="700" fill="#222">{svg_escape(title)}</text>')
        parts.append(f'<line x1="{x + 18}" y1="{card_y + 38}" x2="{x + card_w - 18}" y2="{card_y + 38}" stroke="{stroke}"/>')
        for ridx, (key, value) in enumerate(rows):
            y = card_y + 62 + ridx * 28
            parts.append(f'<text x="{x + 24}" y="{y}" font-family="Consolas, monospace" font-size="12" fill="#444">- {svg_escape(key)}:</text>')
            parts.append(f'<text x="{x + 158}" y="{y}" font-family="Consolas, monospace" font-size="12" font-weight="700" fill="#222">{svg_escape(value)}</text>')

    parts.append('<text x="750" y="940" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#777">Note: Strategy, intensity, repeat-pattern, and safety-sensitive annotations are not scored as independent model outputs unless saved actual predictions exist.</text>')
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', *parts, "</svg>"])


def clean_dashboard_chart(summary: dict[str, Any], hard_checks: list[dict[str, Any]]) -> str:
    """A clean, presentation-first dashboard for the honest audit report."""
    observed = summary["observed_metrics"]
    proxy = summary["proxy_metrics"]
    run = summary["run_summary"]
    flags = summary["suspicious_or_not_measured"]
    check_map = {item["check"]: item for item in hard_checks}

    task_labels = ["Conflict", "Resolve", "Fallback", "Safety", "Empathy A", "Empathy B", "POV", "Overall"]
    accuracy_values = [
        observed["conflict_type"]["accuracy"],
        observed["resolvability"]["accuracy"],
        observed["should_use_fallback"]["accuracy"],
        proxy["output_safety_toxicity"]["accuracy"],
        check_map.get("empathy_a", {}).get("strict_pass_rate", 0),
        check_map.get("empathy_b", {}).get("strict_pass_rate", 0),
        check_map.get("pov_correct", {}).get("strict_pass_rate", 0),
        run["hard_pass_rate"],
    ]
    macro_values = [
        observed["conflict_type"]["macro_f1"],
        observed["resolvability"]["macro_f1"],
        observed["should_use_fallback"]["macro_f1"],
        proxy["output_safety_toxicity"]["macro_f1"],
        check_map.get("empathy_a", {}).get("weighted_pass_rate", 0),
        check_map.get("empathy_b", {}).get("weighted_pass_rate", 0),
        check_map.get("pov_correct", {}).get("weighted_pass_rate", 0),
        run["hard_pass_rate"],
    ]
    weighted_values = [
        observed["conflict_type"]["weighted_f1"],
        observed["resolvability"]["weighted_f1"],
        observed["should_use_fallback"]["weighted_f1"],
        proxy["output_safety_toxicity"]["weighted_f1"],
        check_map.get("empathy_a", {}).get("strict_pass_rate", 0),
        check_map.get("empathy_b", {}).get("strict_pass_rate", 0),
        check_map.get("pov_correct", {}).get("strict_pass_rate", 0),
        run["hard_pass_rate"],
    ]

    width, height = 1600, 1100
    bg = "#f6f2eb"
    panel_fill = "#fffdf8"
    ink = "#192126"
    muted = "#6f777d"
    grid = "#e6ded2"
    border = "#d7cdbc"
    blue = "#246f9f"
    amber = "#c57516"
    teal = "#18877d"
    rose = "#b95366"
    green = "#2f9461"
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
    ]

    def panel(x: int, y: int, w: int, h: int, title: str, subtitle: str = "") -> None:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{panel_fill}" stroke="{border}"/>')
        parts.append(f'<text x="{x + 24}" y="{y + 34}" font-family="Inter, Arial, sans-serif" font-size="18" font-weight="900" fill="{ink}">{svg_escape(title)}</text>')
        if subtitle:
            parts.append(f'<text x="{x + 24}" y="{y + 57}" font-family="Inter, Arial, sans-serif" font-size="12" fill="{muted}">{svg_escape(subtitle)}</text>')

    def chip(x: int, y: int, label_name: str, value: str, color: str) -> int:
        text_w = 18 + len(label_name) * 7 + len(value) * 9 + 26
        parts.append(f'<rect x="{x}" y="{y}" width="{text_w}" height="34" rx="17" fill="#ffffff" stroke="{border}"/>')
        parts.append(f'<circle cx="{x + 18}" cy="{y + 17}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{x + 32}" y="{y + 22}" font-family="Inter, Arial, sans-serif" font-size="12" fill="{muted}">{svg_escape(label_name)}</text>')
        parts.append(f'<text x="{x + text_w - 18}" y="{y + 22}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="900" fill="{ink}">{svg_escape(value)}</text>')
        return text_w + 10

    def y_for(value: float, y: float, h: float) -> float:
        return y + h - max(0, min(1, value)) * h

    def add_axes(x: int, y: int, w: int, h: int, ticks: list[float] | None = None) -> None:
        for tick in ticks or [0, 0.25, 0.5, 0.75, 1.0]:
            yy = y_for(tick, y, h)
            parts.append(f'<line x1="{x}" y1="{yy:.1f}" x2="{x + w}" y2="{yy:.1f}" stroke="{grid}"/>')
            parts.append(f'<text x="{x - 12}" y="{yy + 4:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="11" fill="{muted}">{pct_text(tick)}</text>')
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="{border}"/>')
        parts.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="{border}"/>')

    # Header.
    parts.append(f'<rect x="40" y="28" width="1520" height="100" rx="22" fill="{panel_fill}" stroke="{border}"/>')
    parts.append(f'<text x="72" y="70" font-family="Inter, Arial, sans-serif" font-size="28" font-weight="900" fill="{ink}">ConcordAI Evaluation Dashboard</text>')
    parts.append(f'<text x="72" y="96" font-family="Inter, Arial, sans-serif" font-size="13" fill="{muted}">Leakage-aware report from saved backend evaluation batches. Perfect scores are shown, but flagged when source independence is limited.</text>')
    cx = 1030
    cx += chip(cx, 58, "Cases", str(run["total_cases"]), blue)
    cx += chip(cx, 58, "Hard pass", pct_text(run["hard_pass_rate"]), teal)
    cx += chip(cx, 58, "Fallbacks", str(run["fallbacks"]), amber)

    # Performance profile.
    panel(40, 152, 930, 380, "Performance Profile", "Accuracy and F1 by independently observed or proxy signal")
    x0, y0, w, h = 105, 245, 790, 205
    add_axes(x0, y0, w, h)
    series = [
        ("Accuracy", accuracy_values, blue),
        ("Macro F1", macro_values, amber),
        ("Weighted F1", weighted_values, green),
    ]
    for name, values, color in series:
        points = metric_xy(task_labels, values, x0=x0, y0=y0, width=w, height=h)
        parts.append(f'<path d="{line_path(points)}" fill="none" stroke="{color}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>')
        for idx, (px, py) in enumerate(points):
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{color}" stroke="{panel_fill}" stroke-width="2"/>')
            if values[idx] < 0.93:
                parts.append(f'<text x="{px:.1f}" y="{py - 10:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" font-weight="800" fill="{color}">{pct_text(values[idx])}</text>')
    for idx, label_name in enumerate(task_labels):
        px = x0 + idx * (w / max(1, len(task_labels) - 1))
        parts.append(f'<text x="{px:.1f}" y="{y0 + h + 28}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" fill="{ink}">{svg_escape(label_name)}</text>')
    lx = 720
    for idx, (name, _, color) in enumerate(series):
        ly = 188 + idx * 20
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 22}" y2="{ly}" stroke="{color}" stroke-width="3.2" stroke-linecap="round"/>')
        parts.append(f'<text x="{lx + 30}" y="{ly + 4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="{ink}">{svg_escape(name)}</text>')

    # Precision / recall / F1.
    panel(1000, 152, 560, 380, "Observed Task Scores", "Weighted precision, recall, and F1 for fields present in the eval output")
    rx, ry, rw, rh = 1060, 268, 425, 182
    add_axes(rx, ry, rw, rh)
    right_tasks = [
        ("Conflict", observed["conflict_type"]),
        ("Resolve", observed["resolvability"]),
        ("Fallback", observed["should_use_fallback"]),
    ]
    right_metrics = [("Precision", blue), ("Recall", amber), ("F1", green)]
    group_w = rw / len(right_tasks)
    bar_w = 30
    for idx, (task_name, metric) in enumerate(right_tasks):
        center = rx + idx * group_w + group_w / 2
        values = [
            weighted_class_metric(metric, "precision"),
            weighted_class_metric(metric, "recall"),
            metric["weighted_f1"],
        ]
        for bidx, value in enumerate(values):
            color = right_metrics[bidx][1]
            bh = max(0, min(1, value)) * rh
            bx = center + (bidx - 1) * (bar_w + 8) - bar_w / 2
            by = ry + rh - bh
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" rx="5" fill="{color}"/>')
            if value < 0.98:
                parts.append(f'<text x="{bx + bar_w / 2:.1f}" y="{by - 8:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" font-weight="800" fill="{color}">{pct_text(value)}</text>')
        parts.append(f'<text x="{center:.1f}" y="{ry + rh + 30}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" fill="{ink}">{svg_escape(task_name)}</text>')
    parts.append(f'<rect x="1058" y="212" width="428" height="34" rx="17" fill="#ffffff" stroke="{border}"/>')
    for idx, (name, color) in enumerate(right_metrics):
        lx = 1082 + idx * 132
        ly = 233
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="16" height="11" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{lx + 24}" y="{ly}" font-family="Inter, Arial, sans-serif" font-size="12" font-weight="800" fill="{ink}">{svg_escape(name)}</text>')

    # Hard checks.
    panel(40, 560, 930, 290, "Response Quality Checks", "Strict pass rates from the existing backend evaluator")
    ordered_checks = [item for item in hard_checks if item["check"] in HARD_CHECKS]
    for idx, item in enumerate(ordered_checks):
        col = 0 if idx < 5 else 1
        row = idx if idx < 5 else idx - 5
        x = 78 + col * 445
        y = 635 + row * 42
        rate = float(item["strict_pass_rate"])
        failed = int(item["failed"])
        color = teal if rate >= 0.98 else (amber if rate >= 0.9 else rose)
        name = item["check"].replace("_", " ").title()
        parts.append(f'<text x="{x}" y="{y + 15}" font-family="Inter, Arial, sans-serif" font-size="12" font-weight="800" fill="{ink}">{svg_escape(name)}</text>')
        parts.append(f'<rect x="{x + 158}" y="{y}" width="220" height="18" rx="9" fill="#eee8df"/>')
        parts.append(f'<rect x="{x + 158}" y="{y}" width="{220 * rate:.1f}" height="18" rx="9" fill="{color}"/>')
        parts.append(f'<text x="{x + 392}" y="{y + 14}" font-family="Inter, Arial, sans-serif" font-size="12" font-weight="900" fill="{ink}">{pct_text(rate)}</text>')
        if failed:
            parts.append(f'<text x="{x + 392}" y="{y + 31}" font-family="Inter, Arial, sans-serif" font-size="10" fill="{rose}">{failed} fail</text>')

    # Integrity panel.
    panel(1000, 560, 560, 290, "Audit Integrity", "Shows what is really observed versus annotation-only")
    observed_count = len(OBSERVED_TASKS)
    proxy_count = len(PROXY_TASKS)
    not_measured_count = sum(1 for item in flags if item["severity"] == "high")
    review_count = sum(1 for item in flags if item["severity"] == "review")
    integrity_cards = [
        ("Observed", observed_count, teal, "from result JSON"),
        ("Proxy", proxy_count, blue, "measured indirectly"),
        ("Not measured", not_measured_count, rose, "annotation only"),
        ("Review", review_count, amber, "perfect-score check"),
    ]
    for idx, (name, count, color, note) in enumerate(integrity_cards):
        x = 1035 + (idx % 2) * 250
        y = 635 + (idx // 2) * 88
        parts.append(f'<rect x="{x}" y="{y}" width="220" height="68" rx="14" fill="#ffffff" stroke="{border}"/>')
        parts.append(f'<circle cx="{x + 26}" cy="{y + 26}" r="8" fill="{color}"/>')
        parts.append(f'<text x="{x + 46}" y="{y + 28}" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="900" fill="{ink}">{svg_escape(name)}</text>')
        parts.append(f'<text x="{x + 190}" y="{y + 31}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="22" font-weight="900" fill="{color}">{count}</text>')
        parts.append(f'<text x="{x + 46}" y="{y + 50}" font-family="Inter, Arial, sans-serif" font-size="11" fill="{muted}">{svg_escape(note)}</text>')

    # Bottom readout cards.
    card_y, card_w, card_h, gap = 880, 490, 158, 25
    cards = [
        (
            40,
            "Core Labels",
            teal,
            [("Conflict", pct_text(observed["conflict_type"]["accuracy"])), ("Resolvability", pct_text(observed["resolvability"]["accuracy"]))],
            "Saved outputs match the expected labels on this batch. Keep the review flag because perfect observed metrics need fresh-run confirmation.",
        ),
        (
            40 + card_w + gap,
            "Response Behavior",
            blue,
            [("Hard pass", pct_text(run["hard_pass_rate"])), ("Empathy A/B", f'{pct_text(check_map.get("empathy_a", {}).get("strict_pass_rate", 0))} / {pct_text(check_map.get("empathy_b", {}).get("strict_pass_rate", 0))}')],
            "The visible product quality is strong. Remaining misses are concentrated in response-level checks, not classification.",
        ),
        (
            40 + (card_w + gap) * 2,
            "Reliability",
            amber,
            [("Fallbacks", f"{run['fallbacks']} / {run['total_cases']}"), ("Errors", str(run["errors"]))],
            "Groq fallback is visible but controlled. The audit separates operational fallback behavior from core reasoning accuracy.",
        ),
    ]
    for x, title, color, rows, text in cards:
        parts.append(f'<rect x="{x}" y="{card_y}" width="{card_w}" height="{card_h}" rx="18" fill="{panel_fill}" stroke="{border}"/>')
        parts.append(f'<rect x="{x}" y="{card_y}" width="6" height="{card_h}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x + 26}" y="{card_y + 34}" font-family="Inter, Arial, sans-serif" font-size="17" font-weight="900" fill="{ink}">{svg_escape(title)}</text>')
        for ridx, (key, value) in enumerate(rows):
            y = card_y + 65 + ridx * 28
            parts.append(f'<text x="{x + 26}" y="{y}" font-family="Inter, Arial, sans-serif" font-size="12" fill="{muted}">{svg_escape(key)}</text>')
            parts.append(f'<text x="{x + 150}" y="{y}" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="900" fill="{ink}">{svg_escape(value)}</text>')
        for line_idx, line in enumerate(wrap_svg_text(text, 58)):
            parts.append(f'<text x="{x + 26}" y="{card_y + 122 + line_idx * 16}" font-family="Inter, Arial, sans-serif" font-size="11" fill="{muted}">{svg_escape(line)}</text>')

    parts.append(f'<text x="800" y="1072" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" fill="{muted}">Note: Strategy, intensity, repeat-pattern, and safety-sensitive annotations are reported as coverage unless independent actual predictions are saved.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def presentation_dashboard_chart(summary: dict[str, Any], hard_checks: list[dict[str, Any]]) -> str:
    """A larger, cleaner presentation dashboard with readable KPIs and legends."""
    observed = summary["observed_metrics"]
    proxy = summary["proxy_metrics"]
    run = summary["run_summary"]
    flags = summary["suspicious_or_not_measured"]
    check_map = {item["check"]: item for item in hard_checks}

    labels = ["Conflict", "Resolve", "Fallback", "Safety", "Empathy A", "Empathy B", "POV", "Overall"]
    accuracy = [
        observed["conflict_type"]["accuracy"],
        observed["resolvability"]["accuracy"],
        observed["should_use_fallback"]["accuracy"],
        proxy["output_safety_toxicity"]["accuracy"],
        check_map.get("empathy_a", {}).get("strict_pass_rate", 0),
        check_map.get("empathy_b", {}).get("strict_pass_rate", 0),
        check_map.get("pov_correct", {}).get("strict_pass_rate", 0),
        run["hard_pass_rate"],
    ]
    macro_f1 = [
        observed["conflict_type"]["macro_f1"],
        observed["resolvability"]["macro_f1"],
        observed["should_use_fallback"]["macro_f1"],
        proxy["output_safety_toxicity"]["macro_f1"],
        check_map.get("empathy_a", {}).get("weighted_pass_rate", 0),
        check_map.get("empathy_b", {}).get("weighted_pass_rate", 0),
        check_map.get("pov_correct", {}).get("weighted_pass_rate", 0),
        run["hard_pass_rate"],
    ]
    weighted_f1 = [
        observed["conflict_type"]["weighted_f1"],
        observed["resolvability"]["weighted_f1"],
        observed["should_use_fallback"]["weighted_f1"],
        proxy["output_safety_toxicity"]["weighted_f1"],
        check_map.get("empathy_a", {}).get("strict_pass_rate", 0),
        check_map.get("empathy_b", {}).get("strict_pass_rate", 0),
        check_map.get("pov_correct", {}).get("strict_pass_rate", 0),
        run["hard_pass_rate"],
    ]

    width, height = 1800, 1260
    bg = "#f7f3ec"
    panel_fill = "#fffdf8"
    ink = "#182127"
    muted = "#657079"
    grid = "#e9e0d3"
    border = "#d8ccbb"
    blue = "#2474a6"
    amber = "#c77918"
    teal = "#178b7f"
    rose = "#c45a6d"
    green = "#2f9b63"
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
    ]

    def panel(x: int, y: int, w: int, h: int, title: str, subtitle: str = "") -> None:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="{panel_fill}" stroke="{border}" stroke-width="1.4"/>')
        parts.append(f'<text x="{x + 28}" y="{y + 42}" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="900" fill="{ink}">{svg_escape(title)}</text>')
        if subtitle:
            parts.append(f'<text x="{x + 28}" y="{y + 70}" font-family="Inter, Arial, sans-serif" font-size="14" fill="{muted}">{svg_escape(subtitle)}</text>')

    def y_for(value: float, y: int, h: int) -> float:
        return y + h - max(0, min(1, value)) * h

    def axes(x: int, y: int, w: int, h: int) -> None:
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            yy = y_for(tick, y, h)
            parts.append(f'<line x1="{x}" y1="{yy:.1f}" x2="{x + w}" y2="{yy:.1f}" stroke="{grid}"/>')
            parts.append(f'<text x="{x - 14}" y="{yy + 5:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="13" fill="{muted}">{pct_text(tick)}</text>')
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="{border}"/>')
        parts.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="{border}"/>')

    def legend(x: int, y: int, items: list[tuple[str, str]]) -> None:
        total_w = sum(52 + len(name) * 8 for name, _ in items)
        parts.append(f'<rect x="{x}" y="{y}" width="{total_w}" height="40" rx="20" fill="#ffffff" stroke="{border}"/>')
        cx = x + 18
        for name, color in items:
            parts.append(f'<rect x="{cx}" y="{y + 14}" width="17" height="12" rx="4" fill="{color}"/>')
            parts.append(f'<text x="{cx + 25}" y="{y + 25}" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="800" fill="{ink}">{svg_escape(name)}</text>')
            cx += 52 + len(name) * 8

    def kpi(x: int, label_name: str, value: str, color: str, note: str) -> None:
        parts.append(f'<rect x="{x}" y="54" width="150" height="72" rx="18" fill="#ffffff" stroke="{border}"/>')
        parts.append(f'<circle cx="{x + 24}" cy="80" r="6" fill="{color}"/>')
        parts.append(f'<text x="{x + 42}" y="84" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="800" fill="{muted}">{svg_escape(label_name)}</text>')
        parts.append(f'<text x="{x + 24}" y="112" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="900" fill="{ink}">{svg_escape(value)}</text>')
        parts.append(f'<text x="{x + 88}" y="112" font-family="Inter, Arial, sans-serif" font-size="11" fill="{muted}">{svg_escape(note)}</text>')

    parts.append(f'<rect x="52" y="34" width="1696" height="124" rx="26" fill="{panel_fill}" stroke="{border}" stroke-width="1.4"/>')
    parts.append(f'<text x="88" y="85" font-family="Inter, Arial, sans-serif" font-size="34" font-weight="900" fill="{ink}">ConcordAI Evaluation Dashboard</text>')
    parts.append(f'<text x="88" y="116" font-family="Inter, Arial, sans-serif" font-size="14" fill="{muted}">Leakage-aware ML report from saved backend evaluation batches.</text>')
    parts.append(f'<text x="88" y="138" font-family="Inter, Arial, sans-serif" font-size="14" fill="{muted}">Perfect metrics are shown with review context, not treated as blind proof.</text>')
    kpi(1188, "Cases", str(run["total_cases"]), blue, "evaluated")
    kpi(1354, "Hard pass", pct_text(run["hard_pass_rate"]), teal, "strict")
    kpi(1520, "Fallbacks", str(run["fallbacks"]), amber, "used")

    panel(52, 190, 1040, 430, "Performance Profile", "Accuracy and F1 across observed/proxy signals")
    legend(730, 226, [("Accuracy", blue), ("Macro F1", amber), ("Weighted F1", green)])
    x0, y0, chart_w, chart_h = 128, 300, 880, 230
    axes(x0, y0, chart_w, chart_h)
    for values, color in [(accuracy, blue), (macro_f1, amber), (weighted_f1, green)]:
        points = metric_xy(labels, values, x0=x0, y0=y0, width=chart_w, height=chart_h)
        parts.append(f'<path d="{line_path(points)}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
        for idx, (px, py) in enumerate(points):
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{color}" stroke="{panel_fill}" stroke-width="2"/>')
            if values[idx] < 0.94:
                parts.append(f'<text x="{px:.1f}" y="{py - 13:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="900" fill="{color}">{pct_text(values[idx])}</text>')
    for idx, name in enumerate(labels):
        px = x0 + idx * (chart_w / max(1, len(labels) - 1))
        parts.append(f'<text x="{px:.1f}" y="{y0 + chart_h + 34}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="{ink}">{svg_escape(name)}</text>')

    panel(1128, 190, 620, 430, "Observed Task Scores", "Weighted precision, recall, and F1 from saved result fields")
    legend(1206, 262, [("Precision", blue), ("Recall", amber), ("F1", green)])
    rx, ry, rw, rh = 1206, 338, 450, 190
    axes(rx, ry, rw, rh)
    task_metrics = [("Conflict", observed["conflict_type"]), ("Resolve", observed["resolvability"]), ("Fallback", observed["should_use_fallback"])]
    group_w = rw / len(task_metrics)
    bar_w = 34
    for idx, (task_name, metric) in enumerate(task_metrics):
        center = rx + idx * group_w + group_w / 2
        values = [weighted_class_metric(metric, "precision"), weighted_class_metric(metric, "recall"), metric["weighted_f1"]]
        for bidx, value in enumerate(values):
            color = [blue, amber, green][bidx]
            bh = max(0, min(1, value)) * rh
            bx = center + (bidx - 1) * (bar_w + 11) - bar_w / 2
            by = ry + rh - bh
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" rx="6" fill="{color}"/>')
            if value < 0.98:
                parts.append(f'<text x="{bx + bar_w / 2:.1f}" y="{by - 10:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="900" fill="{color}">{pct_text(value)}</text>')
        parts.append(f'<text x="{center:.1f}" y="{ry + rh + 34}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="14" fill="{ink}">{svg_escape(task_name)}</text>')

    panel(52, 658, 1040, 330, "Response Quality Checks", "Strict pass rates from the backend evaluator")
    ordered_checks = [item for item in hard_checks if item["check"] in HARD_CHECKS]
    for idx, item in enumerate(ordered_checks):
        col = 0 if idx < 5 else 1
        row = idx if idx < 5 else idx - 5
        x = 98 + col * 504
        y = 738 + row * 50
        rate = float(item["strict_pass_rate"])
        failed = int(item["failed"])
        color = teal if rate >= 0.98 else (amber if rate >= 0.9 else rose)
        name = item["check"].replace("_", " ").title()
        parts.append(f'<text x="{x}" y="{y + 17}" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="900" fill="{ink}">{svg_escape(name)}</text>')
        parts.append(f'<rect x="{x + 172}" y="{y}" width="260" height="22" rx="11" fill="#eee8df"/>')
        parts.append(f'<rect x="{x + 172}" y="{y}" width="{260 * rate:.1f}" height="22" rx="11" fill="{color}"/>')
        parts.append(f'<text x="{x + 450}" y="{y + 17}" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="900" fill="{ink}">{pct_text(rate)}</text>')
        if failed:
            parts.append(f'<text x="{x + 450}" y="{y + 35}" font-family="Inter, Arial, sans-serif" font-size="11" fill="{rose}">{failed} fail</text>')

    panel(1128, 658, 620, 330, "Audit Integrity", "Measured signals versus annotation-only labels")
    observed_count = len(OBSERVED_TASKS)
    proxy_count = len(PROXY_TASKS)
    not_measured_count = sum(1 for item in flags if item["severity"] == "high")
    review_count = sum(1 for item in flags if item["severity"] == "review")
    cards = [("Observed", observed_count, teal, "direct fields"), ("Proxy", proxy_count, blue, "indirect"), ("Not measured", not_measured_count, rose, "annotation-only"), ("Review", review_count, amber, "perfect-score")]
    for idx, (name, count, color, note) in enumerate(cards):
        x = 1170 + (idx % 2) * 282
        y = 748 + (idx // 2) * 102
        parts.append(f'<rect x="{x}" y="{y}" width="244" height="78" rx="16" fill="#ffffff" stroke="{border}"/>')
        parts.append(f'<circle cx="{x + 30}" cy="{y + 30}" r="9" fill="{color}"/>')
        parts.append(f'<text x="{x + 54}" y="{y + 33}" font-family="Inter, Arial, sans-serif" font-size="15" font-weight="900" fill="{ink}">{svg_escape(name)}</text>')
        parts.append(f'<text x="{x + 210}" y="{y + 36}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="28" font-weight="900" fill="{color}">{count}</text>')
        parts.append(f'<text x="{x + 54}" y="{y + 57}" font-family="Inter, Arial, sans-serif" font-size="12" fill="{muted}">{svg_escape(note)}</text>')

    bottom_cards = [
        ("Core Labels", teal, [("Conflict", pct_text(observed["conflict_type"]["accuracy"])), ("Resolvability", pct_text(observed["resolvability"]["accuracy"]))], "Saved outputs match expected labels on this batch. Perfect scores remain review-marked until fresh independent runs confirm them."),
        ("Response Behavior", blue, [("Hard pass", pct_text(run["hard_pass_rate"])), ("Empathy A/B", f'{pct_text(check_map.get("empathy_a", {}).get("strict_pass_rate", 0))} / {pct_text(check_map.get("empathy_b", {}).get("strict_pass_rate", 0))}')], "Visible product quality is strong. Remaining misses concentrate in response-level checks, not core labels."),
        ("Reliability", amber, [("Fallbacks", f"{run['fallbacks']} / {run['total_cases']}"), ("Errors", str(run["errors"]))], "Fallback use is visible and controlled. The audit separates operational fallback behavior from core reasoning accuracy."),
    ]
    for idx, (title, color, rows, text) in enumerate(bottom_cards):
        x = 52 + idx * 568
        y = 1030
        parts.append(f'<rect x="{x}" y="{y}" width="544" height="158" rx="20" fill="{panel_fill}" stroke="{border}"/>')
        parts.append(f'<rect x="{x}" y="{y}" width="7" height="158" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{x + 32}" y="{y + 38}" font-family="Inter, Arial, sans-serif" font-size="20" font-weight="900" fill="{ink}">{svg_escape(title)}</text>')
        for ridx, (key, value) in enumerate(rows):
            yy = y + 70 + ridx * 26
            parts.append(f'<text x="{x + 32}" y="{yy}" font-family="Inter, Arial, sans-serif" font-size="13" fill="{muted}">{svg_escape(key)}</text>')
            parts.append(f'<text x="{x + 170}" y="{yy}" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="900" fill="{ink}">{svg_escape(value)}</text>')
        for line_idx, line in enumerate(wrap_svg_text(text, 54)):
            parts.append(f'<text x="{x + 32}" y="{y + 122 + line_idx * 16}" font-family="Inter, Arial, sans-serif" font-size="12" fill="{muted}">{svg_escape(line)}</text>')

    parts.append(f'<text x="900" y="1228" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" fill="{muted}">Note: Strategy, intensity, repeat-pattern, and safety-sensitive annotations are reported as coverage unless independent actual predictions are saved.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def wrap_svg_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(item) for item in current) + len(current) + len(word) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines[:3]


def write_chart_index(out_dir: Path, chart_names: list[str]) -> None:
    sections = "\n".join(
        f'<section><h2>{svg_escape(name.replace("_", " ").replace(".svg", "").title())}</h2><img src="charts/{svg_escape(name)}" alt="{svg_escape(name)}"></section>'
        for name in chart_names
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ConcordAI Evaluation Audit</title>
  <style>
    body {{ margin: 0; padding: 32px; background: #f3f0ea; color: #1f252b; font-family: Inter, Arial, sans-serif; }}
    main {{ max-width: 1840px; margin: 0 auto; }}
    h1 {{ margin: 0 0 20px; }}
    section {{ margin-bottom: 28px; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    img {{ width: 100%; height: auto; display: block; border: 1px solid #d4cabb; border-radius: 18px; background: #fbfaf7; }}
  </style>
</head>
<body>
  <main>
    <h1>ConcordAI Evaluation Audit</h1>
    {sections}
  </main>
</body>
</html>
"""
    (out_dir / "charts.html").write_text(html_text, encoding="utf-8")


def write_charts(out_dir: Path, summary: dict[str, Any], hard_checks: list[dict[str, Any]]) -> None:
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "ml_style_dashboard.svg": presentation_dashboard_chart(summary, hard_checks),
        "honest_performance_summary.svg": honest_summary_chart(summary, hard_checks),
        "hard_check_breakdown.svg": hard_checks_chart(hard_checks),
        "metric_provenance.svg": provenance_chart(summary),
    }
    for name, content in charts.items():
        (charts_dir / name).write_text(content, encoding="utf-8")
    write_chart_index(out_dir, list(charts))


def write_report(
    *,
    out_dir: Path,
    cases_file: Path,
    result_files: list[Path],
    rows: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    observed = {task: classification_metrics(observed_pairs(rows, task)) for task in OBSERVED_TASKS}
    proxy = {task: classification_metrics(observed_pairs(rows, task)) for task in PROXY_TASKS}
    soft = soft_check_coverage(rows)
    hard_checks = hard_check_rates(rows)
    distribution = dataset_distribution(cases)
    flags = suspicious_metrics(rows, observed, soft)
    mismatch_rows = mismatches(rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_file": str(cases_file),
        "result_files": [str(path) for path in result_files],
        "message": "This audit separates independently observed metrics from proxy or non-measured annotation metrics.",
        "run_summary": hard_run_summary(rows),
        "observed_metrics": observed,
        "proxy_metrics": proxy,
        "annotation_metric_coverage": soft,
        "suspicious_or_not_measured": flags,
        "dataset_distribution": distribution,
    }
    (out_dir / "honest_summary.json").write_text(json.dumps(round_floats(summary), indent=2), encoding="utf-8")

    metric_rows: list[dict[str, Any]] = []
    for source, metrics in (("observed", observed), ("proxy", proxy)):
        for task, metric in metrics.items():
            metric_rows.append(
                {
                    "source": source,
                    "task": task,
                    "total": metric["total"],
                    "accuracy": round(metric["accuracy"], 4),
                    "micro_f1": round(metric["micro_f1"], 4),
                    "macro_f1": round(metric["macro_f1"], 4),
                    "weighted_f1": round(metric["weighted_f1"], 4),
                    "mismatch_count": metric["mismatch_count"],
                }
            )
    write_csv(out_dir / "honest_task_metrics.csv", metric_rows, ["source", "task", "total", "accuracy", "micro_f1", "macro_f1", "weighted_f1", "mismatch_count"])
    write_csv(out_dir / "hard_check_rates.csv", [round_floats(row) for row in hard_checks], ["check", "total", "passed", "warnings", "failed", "strict_pass_rate", "weighted_pass_rate"])
    write_csv(out_dir / "annotation_metric_coverage.csv", [dict(task=task, **data) for task, data in soft.items()], ["task", "total", "present", "missing", "matches", "mismatches", "status", "warning"])
    write_csv(out_dir / "honest_mismatches.csv", mismatch_rows, ["task", "id", "expected", "actual", "confidence", "fallback_used", "passed", "category", "description", "error"])
    write_csv(out_dir / "suspicious_metrics.csv", flags, ["task", "severity", "reason", "source"])
    write_csv(out_dir / "dataset_distribution.csv", [round_floats(row) for row in distribution], ["field", "label", "count", "percent"])
    write_charts(out_dir, summary, hard_checks)
    return summary


def print_report(summary: dict[str, Any]) -> None:
    run = summary["run_summary"]
    print("\nConcordAI Evaluation Audit")
    print("=" * 72)
    print(f"Total cases      : {run['total_cases']}")
    print(f"Passed / Failed  : {run['passed']} / {run['failed']}")
    print(f"Errors           : {run['errors']}")
    print(f"Fallbacks        : {run['fallbacks']}")
    print(f"Hard pass rate   : {run['hard_pass_rate']:.1%}")
    print("\nIndependently Observed Metrics")
    print("-" * 72)
    print(f"{'Task':<24} {'Accuracy':>9} {'MacroF1':>9} {'WeightedF1':>11} {'Mismatch':>9}")
    for task, metric in summary["observed_metrics"].items():
        print(f"{task:<24} {metric['accuracy']:>9.1%} {metric['macro_f1']:>9.1%} {metric['weighted_f1']:>11.1%} {metric['mismatch_count']:>9}")
    print("\nProxy Metrics")
    print("-" * 72)
    for task, metric in summary["proxy_metrics"].items():
        print(f"{task:<24} {metric['accuracy']:>9.1%} {metric['macro_f1']:>9.1%} {metric['weighted_f1']:>11.1%} {metric['mismatch_count']:>9}")
    print("\nNot Independently Measured / Leakage Risk")
    print("-" * 72)
    for item in summary["suspicious_or_not_measured"]:
        print(f"{item['severity'].upper():<7} {item['task']:<24} {item['reason']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-aware audit for backend eval result files.")
    parser.add_argument("--results", nargs="+", required=True, help="One or more backend_eval.py result JSON files.")
    parser.add_argument("--cases", default="test_cases.json", help="Ground-truth test cases JSON file.")
    parser.add_argument("--out", default="honest_eval_report", help="Output directory for audit artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_files = [Path(path) for path in args.results]
    cases_file = Path(args.cases)
    try:
        cases = load_cases(cases_file)
        results = load_results(result_files)
        rows = build_rows(cases, results)
        if not rows:
            raise ValueError("No result rows found.")
        summary = write_report(
            out_dir=Path(args.out),
            cases_file=cases_file,
            result_files=result_files,
            rows=rows,
            cases=cases,
        )
        print_report(summary)
        print(f"\n[audit] wrote honest audit to {args.out}")
        return 0
    except Exception as exc:
        print(f"[audit:error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
