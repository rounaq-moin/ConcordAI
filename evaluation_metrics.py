"""ML-style metrics for saved backend evaluation runs.

This script is intentionally standalone. It reads saved backend_eval.py JSON
results and test_cases.json, then writes reusable metrics artifacts without
importing the mediation pipeline or any heavy runtime dependencies.

Examples:
    python evaluation_metrics.py --results all_100_after.json
    python evaluation_metrics.py --results b01.json b02.json b03.json --out metrics_report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPEAT_CUES = ["again", "always", "keeps", "keep", "every time", "for months", "for years", "third time", "over and over"]
NEGATION_CUES = ["not always", "not anymore", "no longer"]
SAFETY_CUES = [
    "afraid",
    "scared",
    "unsafe",
    "threat",
    "threatened",
    "shouting",
    "yelling",
    "screaming",
    "harm",
    "in danger",
    "better off without me",
    "better off if i disappeared",
    "disappeared",
    "blocked the doorway",
    "trapped",
]
ESCALATION_CUES = ["always", "never", "ruin", "ruined", "can't stand", "fed up", "enough", "again"]
ETHICS_CUES = [
    "fraud",
    "illegal",
    "not ethical",
    "against my conscience",
    "i will not",
    "i refuse",
    "will not sign",
    "whistleblow",
    "harassment",
    "audit",
    "refuse to serve",
    "end of life",
    "end-of-life",
    "medical treatment",
]
HIGH_STAKES_CUES = [
    "fraud",
    "illegal",
    "audit",
    "laying off",
    "forty people",
    "end of life",
    "end-of-life",
    "medical treatment",
    "whistleblow",
    "harassment",
    "security",
    "legal exposure",
    "unsafe",
    "in danger",
]
STRATEGY_BY_CONFLICT = {
    "emotional": "emotional_validation",
    "misunderstanding": "clarify_intent",
    "value": "boundary_setting",
    "logical": "tradeoff_alignment",
}

TASKS = [
    "conflict_type",
    "resolvability",
    "expected_strategy",
    "is_safety_sensitive",
    "intensity",
    "is_repeat_pattern",
    "should_use_fallback",
]


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _as_list(payload: Any, *, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    raise ValueError(f"{source} must contain a JSON list or an object with a 'results' list.")


def _label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "unknown"
    return str(value)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _combined_text(case: dict[str, Any]) -> str:
    return " ".join((str(case.get("text_a", "")) + " " + str(case.get("text_b", ""))).lower().split())


def _cue_hits(text: str, cues: list[str]) -> int:
    return sum(1 for cue in cues if cue in text)


def infer_repeat_pattern(combined: str) -> bool:
    repeat_hits = _cue_hits(combined, REPEAT_CUES)
    negation_hits = _cue_hits(combined, NEGATION_CUES)
    return max(0, repeat_hits - negation_hits) > 0


def infer_ethics_refusal(combined: str) -> bool:
    return any(cue in combined for cue in ETHICS_CUES)


def infer_safety_sensitive(combined: str, safety_score: float | None) -> bool:
    if safety_score is not None and safety_score >= 0.4:
        return True
    return any(cue in combined for cue in SAFETY_CUES)


def infer_strategy(conflict_type: str, *, safety_sensitive: bool, ethics_refusal: bool) -> str:
    if safety_sensitive:
        return "emotional_validation"
    if ethics_refusal:
        return "boundary_setting"
    return STRATEGY_BY_CONFLICT.get(conflict_type, "emotional_validation")


def infer_intensity(
    combined: str,
    conflict_type: str,
    *,
    safety_sensitive: bool,
    repeat_pattern: bool,
    ethics_refusal: bool,
) -> str:
    score = 0
    if safety_sensitive:
        score += 2
    if repeat_pattern:
        score += 1
    if conflict_type == "value" or ethics_refusal:
        score += 1
    if _cue_hits(combined, ESCALATION_CUES) or _cue_hits(combined, HIGH_STAKES_CUES):
        score += 1
    score = min(score, 4)
    if score <= 1:
        return "low"
    if score <= 3:
        return "medium"
    return "high"


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    cases = _as_list(_json_load(path), source=path)
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
        raise ValueError(f"{path} contains duplicate case ids: {', '.join(sorted(set(duplicates)))}")
    return by_id


def load_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}
    duplicates: list[str] = []
    for path in paths:
        results = _as_list(_json_load(path), source=path)
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


def soft_actual(result: dict[str, Any], name: str) -> Any:
    soft = result.get("soft_checks")
    if isinstance(soft, dict):
        item = soft.get(name)
        if isinstance(item, dict) and "actual" in item:
            return item["actual"]
    return None


def actuals_for_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    combined = _combined_text(case)
    safety_score = result.get("safety_score")
    try:
        safety_value = float(safety_score) if safety_score is not None else None
    except (TypeError, ValueError):
        safety_value = None

    conflict_type = result.get("conflict_type", "unknown")
    repeat_pattern = soft_actual(result, "is_repeat_pattern")
    if repeat_pattern is None:
        repeat_pattern = infer_repeat_pattern(combined)
    repeat_pattern = _boolish(repeat_pattern)

    ethics_refusal = infer_ethics_refusal(combined)
    safety_sensitive = soft_actual(result, "is_safety_sensitive")
    if safety_sensitive is None:
        safety_sensitive = infer_safety_sensitive(combined, safety_value)
    safety_sensitive = _boolish(safety_sensitive)

    strategy = soft_actual(result, "expected_strategy")
    if strategy is None:
        strategy = infer_strategy(
            str(conflict_type),
            safety_sensitive=safety_sensitive,
            ethics_refusal=ethics_refusal,
        )

    intensity = soft_actual(result, "intensity")
    if intensity is None:
        intensity = infer_intensity(
            combined,
            str(conflict_type),
            safety_sensitive=safety_sensitive,
            repeat_pattern=repeat_pattern,
            ethics_refusal=ethics_refusal,
        )

    fallback = soft_actual(result, "should_use_fallback")
    if fallback is None:
        fallback = result.get("fallback_used", False)

    return {
        "conflict_type": conflict_type,
        "resolvability": result.get("resolvability", "unknown"),
        "expected_strategy": strategy,
        "is_safety_sensitive": safety_sensitive,
        "intensity": intensity,
        "is_repeat_pattern": repeat_pattern,
        "should_use_fallback": _boolish(fallback),
    }


def expected_for_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "conflict_type": case.get("expected_conflict_type", "unknown"),
        "resolvability": case.get("expected_resolvability", "unknown"),
        "expected_strategy": case.get(
            "expected_strategy",
            STRATEGY_BY_CONFLICT.get(case.get("expected_conflict_type", ""), "emotional_validation"),
        ),
        "is_safety_sensitive": _boolish(case.get("is_safety_sensitive", False)),
        "intensity": case.get("intensity", "unknown"),
        "is_repeat_pattern": _boolish(case.get("is_repeat_pattern", False)),
        "should_use_fallback": _boolish(case.get("should_use_fallback", False)),
    }


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: pos for pos, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for expected, actual in zip(y_true, y_pred):
        matrix[index[expected]][index[actual]] += 1
    return matrix


def normalize_matrix(matrix: list[list[int]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in matrix:
        total = sum(row)
        if total == 0:
            normalized.append([0.0 for _ in row])
        else:
            normalized.append([round(value / total, 4) for value in row])
    return normalized


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def task_metrics(task: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [_label(row["expected"][task]) for row in rows]
    y_pred = [_label(row["actual"][task]) for row in rows]
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels)
    total = len(y_true)
    correct = sum(1 for expected, actual in zip(y_true, y_pred) if expected == actual)
    accuracy = safe_div(correct, total)

    per_class: list[dict[str, Any]] = []
    for pos, label in enumerate(labels):
        tp = matrix[pos][pos]
        fp = sum(matrix[row][pos] for row in range(len(labels)) if row != pos)
        fn = sum(matrix[pos][col] for col in range(len(labels)) if col != pos)
        support = sum(matrix[pos])
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class.append(
            {
                "task": task,
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    weighted_precision = safe_div(sum(item["precision"] * item["support"] for item in per_class), total)
    weighted_recall = safe_div(sum(item["recall"] * item["support"] for item in per_class), total)
    weighted_f1 = safe_div(sum(item["f1"] * item["support"] for item in per_class), total)
    macro_precision = safe_div(sum(item["precision"] for item in per_class), len(per_class))
    macro_recall = safe_div(sum(item["recall"] for item in per_class), len(per_class))
    macro_f1 = safe_div(sum(item["f1"] for item in per_class), len(per_class))

    # For single-label classification, micro F1 equals micro precision/recall.
    total_tp = correct
    total_fp = total - correct
    total_fn = total - correct
    micro_precision = safe_div(total_tp, total_tp + total_fp)
    micro_recall = safe_div(total_tp, total_tp + total_fn)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    return {
        "task": task,
        "total": total,
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "labels": labels,
        "confusion_matrix": matrix,
        "confusion_matrix_normalized": normalize_matrix(matrix),
        "per_class": per_class,
        "mismatch_count": total - correct,
    }


def build_rows(cases: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    unknown_result_ids = sorted(set(results) - set(cases))
    if unknown_result_ids:
        raise ValueError("Results contain ids that are not in test_cases.json: " + ", ".join(unknown_result_ids))

    missing_result_ids = sorted(set(cases) - set(results), key=_case_sort_key)
    rows: list[dict[str, Any]] = []
    for case_id in sorted(results, key=_case_sort_key):
        case = cases[case_id]
        result = results[case_id]
        rows.append(
            {
                "id": case_id,
                "category": case.get("category", ""),
                "description": case.get("description", ""),
                "expected": expected_for_case(case),
                "actual": actuals_for_case(case, result),
                "result": result,
            }
        )
    return rows, missing_result_ids


def _case_sort_key(case_id: str) -> tuple[str, int, str]:
    match = re.match(r"^([A-Za-z]+)(\d+)$", case_id)
    if not match:
        return (case_id, -1, case_id)
    return (match.group(1), int(match.group(2)), case_id)


def run_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if _boolish(row["result"].get("passed")))
    errors = sum(1 for row in rows if row["result"].get("error"))
    fallbacks = sum(1 for row in rows if _boolish(row["result"].get("fallback_used")))
    latencies = [float(row["result"].get("elapsed", 0) or 0) for row in rows]
    confidences = [float(row["result"].get("confidence", 0) or 0) for row in rows]
    return {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "errors": errors,
        "fallbacks": fallbacks,
        "hard_pass_rate": safe_div(passed, total),
        "avg_latency_seconds": safe_div(sum(latencies), len(latencies)),
        "avg_confidence": safe_div(sum(confidences), len(confidences)),
    }


def distribution_rows(cases: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluated_ids = {row["id"] for row in rows}
    scopes = [
        ("all_cases", list(cases.values())),
        ("evaluated_cases", [cases[case_id] for case_id in sorted(evaluated_ids, key=_case_sort_key)]),
    ]
    fields = {
        "conflict_type": "expected_conflict_type",
        "resolvability": "expected_resolvability",
        "strategy": "expected_strategy",
        "intensity": "intensity",
        "safety_sensitive": "is_safety_sensitive",
        "repeat_pattern": "is_repeat_pattern",
        "should_use_fallback": "should_use_fallback",
    }
    out: list[dict[str, Any]] = []
    for scope, scope_cases in scopes:
        for field, key in fields.items():
            counts = Counter(_label(case.get(key, "unknown")) for case in scope_cases)
            total = sum(counts.values())
            for label, count in sorted(counts.items()):
                out.append(
                    {
                        "scope": scope,
                        "field": field,
                        "label": label,
                        "count": count,
                        "percent": safe_div(count, total),
                    }
                )
    return out


def mismatch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        result = row["result"]
        for task in TASKS:
            expected = _label(row["expected"][task])
            actual = _label(row["actual"][task])
            if expected != actual:
                mismatches.append(
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
    return mismatches


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rounded(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return round(value, 4)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def write_artifacts(
    *,
    out_dir: Path,
    cases_file: Path,
    result_files: list[Path],
    rows: list[dict[str, Any]],
    missing_result_ids: list[str],
    task_results: list[dict[str, Any]],
    distributions: list[dict[str, Any]],
    make_charts: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_file": str(cases_file),
        "result_files": [str(path) for path in result_files],
        "run_summary": run_summary(rows),
        "missing_result_ids": missing_result_ids,
        "tasks": {
            item["task"]: {
                key: value
                for key, value in item.items()
                if key not in {"per_class"}
            }
            for item in task_results
        },
        "dataset_distribution": distributions,
    }
    (out_dir / "summary.json").write_text(json.dumps(rounded(summary), indent=2), encoding="utf-8")

    task_rows = [
        {
            "task": item["task"],
            "total": item["total"],
            "accuracy": item["accuracy"],
            "micro_f1": item["micro_f1"],
            "macro_precision": item["macro_precision"],
            "macro_recall": item["macro_recall"],
            "macro_f1": item["macro_f1"],
            "weighted_precision": item["weighted_precision"],
            "weighted_recall": item["weighted_recall"],
            "weighted_f1": item["weighted_f1"],
            "mismatch_count": item["mismatch_count"],
        }
        for item in task_results
    ]
    write_csv(
        out_dir / "task_metrics.csv",
        [rounded(row) for row in task_rows],
        [
            "task",
            "total",
            "accuracy",
            "micro_f1",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "mismatch_count",
        ],
    )
    per_class = [rounded(row) for item in task_results for row in item["per_class"]]
    write_csv(out_dir / "per_class_metrics.csv", per_class, ["task", "label", "precision", "recall", "f1", "support"])
    write_csv(
        out_dir / "mismatches.csv",
        mismatch_rows(rows),
        ["task", "id", "expected", "actual", "confidence", "fallback_used", "passed", "category", "description", "error"],
    )
    write_csv(
        out_dir / "dataset_distribution.csv",
        [rounded(row) for row in distributions],
        ["scope", "field", "label", "count", "percent"],
    )
    if make_charts:
        write_charts(out_dir / "charts", task_results)


def write_charts(charts_dir: Path, task_results: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[charts] skipped: matplotlib unavailable ({exc})")
        return

    charts_dir.mkdir(parents=True, exist_ok=True)
    task_names = [item["task"] for item in task_results]
    accuracy = [item["accuracy"] for item in task_results]
    weighted_f1 = [item["weighted_f1"] for item in task_results]
    macro_f1 = [item["macro_f1"] for item in task_results]

    fig, ax = plt.subplots(figsize=(max(8, len(task_names) * 1.2), 5))
    xs = list(range(len(task_names)))
    width = 0.25
    ax.bar([x - width for x in xs], accuracy, width, label="Accuracy")
    ax.bar(xs, weighted_f1, width, label="Weighted F1")
    ax.bar([x + width for x in xs], macro_f1, width, label="Macro F1")
    ax.set_ylim(0, 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(task_names, rotation=30, ha="right")
    ax.set_title("Evaluation Metrics by Task")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "task_metrics.png", dpi=160)
    plt.close(fig)

    for item in task_results:
        for normalized in (False, True):
            matrix = item["confusion_matrix_normalized"] if normalized else item["confusion_matrix"]
            labels = item["labels"]
            fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.2), max(4, len(labels))))
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else None)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Expected")
            title_suffix = "Normalized" if normalized else "Raw"
            ax.set_title(f"{item['task']} Confusion Matrix ({title_suffix})")
            for row_idx, row in enumerate(matrix):
                for col_idx, value in enumerate(row):
                    text = f"{value:.0%}" if normalized else str(value)
                    ax.text(col_idx, row_idx, text, ha="center", va="center", color="#111")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            suffix = "normalized" if normalized else "raw"
            fig.savefig(charts_dir / f"{item['task']}_confusion_{suffix}.png", dpi=160)
            plt.close(fig)


def print_console(task_results: list[dict[str, Any]], rows: list[dict[str, Any]], distributions: list[dict[str, Any]], missing: list[str]) -> None:
    summary = run_summary(rows)
    print("\nML Evaluation Metrics")
    print("=" * 72)
    print(f"Total cases      : {summary['total_cases']}")
    print(f"Passed / Failed  : {summary['passed']} / {summary['failed']}")
    print(f"Errors           : {summary['errors']}")
    print(f"Fallbacks        : {summary['fallbacks']}")
    print(f"Hard pass rate   : {summary['hard_pass_rate']:.1%}")
    print(f"Avg confidence   : {summary['avg_confidence']:.1%}")
    if missing:
        preview = ", ".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        print(f"Missing results  : {len(missing)} ({preview}{suffix})")

    print("\nTask Metrics")
    print("-" * 104)
    print(f"{'Task':<24} {'Acc':>7} {'MicroF1':>8} {'MacroF1':>8} {'WeightedF1':>11} {'Mismatch':>9}")
    for item in task_results:
        print(
            f"{item['task']:<24} "
            f"{item['accuracy']:>7.1%} "
            f"{item['micro_f1']:>8.1%} "
            f"{item['macro_f1']:>8.1%} "
            f"{item['weighted_f1']:>11.1%} "
            f"{item['mismatch_count']:>9}"
        )

    print("\nEvaluated Dataset Distribution")
    print("-" * 72)
    evaluated = [row for row in distributions if row["scope"] == "evaluated_cases"]
    for field in sorted({row["field"] for row in evaluated}):
        parts = [
            f"{row['label']}={row['count']} ({row['percent']:.0%})"
            for row in evaluated
            if row["field"] == field
        ]
        print(f"{field:<22} " + ", ".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ML-style metrics from saved backend_eval.py results.")
    parser.add_argument("--results", nargs="+", required=True, help="One or more backend_eval.py result JSON files.")
    parser.add_argument("--cases", default="test_cases.json", help="Ground-truth test cases JSON file.")
    parser.add_argument("--out", default="metrics_report", help="Output directory for metrics artifacts.")
    parser.add_argument("--no-charts", action="store_true", help="Skip optional PNG chart generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_files = [Path(path) for path in args.results]
    cases_file = Path(args.cases)
    out_dir = Path(args.out)

    try:
        if not cases_file.exists():
            raise FileNotFoundError(f"Cases file not found: {cases_file}")
        for path in result_files:
            if not path.exists():
                raise FileNotFoundError(f"Results file not found: {path}")

        cases = load_cases(cases_file)
        results = load_results(result_files)
        rows, missing = build_rows(cases, results)
        if not rows:
            raise ValueError("No result rows found.")
        task_results = [task_metrics(task, rows) for task in TASKS]
        distributions = distribution_rows(cases, rows)
        print_console(task_results, rows, distributions, missing)
        write_artifacts(
            out_dir=out_dir,
            cases_file=cases_file,
            result_files=result_files,
            rows=rows,
            missing_result_ids=missing,
            task_results=task_results,
            distributions=distributions,
            make_charts=not args.no_charts,
        )
        print(f"\n[metrics] wrote report to {out_dir}")
        return 0
    except Exception as exc:
        print(f"[metrics:error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
