from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "honest_eval_report_batches"
OUT = REPORT_DIR / "charts" / "concordai_dashboard_poster_3600x2520.png"
SECTIONS_DIR = REPORT_DIR / "charts" / "poster_sections"
W, H = 3600, 2520

COLORS = {
    "bg": "#f7f3ec",
    "panel": "#fffdf8",
    "ink": "#172027",
    "muted": "#657079",
    "grid": "#e9e0d3",
    "border": "#d8ccbb",
    "blue": "#2474a6",
    "amber": "#c77918",
    "teal": "#178b7f",
    "rose": "#c45a6d",
    "green": "#2f9b63",
    "bar_bg": "#eee8df",
}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = {
        "regular": ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"],
        "bold": ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"],
        "mono": ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"],
    }[name]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F = {
    "title": font("bold", 72),
    "section": font("bold", 48),
    "sub": font("regular", 28),
    "label": font("regular", 28),
    "label_b": font("bold", 28),
    "small": font("regular", 24),
    "small_b": font("bold", 24),
    "kpi": font("bold", 54),
    "card_title": font("bold", 40),
}


def load_summary() -> dict:
    return json.loads((REPORT_DIR / "honest_summary.json").read_text(encoding="utf-8"))


def load_hard_checks() -> dict[str, dict]:
    with (REPORT_DIR / "hard_check_rates.csv").open("r", encoding="utf-8", newline="") as f:
        return {row["check"]: row for row in csv.DictReader(f)}


def pct(v: float) -> str:
    return f"{round(float(v) * 100):.0f}%"


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill: str, outline: str | None = None, width: int = 2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fill: str, fnt: ImageFont.FreeTypeFont, anchor: str | None = None) -> None:
    draw.text(xy, value, fill=fill, font=fnt, anchor=anchor)


def wrap(draw: ImageDraw.ImageDraw, value: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def panel(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, title: str, subtitle: str = "") -> None:
    rounded(draw, (x, y, x + w, y + h), 44, COLORS["panel"], COLORS["border"], 3)
    text(draw, (x + 56, y + 44), title, COLORS["ink"], F["section"])
    if subtitle:
        text(draw, (x + 56, y + 100), subtitle, COLORS["muted"], F["sub"])


def legend(draw: ImageDraw.ImageDraw, x: int, y: int, items: list[tuple[str, str]]) -> None:
    width = sum(98 + len(name) * 14 for name, _ in items)
    rounded(draw, (x, y, x + width, y + 64), 32, "#ffffff", COLORS["border"], 2)
    cx = x + 30
    for name, color in items:
        rounded(draw, (cx, y + 23, cx + 32, y + 43), 8, color)
        text(draw, (cx + 46, y + 18), name, COLORS["ink"], F["small_b"])
        cx += 98 + len(name) * 14


def axes(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = y + h - int(tick * h)
        draw.line((x, yy, x + w, yy), fill=COLORS["grid"], width=2)
        text(draw, (x - 18, yy), pct(tick), COLORS["muted"], F["small"], "rm")
    draw.line((x, y, x, y + h), fill=COLORS["border"], width=2)
    draw.line((x, y + h, x + w, y + h), fill=COLORS["border"], width=2)


def y_for(value: float, y: int, h: int) -> int:
    return y + h - int(max(0, min(1, value)) * h)


def metric_points(values: list[float], x: int, y: int, w: int, h: int) -> list[tuple[int, int]]:
    step = w / max(1, len(values) - 1)
    return [(int(x + idx * step), y_for(value, y, h)) for idx, value in enumerate(values)]


def weighted(metric: dict, key: str) -> float:
    rows = metric.get("per_class", [])
    total = sum(float(row.get("support", 0)) for row in rows)
    if not total:
        return 0.0
    return sum(float(row.get(key, 0)) * float(row.get("support", 0)) for row in rows) / total


def kpi(draw: ImageDraw.ImageDraw, x: int, label: str, value: str, color: str, note: str) -> None:
    rounded(draw, (x, 108, x + 300, 108 + 144), 36, "#ffffff", COLORS["border"], 2)
    draw.ellipse((x + 38, 155, x + 62, 179), fill=color)
    text(draw, (x + 82, 146), label, COLORS["muted"], F["small_b"])
    text(draw, (x + 44, 194), value, COLORS["ink"], F["kpi"])
    text(draw, (x + 176, 207), note, COLORS["muted"], F["small"])


def save_crops(img: Image.Image) -> None:
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    crops = {
        # Individual sections.
        "01_header_kpis.png": (45, 35, 3555, 345),
        "02_performance_profile.png": (45, 325, 2185, 1235),
        "03_observed_task_scores.png": (2185, 325, 3555, 1235),
        "04_response_quality_checks.png": (45, 1245, 2185, 1935),
        "05_audit_integrity.png": (2185, 1245, 3555, 1935),
        "06_bottom_summary_cards.png": (45, 2010, 3555, 2395),
        # Two-per-image groups.
        "pair_01_performance_and_observed.png": (45, 325, 3555, 1235),
        "pair_02_quality_and_integrity.png": (45, 1245, 3555, 1935),
        "pair_03_all_summary_cards.png": (45, 2010, 3555, 2395),
    }
    for name, box in crops.items():
        img.crop(box).save(SECTIONS_DIR / name, format="PNG", optimize=True)


def draw_dashboard() -> None:
    summary = load_summary()
    checks = load_hard_checks()
    observed = summary["observed_metrics"]
    proxy = summary["proxy_metrics"]
    run = summary["run_summary"]
    flags = summary["suspicious_or_not_measured"]

    labels = ["Conflict", "Resolve", "Fallback", "Safety", "Empathy A", "Empathy B", "POV", "Overall"]
    accuracy = [
        observed["conflict_type"]["accuracy"],
        observed["resolvability"]["accuracy"],
        observed["should_use_fallback"]["accuracy"],
        proxy["output_safety_toxicity"]["accuracy"],
        float(checks["empathy_a"]["strict_pass_rate"]),
        float(checks["empathy_b"]["strict_pass_rate"]),
        float(checks["pov_correct"]["strict_pass_rate"]),
        run["hard_pass_rate"],
    ]
    macro_f1 = [
        observed["conflict_type"]["macro_f1"],
        observed["resolvability"]["macro_f1"],
        observed["should_use_fallback"]["macro_f1"],
        proxy["output_safety_toxicity"]["macro_f1"],
        float(checks["empathy_a"]["weighted_pass_rate"]),
        float(checks["empathy_b"]["weighted_pass_rate"]),
        float(checks["pov_correct"]["weighted_pass_rate"]),
        run["hard_pass_rate"],
    ]
    weighted_f1 = [
        observed["conflict_type"]["weighted_f1"],
        observed["resolvability"]["weighted_f1"],
        observed["should_use_fallback"]["weighted_f1"],
        proxy["output_safety_toxicity"]["weighted_f1"],
        float(checks["empathy_a"]["strict_pass_rate"]),
        float(checks["empathy_b"]["strict_pass_rate"]),
        float(checks["pov_correct"]["strict_pass_rate"]),
        run["hard_pass_rate"],
    ]

    img = Image.new("RGB", (W, H), COLORS["bg"])
    draw = ImageDraw.Draw(img)

    rounded(draw, (80, 70, 3520, 310), 52, COLORS["panel"], COLORS["border"], 3)
    text(draw, (130, 126), "ConcordAI Evaluation Dashboard", COLORS["ink"], F["title"])
    text(draw, (130, 208), "Leakage-aware ML report from saved backend evaluation batches.", COLORS["muted"], F["sub"])
    text(draw, (130, 250), "Perfect metrics are shown with review context, not treated as blind proof.", COLORS["muted"], F["sub"])
    kpi(draw, 2370, "Cases", str(run["total_cases"]), COLORS["blue"], "evaluated")
    kpi(draw, 2705, "Hard pass", pct(run["hard_pass_rate"]), COLORS["teal"], "strict")
    kpi(draw, 3040, "Fallbacks", str(run["fallbacks"]), COLORS["amber"], "used")

    panel(draw, 80, 360, 2070, 840, "Performance Profile", "Accuracy and F1 across observed/proxy signals")
    legend(draw, 1395, 430, [("Accuracy", COLORS["blue"]), ("Macro F1", COLORS["amber"]), ("Weighted F1", COLORS["green"])])
    x0, y0, cw, ch = 210, 590, 1760, 445
    axes(draw, x0, y0, cw, ch)
    for values, color in [(accuracy, COLORS["blue"]), (macro_f1, COLORS["amber"]), (weighted_f1, COLORS["green"])]:
        pts = metric_points(values, x0, y0, cw, ch)
        draw.line(pts, fill=color, width=8, joint="curve")
        for idx, (px, py) in enumerate(pts):
            draw.ellipse((px - 11, py - 11, px + 11, py + 11), fill=color, outline=COLORS["panel"], width=4)
            if values[idx] < 0.94:
                text(draw, (px, py - 28), pct(values[idx]), color, F["small_b"], "mm")
    for idx, label in enumerate(labels):
        px = int(x0 + idx * (cw / max(1, len(labels) - 1)))
        text(draw, (px, y0 + ch + 58), label, COLORS["ink"], F["small"], "mm")

    panel(draw, 2220, 360, 1300, 840, "Observed Task Scores", "Weighted precision, recall, and F1 from saved result fields")
    legend(draw, 2400, 500, [("Precision", COLORS["blue"]), ("Recall", COLORS["amber"]), ("F1", COLORS["green"])])
    rx, ry, rw, rh = 2385, 660, 880, 370
    axes(draw, rx, ry, rw, rh)
    tasks = [("Conflict", observed["conflict_type"]), ("Resolve", observed["resolvability"]), ("Fallback", observed["should_use_fallback"])]
    group_w = rw / len(tasks)
    bar_w = 66
    for idx, (task, metric) in enumerate(tasks):
        center = int(rx + idx * group_w + group_w / 2)
        values = [weighted(metric, "precision"), weighted(metric, "recall"), metric["weighted_f1"]]
        for bidx, value in enumerate(values):
            color = [COLORS["blue"], COLORS["amber"], COLORS["green"]][bidx]
            bh = int(value * rh)
            bx = int(center + (bidx - 1) * (bar_w + 22) - bar_w / 2)
            by = ry + rh - bh
            rounded(draw, (bx, by, bx + bar_w, ry + rh), 12, color)
            if value < 0.98:
                text(draw, (bx + bar_w // 2, by - 28), pct(value), color, F["small_b"], "mm")
        text(draw, (center, ry + rh + 58), task, COLORS["ink"], F["label"], "mm")

    panel(draw, 80, 1280, 2070, 620, "Response Quality Checks", "Strict pass rates from the backend evaluator")
    ordered = [item for item in checks.values() if item["check"] in ["conflict_type", "resolvability", "distinct_responses", "specific_a", "specific_b", "empathy_a", "empathy_b", "pov_correct", "safety"]]
    for idx, item in enumerate(ordered):
        col = 0 if idx < 5 else 1
        row = idx if idx < 5 else idx - 5
        x = 150 + col * 1000
        y = 1430 + row * 92
        rate = float(item["strict_pass_rate"])
        failed = int(item["failed"])
        color = COLORS["teal"] if rate >= 0.98 else (COLORS["amber"] if rate >= 0.9 else COLORS["rose"])
        text(draw, (x, y + 28), item["check"].replace("_", " ").title(), COLORS["ink"], F["label_b"])
        rounded(draw, (x + 330, y, x + 860, y + 30), 15, COLORS["bar_bg"])
        rounded(draw, (x + 330, y, x + 330 + int(530 * rate), y + 30), 15, color)
        text(draw, (x + 895, y + 28), pct(rate), COLORS["ink"], F["label_b"])
        if failed:
            text(draw, (x + 895, y + 62), f"{failed} fail", COLORS["rose"], F["small"])

    panel(draw, 2220, 1280, 1300, 620, "Audit Integrity", "Measured signals versus annotation-only labels")
    card_data = [
        ("Observed", len(["conflict_type", "resolvability", "should_use_fallback"]), COLORS["teal"], "direct fields"),
        ("Proxy", 1, COLORS["blue"], "indirect"),
        ("Not measured", sum(1 for f in flags if f["severity"] == "high"), COLORS["rose"], "annotation-only"),
        ("Review", sum(1 for f in flags if f["severity"] == "review"), COLORS["amber"], "perfect-score"),
    ]
    for idx, (name, count, color, note) in enumerate(card_data):
        x = 2300 + (idx % 2) * 565
        y = 1450 + (idx // 2) * 190
        rounded(draw, (x, y, x + 480, y + 128), 28, "#ffffff", COLORS["border"], 2)
        draw.ellipse((x + 40, y + 42, x + 72, y + 74), fill=color)
        text(draw, (x + 95, y + 42), name, COLORS["ink"], F["label_b"])
        text(draw, (x + 410, y + 42), str(count), color, F["kpi"], "la")
        text(draw, (x + 95, y + 84), note, COLORS["muted"], F["small"])

    bottom = [
        ("Core Labels", COLORS["teal"], [("Conflict", pct(observed["conflict_type"]["accuracy"])), ("Resolvability", pct(observed["resolvability"]["accuracy"]))], "Saved outputs match expected labels on this batch. Perfect scores remain review-marked until fresh independent runs confirm them."),
        ("Response Behavior", COLORS["blue"], [("Hard pass", pct(run["hard_pass_rate"])), ("Empathy A/B", f'{pct(float(checks["empathy_a"]["strict_pass_rate"]))} / {pct(float(checks["empathy_b"]["strict_pass_rate"]))}')], "Visible product quality is strong. Remaining misses concentrate in response-level checks, not core labels."),
        ("Reliability", COLORS["amber"], [("Fallbacks", f"{run['fallbacks']} / {run['total_cases']}"), ("Errors", str(run["errors"]))], "Fallback use is visible and controlled. The audit separates operational fallback behavior from core reasoning accuracy."),
    ]
    for idx, (title, color, rows, desc) in enumerate(bottom):
        x = 80 + idx * 1135
        y = 2045
        rounded(draw, (x, y, x + 1075, y + 305), 36, COLORS["panel"], COLORS["border"], 3)
        rounded(draw, (x, y, x + 14, y + 305), 7, color)
        text(draw, (x + 50, y + 62), title, COLORS["ink"], F["card_title"])
        for ridx, (key, value) in enumerate(rows):
            yy = y + 125 + ridx * 48
            text(draw, (x + 50, yy), key, COLORS["muted"], F["label"])
            text(draw, (x + 320, yy), value, COLORS["ink"], F["label_b"])
        for line_idx, line in enumerate(wrap(draw, desc, F["small"], 920)):
            text(draw, (x + 50, y + 230 + line_idx * 34), line, COLORS["muted"], F["small"])

    text(draw, (W // 2, 2445), "Note: Strategy, intensity, repeat-pattern, and safety-sensitive annotations are reported as coverage unless independent actual predictions are saved.", COLORS["muted"], F["small"], "mm")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, format="PNG", optimize=True)
    save_crops(img)
    print(OUT)
    print(SECTIONS_DIR)


if __name__ == "__main__":
    draw_dashboard()
