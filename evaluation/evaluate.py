"""Evaluate classifier signals over the 96-scenario seed dataset."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from agents import safety_agent, validator
from config import SCENARIOS_PATH
from models.schemas import SafetyResult


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _labels(scenario: dict) -> dict:
    labels = scenario.get("labels", {})
    return {
        "stance_a": labels.get("stance_a", "attack"),
        "stance_b": labels.get("stance_b", "attack"),
        "sentiment_a": labels.get("sentiment_a", "negative"),
        "sentiment_b": labels.get("sentiment_b", "negative"),
        "safety": labels.get("safety", "safe"),
    }


def _row(name: str, truth: list[str], pred: list[str]) -> str:
    return (
        f"{name:<28} "
        f"Acc={accuracy_score(truth, pred):.2f} "
        f"Prec={precision_score(truth, pred, average='weighted', zero_division=0):.2f} "
        f"Rec={recall_score(truth, pred, average='weighted', zero_division=0):.2f} "
        f"F1={f1_score(truth, pred, average='weighted', zero_division=0):.2f}"
    )


async def run_evaluation() -> None:
    scenarios = load_scenarios()
    gt = {key: [] for key in ["stance_a", "stance_b", "sentiment_a", "sentiment_b", "safety"]}
    pred = {key: [] for key in gt}

    print(f"[eval] Evaluating {len(scenarios)} scenarios...")
    neutral_safety = SafetyResult(approved=True)
    for index, scenario in enumerate(scenarios, 1):
        labels = _labels(scenario)
        for key, value in labels.items():
            gt[key].append(value)

        val = await validator.validate(
            text_a=scenario["text_a"],
            text_b=scenario["text_b"],
            response_a="",
            response_b="",
            safety_result=neutral_safety,
        )
        raw_safety = await safety_agent.check(scenario["text_a"], scenario["text_b"])

        pred["stance_a"].append(val.stance.user_a_stance)
        pred["stance_b"].append(val.stance.user_b_stance)
        pred["sentiment_a"].append(val.emotion.user_a_sentiment)
        pred["sentiment_b"].append(val.emotion.user_b_sentiment)
        pred["safety"].append("safe" if raw_safety.approved else "unsafe")
        print(f"[eval] {index:02d}/{len(scenarios)} done")

    print("\n" + "=" * 72)
    print(f"{'EVALUATION RESULTS':^72}")
    print("=" * 72)
    print(_row("Stance - User A", gt["stance_a"], pred["stance_a"]))
    print(_row("Stance - User B", gt["stance_b"], pred["stance_b"]))
    print(_row("Sentiment - User A", gt["sentiment_a"], pred["sentiment_a"]))
    print(_row("Sentiment - User B", gt["sentiment_b"], pred["sentiment_b"]))
    print(_row("Safety", gt["safety"], pred["safety"]))
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(run_evaluation())

