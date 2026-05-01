"""Layered safety check using Detoxify and toxic-bert."""

from __future__ import annotations

import asyncio
from typing import Any

from detoxify import Detoxify
from transformers import pipeline

from config import DETOXIFY_THRESHOLD, TOXIC_BERT_MODEL, TOXIC_BERT_THRESHOLD
from models.schemas import SafetyResult


_detoxify_model: Any | None = None
_toxic_bert_pipe: Any | None = None
_detoxify_error: str | None = None
_toxic_bert_error: str | None = None


def _get_detoxify():
    global _detoxify_model, _detoxify_error
    if _detoxify_model is None:
        try:
            print("[safety] Loading Detoxify...")
            _detoxify_model = Detoxify("original")
        except Exception as exc:  # pragma: no cover - runtime dependency behavior
            _detoxify_error = str(exc)
            raise
    return _detoxify_model


def _get_toxic_bert():
    global _toxic_bert_pipe, _toxic_bert_error
    if _toxic_bert_pipe is None:
        try:
            print("[safety] Loading toxic-bert...")
            _toxic_bert_pipe = pipeline("text-classification", model=TOXIC_BERT_MODEL, device=-1)
        except Exception as exc:  # pragma: no cover - runtime dependency behavior
            _toxic_bert_error = str(exc)
            raise
    return _toxic_bert_pipe


def _detoxify_score(text: str) -> tuple[float, list[str], str | None]:
    try:
        scores = _get_detoxify().predict(text)
        max_score = float(max(scores.values())) if scores else 0.0
        flagged = [name for name, score in scores.items() if float(score) > DETOXIFY_THRESHOLD]
        return round(max_score, 4), flagged, None
    except Exception as exc:  # pragma: no cover - runtime dependency behavior
        return 0.0, [], f"Detoxify skipped: {exc}"


def _toxic_bert_score(text: str) -> tuple[float, str | None]:
    try:
        result = _get_toxic_bert()(text[:512])[0]
        if result["label"].upper() == "TOXIC":
            return round(float(result["score"]), 4), None
        return round(1.0 - float(result["score"]), 4), None
    except Exception as exc:  # pragma: no cover - runtime dependency behavior
        return 0.0, f"toxic-bert skipped: {exc}"


def _check_sync(response_a: str, response_b: str, *, use_toxic_bert: bool = True) -> SafetyResult:
    combined = f"{response_a} {response_b}".strip()
    detox_score, flagged, detox_error = _detoxify_score(combined)
    if use_toxic_bert:
        bert_score, bert_error = _toxic_bert_score(combined)
    else:
        bert_score, bert_error = 0.0, None

    approved = detox_score <= DETOXIFY_THRESHOLD and bert_score <= TOXIC_BERT_THRESHOLD
    feedback_parts: list[str] = []
    if detox_score > DETOXIFY_THRESHOLD:
        feedback_parts.append(
            f"Detoxify flagged {flagged or ['toxicity']} at {detox_score:.2f}; rewrite in neutral language."
        )
    if bert_score > TOXIC_BERT_THRESHOLD:
        feedback_parts.append(
            f"toxic-bert score {bert_score:.2f}; remove hostile or identity-attacking wording."
        )
    if detox_error:
        feedback_parts.append(detox_error)
    if bert_error:
        feedback_parts.append(bert_error)

    return SafetyResult(
        approved=approved,
        detoxify_score=detox_score,
        toxic_bert_score=bert_score,
        flagged_categories=flagged,
        feedback=" | ".join(feedback_parts) if feedback_parts else None,
    )


async def check(response_a: str, response_b: str, *, use_toxic_bert: bool = True) -> SafetyResult:
    return await asyncio.to_thread(
        _check_sync,
        response_a,
        response_b,
        use_toxic_bert=use_toxic_bert,
    )


async def warmup(*, use_toxic_bert: bool = True) -> dict[str, bool]:
    tasks = [asyncio.to_thread(_get_detoxify)]
    if use_toxic_bert:
        tasks.append(asyncio.to_thread(_get_toxic_bert))
    await asyncio.gather(*tasks, return_exceptions=True)
    return {
        "detoxify_loaded": _detoxify_model is not None,
        "toxic_bert_loaded": _toxic_bert_pipe is not None,
    }


def status() -> dict[str, Any]:
    return {
        "detoxify_loaded": _detoxify_model is not None,
        "toxic_bert_loaded": _toxic_bert_pipe is not None,
        "detoxify_error": _detoxify_error,
        "toxic_bert_error": _toxic_bert_error,
    }
