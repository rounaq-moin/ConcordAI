"""Perspective, emotion, and communicative-intent validation agents."""

from __future__ import annotations

import asyncio
from threading import Lock
from typing import Any

from transformers import pipeline

from config import EMOTION_MODEL, PERSPECTIVE_MODEL, POSITIVE_EMOTIONS
from models.schemas import EmotionResult, IntentResult, SafetyResult, StanceResult, ValidationOutput


_perspective_pipe: Any | None = None
_emotion_pipe: Any | None = None
_perspective_error: str | None = None
_emotion_error: str | None = None
_perspective_lock = Lock()
_emotion_lock = Lock()

INTENT_CONFIDENCE_THRESHOLD = 0.5
INTENT_LABELS = (
    "seek_acknowledgment",
    "set_boundary",
    "request_change",
    "explain_context",
    "defend_decision",
    "express_hurt",
    "refuse_request",
    "repair_relationship",
    "challenge_claim",
    "unknown",
)
INTENT_LABEL_TEXT = {
    "seek_acknowledgment": "seek acknowledgment or recognition",
    "set_boundary": "set a boundary or limit",
    "request_change": "request a concrete change",
    "explain_context": "explain context or clarify background",
    "defend_decision": "defend a decision or choice",
    "express_hurt": "express hurt or emotional impact",
    "refuse_request": "refuse a request or demand",
    "repair_relationship": "repair the relationship or apologize",
    "challenge_claim": "challenge a claim or dispute facts",
    "unknown": "unclear communicative intent",
}
_INTENT_TEXT_TO_LABEL = {value: key for key, value in INTENT_LABEL_TEXT.items()}


def _get_perspective():
    global _perspective_pipe, _perspective_error
    if _perspective_pipe is None:
        with _perspective_lock:
            if _perspective_pipe is None:
                try:
                    print("[validator] Loading perspective model...")
                    _perspective_pipe = pipeline("zero-shot-classification", model=PERSPECTIVE_MODEL, device=-1)
                except Exception as exc:  # pragma: no cover - runtime dependency behavior
                    _perspective_error = str(exc)
                    raise
    return _perspective_pipe


def _get_emotion():
    global _emotion_pipe, _emotion_error
    if _emotion_pipe is None:
        with _emotion_lock:
            if _emotion_pipe is None:
                try:
                    print("[validator] Loading emotion model...")
                    _emotion_pipe = pipeline("text-classification", model=EMOTION_MODEL, top_k=None, device=-1)
                except Exception as exc:  # pragma: no cover - runtime dependency behavior
                    _emotion_error = str(exc)
                    raise
    return _emotion_pipe


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _neutral_stance(feedback: str | None = None) -> StanceResult:
    return StanceResult(
        user_a_stance="neutral",
        user_b_stance="neutral",
        response_consistent=True,
        feedback=feedback,
    )


def _neutral_emotion(feedback: str | None = None) -> EmotionResult:
    return EmotionResult(
        user_a_emotion="neutral",
        user_b_emotion="neutral",
        user_a_sentiment="neutral",
        user_b_sentiment="neutral",
        response_empathetic=True,
        feedback=feedback,
    )


def _neutral_intent(feedback: str | None = None) -> IntentResult:
    return IntentResult(
        user_a_intent="unknown",
        user_b_intent="unknown",
        confidence_a=0.0,
        confidence_b=0.0,
        feedback=feedback,
    )


def infer_intent(text: str) -> tuple[str, float]:
    """Infer observable communicative action, not hidden motive."""
    normalized = _norm(text)
    if not normalized:
        return "unknown", 0.0

    rules: tuple[tuple[str, float, tuple[str, ...]], ...] = (
        ("refuse_request", 0.9, ("i will not", "i won't", "i refuse", "will not sign", "won't sign", "i can't do that")),
        ("repair_relationship", 0.84, ("i'm sorry", "i am sorry", "i apologize", "make it right", "repair this", "i regret")),
        ("request_change", 0.78, ("can you", "could you", "please", "i need you to", "would you", "change how")),
        ("set_boundary", 0.78, ("boundary", "not okay", "crossed a line", "without my permission", "respect my", "stop doing")),
        ("seek_acknowledgment", 0.76, ("acknowledge", "recognize", "credit", "appreciate", "mention", "invisible", "undervalued")),
        ("defend_decision", 0.74, ("i was trying", "i had to", "i chose", "i made the call", "deadline", "because we needed")),
        ("explain_context", 0.7, ("i thought", "i assumed", "i meant", "i didn't mean", "i did not mean", "i didn't realize", "i did not realize", "from my perspective")),
        ("challenge_claim", 0.7, ("not true", "that's false", "that is false", "evidence", "prove", "unsupported", "no evidence", "wrong about")),
        ("express_hurt", 0.68, ("i feel", "i felt", "hurt", "upset", "frustrated", "angry", "betrayed", "sad")),
    )
    for intent, confidence, cues in rules:
        if any(cue in normalized for cue in cues):
            return intent, confidence
    return "unknown", 0.0


def infer_intents(text_a: str, text_b: str, feedback: str | None = None) -> IntentResult:
    intent_a, confidence_a = infer_intent(text_a)
    intent_b, confidence_b = infer_intent(text_b)
    return IntentResult(
        user_a_intent=intent_a,  # type: ignore[arg-type]
        user_b_intent=intent_b,  # type: ignore[arg-type]
        confidence_a=confidence_a,
        confidence_b=confidence_b,
        feedback=feedback,
    )


def _perspective_agent(text_a: str, text_b: str, response_a: str, response_b: str) -> StanceResult:
    try:
        pipe = _get_perspective()
        labels = ["support", "attack", "neutral"]
        result_a = pipe(text_a, candidate_labels=labels)
        result_b = pipe(text_b, candidate_labels=labels)
        stance_a = result_a["labels"][0]
        stance_b = result_b["labels"][0]

        response_text = f"{response_a} {response_b}".strip()
        if not response_text:
            return StanceResult(
                user_a_stance=stance_a,
                user_b_stance=stance_b,
                response_consistent=True,
                confidence_a=round(float(result_a["scores"][0]), 3),
                confidence_b=round(float(result_b["scores"][0]), 3),
            )

        check = pipe(
            response_text,
            candidate_labels=["addresses both perspectives", "one-sided"],
        )
        consistent = check["labels"][0] == "addresses both perspectives"
        score = float(check["scores"][0])
        feedback = None
        if not consistent and score > 0.7:
            feedback = (
                f"Response appears one-sided with confidence {score:.2f}. "
                f"Acknowledge User A ({stance_a}) and User B ({stance_b}) explicitly."
            )
        else:
            consistent = True

        return StanceResult(
            user_a_stance=stance_a,
            user_b_stance=stance_b,
            response_consistent=consistent,
            confidence_a=round(float(result_a["scores"][0]), 3),
            confidence_b=round(float(result_b["scores"][0]), 3),
            feedback=feedback,
        )
    except Exception as exc:  # pragma: no cover - runtime dependency behavior
        return _neutral_stance(f"Perspective model skipped: {exc}")


def _emotion_agent(text_a: str, text_b: str, response_a: str, response_b: str) -> EmotionResult:
    try:
        pipe = _get_emotion()

        def top_emotion(text: str) -> tuple[str, str, float]:
            result = pipe(text)
            scores = sorted(result[0], key=lambda item: item["score"], reverse=True)
            emotion = scores[0]["label"]
            sentiment = "positive" if emotion in POSITIVE_EMOTIONS else "negative"
            return emotion, sentiment, round(float(scores[0]["score"]), 3)

        emotion_a, sentiment_a, confidence_a = top_emotion(text_a)
        emotion_b, sentiment_b, confidence_b = top_emotion(text_b)

        response_text = f"{response_a} {response_b}".lower().strip()
        if not response_text:
            return EmotionResult(
                user_a_emotion=emotion_a,
                user_b_emotion=emotion_b,
                user_a_sentiment=sentiment_a,
                user_b_sentiment=sentiment_b,
                response_empathetic=True,
                confidence_a=confidence_a,
                confidence_b=confidence_b,
            )

        empathy_markers = [
            "understand",
            "feel",
            "hear you",
            "makes sense",
            "valid",
            "acknowledge",
            "appreciate",
            "recognize",
            "see why",
        ]
        count = sum(1 for marker in empathy_markers if marker in response_text)
        empathetic = count >= 2
        feedback = None
        if not empathetic:
            feedback = (
                f"Insufficient empathetic language ({count} markers found). "
                f"User A appears {emotion_a}; User B appears {emotion_b}. "
                "Explicitly acknowledge both emotional states."
            )

        return EmotionResult(
            user_a_emotion=emotion_a,
            user_b_emotion=emotion_b,
            user_a_sentiment=sentiment_a,
            user_b_sentiment=sentiment_b,
            response_empathetic=empathetic,
            confidence_a=confidence_a,
            confidence_b=confidence_b,
            feedback=feedback,
        )
    except Exception as exc:  # pragma: no cover - runtime dependency behavior
        return _neutral_emotion(f"Emotion model skipped: {exc}")


def _intent_agent(text_a: str, text_b: str) -> IntentResult:
    try:
        pipe = _get_perspective()
        labels = list(INTENT_LABEL_TEXT.values())

        def top_intent(text: str) -> tuple[str, float]:
            result = pipe(text, candidate_labels=labels)
            label = result["labels"][0]
            score = round(float(result["scores"][0]), 3)
            intent = _INTENT_TEXT_TO_LABEL.get(label, "unknown")
            if score < INTENT_CONFIDENCE_THRESHOLD:
                return "unknown", score
            return intent, score

        intent_a, confidence_a = top_intent(text_a)
        intent_b, confidence_b = top_intent(text_b)
        return IntentResult(
            user_a_intent=intent_a,  # type: ignore[arg-type]
            user_b_intent=intent_b,  # type: ignore[arg-type]
            confidence_a=confidence_a,
            confidence_b=confidence_b,
        )
    except Exception as exc:  # pragma: no cover - runtime dependency behavior
        return infer_intents(text_a, text_b, feedback=f"Intent model skipped: {exc}")


def _aggregate(stance: StanceResult, emotion: EmotionResult, safety: SafetyResult) -> tuple[bool, str | None]:
    issues: list[str] = []
    if stance.feedback:
        issues.append(f"PERSPECTIVE: {stance.feedback}")
    if emotion.feedback:
        issues.append(f"EMOTION: {emotion.feedback}")
    if not safety.approved and safety.feedback:
        issues.append(f"SAFETY: {safety.feedback}")

    hard_fail = not safety.approved or (
        not stance.response_consistent and not emotion.response_empathetic
    )
    return not hard_fail, "\n".join(issues) if issues else None


async def validate(
    *,
    text_a: str,
    text_b: str,
    response_a: str,
    response_b: str,
    safety_result: SafetyResult,
) -> ValidationOutput:
    stance_task = asyncio.to_thread(_perspective_agent, text_a, text_b, response_a, response_b)
    emotion_task = asyncio.to_thread(_emotion_agent, text_a, text_b, response_a, response_b)
    intent_task = asyncio.to_thread(_intent_agent, text_a, text_b)
    stance, emotion, intent = await asyncio.gather(stance_task, emotion_task, intent_task)
    passed, feedback = _aggregate(stance, emotion, safety_result)
    return ValidationOutput(
        stance=stance,
        emotion=emotion,
        intent=intent,
        safety=safety_result,
        overall_passed=passed,
        aggregated_feedback=feedback,
    )


async def warmup() -> dict[str, bool]:
    await asyncio.gather(
        asyncio.to_thread(_get_perspective),
        asyncio.to_thread(_get_emotion),
        return_exceptions=True,
    )
    return {
        "perspective_loaded": _perspective_pipe is not None,
        "emotion_loaded": _emotion_pipe is not None,
        "intent_loaded": _perspective_pipe is not None,
    }


def status() -> dict[str, Any]:
    return {
        "perspective_loaded": _perspective_pipe is not None,
        "emotion_loaded": _emotion_pipe is not None,
        "intent_loaded": _perspective_pipe is not None,
        "perspective_error": _perspective_error,
        "emotion_error": _emotion_error,
        "intent_error": _perspective_error,
    }
