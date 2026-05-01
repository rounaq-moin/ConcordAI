"""Quality and confidence scoring helpers."""

from __future__ import annotations

from config import MAX_RETRIES


def fairness_score(response_a: str, response_b: str) -> float:
    """0.0 means perfectly balanced; 1.0 means completely one-sided."""
    len_a = len((response_a or "").split())
    len_b = len((response_b or "").split())
    total = len_a + len_b
    return round(abs(len_a - len_b) / total, 3) if total else 0.0


def confidence_score(
    *,
    critic_approved: bool,
    retries: int,
    safety_score: float,
    fallback_used: bool,
    critic_skipped: bool = False,
    production_mode: bool = False,
) -> float:
    retry_component = max(0.0, 1.0 - (retries / max(MAX_RETRIES, 1)))
    safety_component = max(0.0, min(1.0, 1.0 - safety_score))
    critic_component = 0.0
    if critic_approved:
        critic_component = 0.28 if critic_skipped else 0.4
    confidence = (
        critic_component
        + retry_component * 0.2
        + safety_component * 0.2
        + (0.0 if fallback_used else 1.0) * 0.2
    )
    # Avoid presenting probabilistic-looking certainty in a subjective mediation task.
    cap = 0.72 if fallback_used else (0.86 if production_mode else 0.92)
    return round(max(0.0, min(cap, confidence)), 3)
