"""Structured response critic agent."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from models.schemas import ReflectionOutput
from utils.llm import call_groq_json


SYSTEM_PROMPT = """You are a strict quality auditor for an AI conflict mediation system.

Evaluate the generated responses against five criteria. Be strict.
A response that is generic, identical to the other response, or fails to address the specific conflict must be marked false.

Criteria:
1. is_fair: both users receive equal attention and validation depth
2. is_empathetic: each response explicitly names and acknowledges that user's specific emotion or concern
3. is_unbiased: no side-taking, no language that implicitly favors one user
4. is_specific: references the actual conflict content, not generic platitudes
5. is_non_escalatory: calm, constructive tone throughout

Critical failure conditions:
- response_a and response_b are identical or near-identical
- either response could apply to almost any conflict
- either response fails to acknowledge that user's specific words or situation

approved must be false if any single criterion is false.
Return only valid JSON with no extra text."""


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _near_identical(response_a: str, response_b: str) -> bool:
    norm_a = _normalized(response_a)
    norm_b = _normalized(response_b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    return SequenceMatcher(None, norm_a, norm_b).ratio() >= 0.86


def basic_response_check(response_a: str, response_b: str) -> ReflectionOutput | None:
    """Cheap deterministic checks that should run even when LLM critique is skipped."""
    if _near_identical(response_a, response_b):
        return ReflectionOutput(
            approved=False,
            is_fair=False,
            is_empathetic=False,
            is_unbiased=True,
            is_specific=False,
            is_non_escalatory=True,
            reason="response_a and response_b are identical or near-identical.",
            suggestion=(
                "Regenerate two materially different responses. Address User A's exact concern in response_a "
                "and User B's exact concern in response_b without reusing the same opening or sentences."
            ),
        )
    return None


def _build_prompt(
    *,
    text_a: str,
    text_b: str,
    response_a: str,
    response_b: str,
    conflict_type: str,
    resolvability: str,
) -> str:
    return f"""Evaluate these mediation responses strictly.

ORIGINAL CONFLICT:
User A: "{text_a}"
User B: "{text_b}"
Conflict type: {conflict_type}
Resolvability: {resolvability}

GENERATED RESPONSES:
For User A: "{response_a}"
For User B: "{response_b}"

Evaluate each criterion carefully. Set approved=false if any criterion fails.
Use actual JSON booleans and null values. Do not copy default values.

Return ONLY a valid JSON object with these exact keys:
- is_fair
- is_empathetic
- is_unbiased
- is_specific
- is_non_escalatory
- approved
- reason
- suggestion"""


async def critique(
    *,
    text_a: str,
    text_b: str,
    response_a: str,
    response_b: str,
    conflict_type: str,
    resolvability: str,
) -> ReflectionOutput:
    basic_failure = basic_response_check(response_a, response_b)
    if basic_failure:
        return basic_failure

    data = await call_groq_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_prompt(
                    text_a=text_a,
                    text_b=text_b,
                    response_a=response_a,
                    response_b=response_b,
                    conflict_type=conflict_type,
                    resolvability=resolvability,
                ),
            },
        ],
        source="critic_agent",
        max_tokens=450,
        temperature=0.05,
    )
    return ReflectionOutput(
        approved=bool(data.get("approved", False)),
        is_fair=bool(data.get("is_fair", False)),
        is_empathetic=bool(data.get("is_empathetic", False)),
        is_unbiased=bool(data.get("is_unbiased", False)),
        is_specific=bool(data.get("is_specific", False)),
        is_non_escalatory=bool(data.get("is_non_escalatory", False)),
        reason=data.get("reason"),
        suggestion=data.get("suggestion"),
    )
