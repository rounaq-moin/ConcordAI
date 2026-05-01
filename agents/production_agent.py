"""Single-call production mediation agent with lightweight verification.

fast_production mode is optimized for arbitrary user input:
- deterministic classification when rules are confident
- "unknown" internal classification when rules are not confident
- one normal LLM call that returns reasoning state + responses
- deterministic verification checks
- targeted repair call only when checks fail
"""

from __future__ import annotations

import json
import re
import hashlib
from difflib import SequenceMatcher
from typing import Any

from agents.reasoning_agent import classify_conflict_type, classify_resolvability
from config import GROQ_MAX_TOKENS, RAG_FINAL_K
from models.schemas import LLMCoreOutput, Reasoning, RetrievedCase
from utils.llm import call_groq_production
from utils.scoring import fairness_score


SYSTEM_PROMPT = """You are a production conflict mediation engine.

Your job is to behave like a careful mediator, not a generic chatbot:
- build a compact internal reasoning state before answering
- separate facts, assumptions, uncertainties, and goals
- use the selected strategy as the decision guide
- give each user a distinct, balanced, practical response
- do not take sides or force agreement

Return only valid JSON."""

VALID_CONFLICT_TYPES = {"emotional", "logical", "misunderstanding", "value"}
VALID_RESOLVABILITY = {"resolvable", "partially_resolvable", "non_resolvable"}
VALID_STATUS = {"Escalating", "Stable", "Improving", "Resolved"}
EMPATHY_MARKERS = (
    "understand", "makes sense", "valid", "hear", "recognize",
    "acknowledge", "sounds", "feel", "frustrating", "hurt",
)


def _content_words(text: str) -> set[str]:
    stop = {
        "the", "and", "but", "that", "this", "with", "have", "you", "your",
        "for", "are", "was", "were", "not", "from", "they", "them", "our",
        "about", "because", "just", "into", "what", "when", "where", "how",
        "why", "can", "could", "would", "should", "feel", "felt", "like",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9']+", text.lower())
        if len(token) > 3 and token not in stop
    }


def _similarity(a: str, b: str) -> float:
    norm_a = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", a.lower())).strip()
    norm_b = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", b.lower())).strip()
    if not norm_a or not norm_b:
        return 0.0
    return SequenceMatcher(None, norm_a, norm_b).ratio()


def _classification_decision(text_a: str, text_b: str) -> dict[str, Any]:
    conflict_type = classify_conflict_type(text_a, text_b)
    if conflict_type:
        return {
            "conflict_type": conflict_type,
            "system_fixed": True,
            "classification_confidence": 0.9,
            "instruction": "Use this conflict_type exactly.",
        }
    return {
        "conflict_type": "unknown",
        "system_fixed": False,
        "classification_confidence": 0.45,
        "instruction": (
            "Infer conflict_type carefully from the user messages. Choose exactly one of: "
            "emotional, logical, misunderstanding, value."
        ),
    }


def _select_strategy(cases: list[RetrievedCase], conflict_type: str) -> dict[str, Any]:
    if not cases:
        return {
            "selected_strategy": "Validate both sides, identify the core need, and suggest one concrete next step.",
            "strategy_source": "rule_default",
            "strategy_confidence": 0.45,
            "supporting_cases": [],
        }

    aligned = [case for case in cases if case.conflict_type == conflict_type]
    selected = (aligned or cases)[0]
    strategy = selected.resolution_strategy or "Validate both perspectives and suggest a concrete repair step."
    return {
        "selected_strategy": strategy,
        "strategy_source": "rag_conflict_aligned" if aligned else "rag_nearest_fallback",
        "strategy_confidence": round(selected.relevance_score, 3),
        "supporting_cases": [
            {
                "conflict_type": selected.conflict_type,
                "relevance_score": selected.relevance_score,
                "strategy": strategy,
            }
        ],
    }


def _build_prompt(
    *,
    text_a: str,
    text_b: str,
    classification: dict[str, Any],
    resolvability: str,
    strategy: dict[str, Any],
    prior_context: str | None,
    intent_a: str = "unknown",
    intent_b: str = "unknown",
) -> str:
    prior = f"\nPRIOR CONTEXT:\n{prior_context}\n" if prior_context else ""
    return f"""Mediate this conflict for arbitrary real user input.
{prior}
USER A:
"{text_a}"

USER B:
"{text_b}"

DETECTED COMMUNICATIVE INTENT:
- User A: {intent_a}
- User B: {intent_b}
- Intent is the observable action each user appears to be taking with their statement, not hidden motive. Use it only as supporting context; do not let it override conflict_type or resolvability.

SYSTEM CLASSIFICATION:
- conflict_type: {classification["conflict_type"]}
- system_fixed: {classification["system_fixed"]}
- confidence: {classification["classification_confidence"]}
- instruction: {classification["instruction"]}
- resolvability: {resolvability}

SELECTED STRATEGY:
- source: {strategy["strategy_source"]}
- confidence: {strategy["strategy_confidence"]}
- strategy: {strategy["selected_strategy"]}
- supporting_cases: {strategy["supporting_cases"]}

CONFLICT TYPE DEFINITIONS:
- emotional: the core injury is a feeling: neglect, invisibility, betrayal, humiliation, abandonment, resentment. The dispute is the emotional impact.
- logical: the disagreement is about facts, evidence, data, process, deadlines, tradeoffs, risk, or decision quality. Emotions may be present, but the dispute is not about the emotion itself.
- misunderstanding: both users are working from different interpretations of the same event, message, timing, ownership, or intent. The facts may overlap; the meaning differs.
- value: the conflict is about identity, faith, moral belief, privacy boundaries, life priorities, family expectations, or non-negotiable principles. Full agreement may not be possible.

RESOLVABILITY INSTRUCTIONS:
- resolvable: both responses should end with one concrete, actionable next step toward agreement.
- partially_resolvable: both responses should identify one workable compromise or boundary without promising full resolution.
- non_resolvable: do not suggest agreement or compromise. Focus on what each user needs respected and what coexistence or boundaries look like. Never imply the conflict can simply be solved.

STRATEGY INTERPRETATION:
- Treat the selected strategy as a step-by-step execution plan, not a topic summary.
- The first sentence of each response must implement the first relevant action in the strategy.
- If the strategy says "validate feelings first", response_a must open by naming the specific feeling or impact from User A's words.
- If the strategy says "acknowledge intent", response_b must open by naming what User B was trying to do or protect.
- Do not summarize the strategy. Execute it sentence by sentence.

REASONING REQUIREMENTS:
- Represent both users' views before writing responses.
- List assumptions and uncertainties instead of pretending certainty.
- If conflict_type is unknown, infer one valid conflict_type and keep uncertainty higher.
- Use the selected strategy as the mediation plan.

RESPONSE REQUIREMENTS:
- response_a addresses User A as "you"; it validates A's specific impact/need, acknowledges B's constraint or intent, and suggests one concrete request.
- response_b addresses User B as "you"; it validates B's specific intent/constraint/concern, acknowledges impact on A, and suggests one concrete repair step.
- response_a must include at least one concrete concept from User A's message, naturally paraphrased.
- response_b must include at least one concrete concept from User B's message, naturally paraphrased.
- Each response must include clear empathy, using natural language such as "that sounds frustrating", "it makes sense", or "you are trying to protect..."
- ROLE LOCK:
  * In response_a, every "you/your" means User A only. Never describe User B's actions, motives, constraints, or words as "you". Refer to User B as "the other person" or "they".
  * In response_b, every "you/your" means User B only. Never describe User A's actions, motives, constraints, or words as "you". Refer to User A as "the other person" or "they".
  * Do not use "I", "me", "my", "we", or "our" as the speaker in either response.
- Never use "Can we", "Let's", "We could", or "We should" because those make the mediator a participant.
  Wrong: "Can we find a way to balance this?"
  Right: "Ask the other person whether a balance is possible." or "Consider whether a different boundary would work."
- NEVER open either response with a mediator first-person phrase.
  Wrong: "I can see why...", "I understand...", "I know this is hard..."
  Right: "It makes sense that...", "That sounds frustrating.", "Your position here is...", "Your concern about this is valid."
- Avoid repeated openings like "You are trying to..." or "The other person is..."; vary neutral phrasing naturally.
- The mediator has no voice. Every sentence must address the user as "you" or refer to the other party as "the other person" or "they".
- REFERENCING THE OTHER PERSON:
  * Do not narrate or report the other person's feelings to the user. That is condescending.
  * Wrong: "The other person feels hurt when you do X."
  * Right: "That choice has had an impact that is worth acknowledging."
  * Reference the other person only to acknowledge their constraint, intent, position, boundary, or impact, not to describe their emotional state to the user.
- CONCRETE STEP RULE:
  * The concrete step must be something the addressed user can say or do themselves, not a joint action proposed by the mediator.
  * Wrong: "Can we find time to discuss this?"
  * Wrong: "Could you consider reviewing their concerns?"
  * Right: "Ask the other person what one change would make the biggest difference."
  * Right: "Name the specific boundary you need respected before the next conversation."
- The responses must not mirror each other or share the same opening template.
- Be specific to the actual words in the conflict.
- Avoid therapist-style phrases like "I acknowledge" or "Going forward" when natural wording is possible.
- If non_resolvable, focus on boundaries/coexistence rather than agreement.

ENUM RULES:
- conflict_type must be exactly one of: emotional, logical, misunderstanding, value
- resolvability must be exactly one of: resolvable, partially_resolvable, non_resolvable
- conversation_status must be exactly one of: Escalating, Stable, Improving, Resolved

Return ONLY valid JSON:
{{
  "reasoning_state": {{
    "beliefs": {{
      "user_a_view": "...",
      "user_b_view": "..."
    }},
    "assumptions": ["..."],
    "uncertainties": ["..."],
    "goals": {{
      "user_a": "...",
      "user_b": "..."
    }},
    "selected_strategy": "{strategy["selected_strategy"]}",
    "strategy_source": "{strategy["strategy_source"]}",
    "uncertainty": {{
      "classification": 0.0,
      "reasoning": 0.0
    }}
  }},
  "conflict_type": "emotional",
  "user_a_goal": "underlying real goal of User A",
  "user_b_goal": "underlying real goal of User B",
  "common_ground": "shared need, value, or practical overlap",
  "resolution_strategy": "specific strategy used",
  "one_line_summary": "one sentence capturing the shared tension",
  "response_a": "personalized mediation response for User A",
  "response_b": "personalized mediation response for User B",
  "conversation_status": "Improving"
}}"""


def _verification_issues(
    *,
    text_a: str,
    text_b: str,
    response_a: str,
    response_b: str,
) -> list[str]:
    issues: list[str] = []
    words_a = len(response_a.split())
    words_b = len(response_b.split())
    if words_a < 35:
        issues.append("response_a is too short to validate and guide User A.")
    if words_b < 35:
        issues.append("response_b is too short to validate and guide User B.")
    if fairness_score(response_a, response_b) > 0.15:
        issues.append("responses are unbalanced in length/depth.")
    if _similarity(response_a, response_b) >= 0.82:
        issues.append("responses are too structurally similar.")

    first_a, second_a = _pronoun_counts(response_a)
    first_b, second_b = _pronoun_counts(response_b)
    if first_a > 0:
        issues.append("response_a uses first-person POV instead of addressing User A as you.")
    if first_b > 0:
        issues.append("response_b uses first-person POV instead of addressing User B as you.")
    if second_a < 1:
        issues.append("response_a does not directly address User A as you.")
    if second_b < 1:
        issues.append("response_b does not directly address User B as you.")
    issues.extend(
        _role_lock_issues(
            text_a=text_a,
            text_b=text_b,
            response_a=response_a,
            response_b=response_b,
        )
    )
    mediator_self_ref = (
        r"\bI (understand|acknowledge|appreciate|can see|want|would|think|believe)\b"
        r"|\bcan we\b"
        r"|\blet's\b"
        r"|\bwe could\b"
        r"|\bwe should\b"
        r"|\bwe (should|need to|can|could|will|would|want|must)\b"
        r"|\bour (goal|task|job|approach|role)\b"
    )
    if re.search(mediator_self_ref, response_a, re.I):
        issues.append("response_a uses mediator first-person phrasing.")
    if re.search(mediator_self_ref, response_b, re.I):
        issues.append("response_b uses mediator first-person phrasing.")

    if not _contains_empathy(response_a):
        issues.append("response_a lacks explicit empathy.")
    if not _contains_empathy(response_b):
        issues.append("response_b lacks explicit empathy.")

    overlap_a = _content_words(text_a) & _content_words(response_a)
    overlap_b = _content_words(text_b) & _content_words(response_b)
    if len(_content_words(text_a)) >= 3 and not overlap_a:
        issues.append("response_a is too generic and does not reference User A's actual issue.")
    if len(_content_words(text_b)) >= 3 and not overlap_b:
        issues.append("response_b is too generic and does not reference User B's actual issue.")

    generic_phrases = (
        "this is a difficult situation for both of you",
        "each perspective deserves to be heard",
        "a useful next step is to pause",
    )
    joined = f"{response_a} {response_b}".lower()
    if any(phrase in joined for phrase in generic_phrases):
        issues.append("responses contain generic fallback phrasing.")

    return issues


def _pronoun_counts(text: str) -> tuple[int, int]:
    tokens = re.findall(r"\b[a-z']+\b", text.lower())
    first_person = {"i", "i'm", "i've", "i'd", "me", "my", "mine"}
    second_person = {"you", "you're", "you've", "your", "yours"}
    return (
        sum(1 for token in tokens if token in first_person),
        sum(1 for token in tokens if token in second_person),
    )


def _contains_empathy(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in EMPATHY_MARKERS)


def _inject_empathy(response: str, *, side: str) -> str:
    response = response.strip()
    if _contains_empathy(response):
        return response
    prefix = (
        "It makes sense that this would feel difficult. "
        if side == "a"
        else "It makes sense that your side of this matters too. "
    )
    return prefix + response


def _ensure_empathy(data: dict[str, Any]) -> None:
    data["response_a"] = _inject_empathy(str(data.get("response_a", "")), side="a")
    data["response_b"] = _inject_empathy(str(data.get("response_b", "")), side="b")


def _self_described_phrases(text: str) -> list[str]:
    """Extract short phrases the speaker describes about themself."""
    lowered = text.lower()
    phrases: list[str] = []

    def add_fragment(fragment: str) -> None:
        words = fragment.split()
        if len(words) < 2:
            return
        for size in (min(5, len(words)), min(4, len(words)), min(3, len(words))):
            if size >= 2:
                phrases.append(" ".join(words[:size]))

    for match in re.finditer(
        r"\b(?:i thought i was|i was|i am|i'm|i have been|i've been|i had|i did|i didn't|i meant|i tried|i wanted|i mentioned|i gave|i refined|i built|i shared|i cancelled|i prioritized|i said|i wrote|i feel|i felt|my)\s+([a-z][a-z\s]{4,56})",
        lowered,
    ):
        fragment = re.split(r"[.,;!?]", match.group(1), 1)[0].strip()
        add_fragment(fragment)

    deduped: list[str] = []
    for phrase in phrases:
        if phrase and phrase not in deduped:
            deduped.append(phrase)
    return deduped[:5]


def _response_assigns_phrase_to_you(response: str, phrase: str) -> bool:
    if not phrase:
        return False
    normalized = re.sub(r"\s+", " ", response.lower())
    phrase_re = re.escape(phrase)
    return bool(
        re.search(rf"\byou(?:r| were| are|'re| have|'ve| had|'d)?\b[^.?!]{{0,90}}\b{phrase_re}\b", normalized)
        or re.search(rf"\b{phrase_re}\b[^.?!]{{0,70}}\byou(?:r| were| are|'re| have|'ve| had|'d)?\b", normalized)
    )


def _role_lock_issues(
    *,
    text_a: str,
    text_b: str,
    response_a: str,
    response_b: str,
    ) -> list[str]:
    """Catch cases where a response assigns the other user's action to "you"."""
    issues: list[str] = []
    for phrase in _self_described_phrases(text_b):
        if _response_assigns_phrase_to_you(response_a, phrase):
            issues.append("response_a describes User B's action or intent as if User A did it.")
            break
    for phrase in _self_described_phrases(text_a):
        if _response_assigns_phrase_to_you(response_b, phrase):
            issues.append("response_b describes User A's action or concern as if User B did it.")
            break
    return issues


def _speaker_hint(text: str) -> str:
    lowered = text.lower()
    rules = (
        (("make time", "last priority"), "feeling like time together is no longer a priority"),
        (("working extra hours", "bills"), "working extra hours to cover the bills"),
        (("data", "marketing", "pivot"), "the campaign data and whether a pivot is needed"),
        (("incomplete", "three weeks", "campaign"), "only having three weeks of campaign data"),
        (("leaving the company", "decision", "betrayal"), "sensitive information about a possible company departure"),
        (("mentioned", "exploring options"), "mentioning that options were being explored"),
        (("code review", "release", "nitpicking"), "the code review blocking the release"),
        (("rushed a release", "outage"), "avoiding another outage after a rushed release"),
        (("decisions without me", "ignored"), "being left out when decisions are made"),
        (("helping", "acting quickly"), "trying to help by acting quickly"),
        (("credit", "idea", "leadership"), "credit and ownership for the idea"),
        (("meeting", "2pm"), "the missed meeting and the expected time"),
        (("3pm", "confirmed"), "having a different meeting time written down"),
    )
    for terms, hint in rules:
        if any(term in lowered for term in terms):
            return hint

    words = [
        word
        for word in re.findall(r"[a-z0-9']+", lowered)
        if len(word) > 3 and word not in {"that", "this", "with", "your", "about", "because", "would", "could"}
    ]
    if words:
        return "the concern about " + " ".join(words[:6])
    return "what happened"


_POOL_OPEN_A = (
    "It makes sense that this feels important from your side.",
    "That sounds genuinely difficult from where you are standing.",
    "Your concern here is real and worth naming directly.",
    "It makes sense that the impact on you deserves a clear and steady response.",
    "It makes sense that what you are reacting to deserves to be taken seriously.",
)

_POOL_OPEN_B = (
    "Your intent here sounds different from how it landed.",
    "Your side of this makes sense as something that deserves to be stated clearly.",
    "The concern behind your position makes sense from your side.",
    "What you were trying to protect deserves to be understood accurately.",
    "Your position makes sense and can be explained without dismissing the impact.",
)

_POOL_CONCRETE_A = (
    "Name the one change that would make the biggest difference to you, and ask for that directly.",
    "Ask for the specific acknowledgment or boundary that would help you feel heard right now.",
    "State what you need next in concrete terms, not only what felt wrong.",
    "Name the expectation you need respected before the next conversation continues.",
    "Ask the other person to respond to the impact first, then discuss intent separately.",
)

_POOL_CONCRETE_B = (
    "Acknowledge the impact first, then offer one specific change you can make from your side.",
    "State what you meant clearly, then name one repair step instead of defending the whole situation.",
    "Start with the effect your action had, then explain your reasoning in a way that leaves room for repair.",
    "Offer one concrete adjustment that shows the other person's concern was taken seriously.",
    "Name what you will do differently next time so your intent does not have to carry the whole repair.",
)

_POOL_CONTEXT = {
    "logical": (
        "Keep the next step tied to evidence, risk, timing, or a decision checkpoint so the disagreement does not become personal.",
        "Separate urgency from proof, then ask what information would make the next step responsible.",
    ),
    "misunderstanding": (
        "Keep the focus on what each person understood at the time, because the repair depends on clearer confirmation.",
        "Treat the gap as a meaning problem first, then make the next confirmation step explicit.",
    ),
    "value": (
        "Keep the focus on the value, principle, or boundary at stake instead of trying to win the whole argument.",
        "Clarify what needs to be respected even if the other person does not fully share the same view.",
    ),
    "emotional": (
        "Separate the emotional impact from the other person's stated intent so both can be addressed without either one erasing the other.",
        "Keep the request specific enough that the other person can respond with a real repair rather than a broad apology.",
    ),
}

_POOL_NONRESOLVABLE_A = (
    "It makes sense that this touches a boundary you cannot treat as negotiable. Name the line that needs to be respected, then focus on what coexistence or separation of responsibilities would look like without asking yourself to compromise the principle.",
    "Your position here sounds tied to a real value or safety boundary. State what cannot be crossed, and ask for a practical arrangement that respects that boundary rather than pretending full agreement is available.",
    "It makes sense that this is not the kind of disagreement that needs a forced compromise. Name what you need protected, then keep the next step focused on boundaries, accountability, or distance where needed.",
)

_POOL_NONRESOLVABLE_B = (
    "Your position makes sense as something deeply held, but the next step is not to force agreement. Name what you need respected while you acknowledge that the other person's boundary may remain firm. Keep the request focused on conduct, not conversion.",
    "It makes sense to explain why this matters to you, but do not frame repair as winning the other person over. State your boundary clearly and focus on what respectful coexistence can realistically allow. Avoid asking for a compromise that violates a principle.",
    "This may not be fully solvable through persuasion. Name your value plainly, acknowledge the impact or boundary in front of you, and avoid asking the other person to cross a line they cannot accept. The useful goal is clarity, not forced agreement.",
)

_STALE_MARKERS = (
    "specific impact of what happened",
    "feels personal before it feels practical",
    "intent or pressure connected to",
    "the specific impact of what happened",
    "before it feels practical",
)


def _stable_choice(options: tuple[str, ...], *parts: str) -> str:
    seed = "||".join(part for part in parts if part)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(options)
    return options[index]


def _is_stale(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _STALE_MARKERS)


def build_fallback_responses(
    conflict_type: str,
    resolvability: str,
    *,
    text_a: str = "",
    text_b: str = "",
) -> tuple[str, str]:
    conflict_type = conflict_type if conflict_type in VALID_CONFLICT_TYPES else "emotional"
    resolvability = resolvability if resolvability in VALID_RESOLVABILITY else "partially_resolvable"
    seed = f"{conflict_type}|{resolvability}|{text_a}|{text_b}"

    if resolvability == "non_resolvable":
        response_a = _stable_choice(_POOL_NONRESOLVABLE_A, seed, "a")
        response_b = _stable_choice(_POOL_NONRESOLVABLE_B, seed, "b")
    else:
        context_options = _POOL_CONTEXT.get(conflict_type, _POOL_CONTEXT["emotional"])
        response_a = " ".join(
            (
                _stable_choice(_POOL_OPEN_A, seed, "open-a"),
                _stable_choice(context_options, seed, "context-a"),
                _stable_choice(_POOL_CONCRETE_A, seed, "concrete-a"),
            )
        )
        response_b = " ".join(
            (
                _stable_choice(_POOL_OPEN_B, seed, "open-b"),
                _stable_choice(context_options, seed, "context-b"),
                _stable_choice(_POOL_CONCRETE_B, seed, "concrete-b"),
            )
        )

    if _is_stale(response_a) or _is_stale(response_b):
        return build_fallback_responses(
            conflict_type,
            resolvability,
            text_a=f"{text_a} retry",
            text_b=f"{text_b} retry",
        )
    return response_a, response_b


def _deterministic_repair(
    conflict_type: str,
    resolvability: str = "partially_resolvable",
    *,
    text_a: str = "",
    text_b: str = "",
) -> tuple[str, str]:
    return build_fallback_responses(
        conflict_type,
        resolvability,
        text_a=text_a,
        text_b=text_b,
    )


def _verification_payload(
    *,
    text_a: str,
    text_b: str,
    response_a: str,
    response_b: str,
    repair_applied: bool,
    repair_source: str = "none",
    llm_calls: int = 1,
) -> dict[str, Any]:
    issues = _verification_issues(
        text_a=text_a,
        text_b=text_b,
        response_a=response_a,
        response_b=response_b,
    )
    hard_failed = _hard_verification_failed(issues)
    return {
        "passed": not hard_failed,
        "issues": issues,
        "hard_failed": hard_failed,
        "fairness_score": fairness_score(response_a, response_b),
        "similarity": round(_similarity(response_a, response_b), 3),
        "repair_applied": repair_applied,
        "repair_source": repair_source,
        "llm_calls": llm_calls,
    }


def _hard_verification_failed(issues: list[str]) -> bool:
    hard_markers = (
        "too short",
        "unbalanced",
        "too structurally similar",
        "first-person POV",
        "does not directly address",
        "describes User",
        "mediator first-person",
        "generic fallback phrasing",
    )
    return any(any(marker in issue for marker in hard_markers) for issue in issues)


def _valid_conflict_type(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value in VALID_CONFLICT_TYPES:
        return value
    return fallback if fallback in VALID_CONFLICT_TYPES else "emotional"


def _valid_resolvability(value: Any) -> str:
    if isinstance(value, str) and value in VALID_RESOLVABILITY:
        return value
    return "partially_resolvable"


def _valid_status(value: Any) -> str:
    if isinstance(value, str) and value in VALID_STATUS:
        return value
    return "Stable"


async def _repair_responses(
    *,
    text_a: str,
    text_b: str,
    data: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    repair = await call_groq_production(
        [
            {
                "role": "system",
                "content": (
                    "You repair mediation responses. Keep the reasoning_state and classification stable. "
                    "Change only response_a, response_b, and conversation_status if needed. "
                    "Never use first-person speaker language. Address each user as you. "
                    "Keep response_a locked to User A's perspective and response_b locked to User B's perspective. "
                    "Return valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"""Original conflict:
User A: "{text_a}"
User B: "{text_b}"

Current JSON:
{json.dumps(data, ensure_ascii=False)}

Deterministic verification issues:
{verification["issues"]}

Repair only the listed issues. Keep responses specific, balanced, empathetic, and structurally different.
Required specificity:
- response_a must naturally reference this User A concept: {_speaker_hint(text_a)}
- response_b must naturally reference this User B concept: {_speaker_hint(text_b)}
Required empathy:
- response_a must include a natural empathy phrase like "it makes sense", "that sounds", "you feel", or "you are concerned".
- response_b must include a natural empathy phrase like "it makes sense", "that sounds", "you are trying", or "you are concerned".
POV rule:
- response_a: "you/your" means User A only. Do not assign User B's actions, motives, or words to User A.
- response_b: "you/your" means User B only. Do not assign User A's actions, motives, or words to User B.
- Do not write as "I", "me", "my", "we", or "our".
- Do not use "Can we", "Let's", "We could", or "We should".
- Do not narrate the other person's feelings to the user. Refer to impact, intent, position, or boundary instead.
- The concrete step must be something the addressed user can say or do themselves.

Return ONLY valid JSON:
{{
  "response_a": "repaired response for User A",
  "response_b": "repaired response for User B",
  "conversation_status": "Improving"
}}""",
            },
        ],
        source="production_agent_repair",
        max_tokens=650,
        temperature=0.35,
    )
    return repair


async def run(
    *,
    text_a: str,
    text_b: str,
    retrieved_cases: list[RetrievedCase],
    prior_context: str | None = None,
    intent_a: str = "unknown",
    intent_b: str = "unknown",
) -> LLMCoreOutput:
    classification = _classification_decision(text_a, text_b)
    initial_type = classification["conflict_type"]
    type_for_strategy = initial_type if initial_type != "unknown" else "emotional"
    resolvability = _valid_resolvability(
        classify_resolvability(type_for_strategy, text_a, text_b)
    )
    strategy = _select_strategy(retrieved_cases, type_for_strategy)

    data = await call_groq_production(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_prompt(
                    text_a=text_a,
                    text_b=text_b,
                    classification=classification,
                    resolvability=resolvability,
                    strategy=strategy,
                    prior_context=prior_context,
                    intent_a=intent_a,
                    intent_b=intent_b,
                ),
            },
        ],
        source="production_agent",
        max_tokens=min(GROQ_MAX_TOKENS, 750),
        temperature=0.45,
    )

    if data.get("conflict_type") not in VALID_CONFLICT_TYPES:
        data["conflict_type"] = type_for_strategy if type_for_strategy in VALID_CONFLICT_TYPES else "emotional"
    if data.get("conversation_status") not in VALID_STATUS:
        data["conversation_status"] = "Stable"

    if classification["system_fixed"]:
        conflict_type = type_for_strategy
    else:
        conflict_type = _valid_conflict_type(data.get("conflict_type"), "emotional")
        resolvability = _valid_resolvability(
            classify_resolvability(conflict_type, text_a, text_b) or resolvability
        )

    _ensure_empathy(data)
    llm_calls = 1
    verification = _verification_payload(
        text_a=text_a,
        text_b=text_b,
        response_a=str(data.get("response_a", "")).strip(),
        response_b=str(data.get("response_b", "")).strip(),
        repair_applied=False,
        repair_source="none",
        llm_calls=llm_calls,
    )

    if verification["issues"]:
        initial_verification = verification
        llm_calls += 1
        try:
            repair_data = await _repair_responses(
                text_a=text_a,
                text_b=text_b,
                data=data,
                verification=verification,
            )
            data["response_a"] = str(repair_data["response_a"]).strip()
            data["response_b"] = str(repair_data["response_b"]).strip()
            if repair_data.get("conversation_status"):
                data["conversation_status"] = repair_data["conversation_status"]
            verification = _verification_payload(
                text_a=text_a,
                text_b=text_b,
                response_a=str(data.get("response_a", "")).strip(),
                response_b=str(data.get("response_b", "")).strip(),
                repair_applied=True,
                repair_source="groq_repair",
                llm_calls=llm_calls,
            )
        except Exception as exc:
            verification = {
                **initial_verification,
                "repair_applied": False,
                "repair_source": "groq_repair_failed",
                "repair_error": str(exc),
                "llm_calls": llm_calls,
            }
        if verification["repair_source"] == "groq_repair_failed" and verification["hard_failed"]:
            data["response_a"], data["response_b"] = _deterministic_repair(
                conflict_type,
                resolvability,
                text_a=text_a,
                text_b=text_b,
            )
            if data.get("conversation_status") == "Escalating":
                data["conversation_status"] = "Stable"
            _ensure_empathy(data)
            verification = _verification_payload(
                text_a=text_a,
                text_b=text_b,
                response_a=str(data.get("response_a", "")).strip(),
                response_b=str(data.get("response_b", "")).strip(),
                repair_applied=True,
                repair_source="deterministic_repair",
                llm_calls=llm_calls,
            )
    _ensure_empathy(data)
    verification = {
        **verification,
        **_verification_payload(
            text_a=text_a,
            text_b=text_b,
            response_a=str(data.get("response_a", "")).strip(),
            response_b=str(data.get("response_b", "")).strip(),
            repair_applied=bool(verification.get("repair_applied")),
            repair_source=str(verification.get("repair_source", "none")),
            llm_calls=llm_calls,
        ),
    }

    reasoning_state = data.get("reasoning_state") if isinstance(data.get("reasoning_state"), dict) else {}
    uncertainty = reasoning_state.get("uncertainty", {}) if isinstance(reasoning_state, dict) else {}
    reasoning_uncertainty = uncertainty.get("reasoning", 0.0)
    try:
        reasoning_uncertainty = float(reasoning_uncertainty)
    except (TypeError, ValueError):
        reasoning_uncertainty = 0.0

    goals = reasoning_state.get("goals", {}) if isinstance(reasoning_state.get("goals"), dict) else {}
    reasoning = Reasoning(
        user_a_goal=data.get("user_a_goal") or goals.get("user_a", "to be heard"),
        user_b_goal=data.get("user_b_goal") or goals.get("user_b", "to be understood"),
        conflict_type=conflict_type,
        resolvability=resolvability,
        common_ground=data.get("common_ground") or "both users want the conflict handled more constructively",
        resolution_strategy=data.get("resolution_strategy") or strategy["selected_strategy"],
        one_line_summary=data.get("one_line_summary"),
    )
    reasoning_state = {
        **reasoning_state,
        "selected_strategy": reasoning_state.get("selected_strategy") or strategy["selected_strategy"],
        "strategy_source": reasoning_state.get("strategy_source") or strategy["strategy_source"],
        "classification_confidence": classification["classification_confidence"],
        "classification_system_fixed": classification["system_fixed"],
        "reasoning_uncertainty": reasoning_uncertainty,
    }
    return LLMCoreOutput(
        reasoning=reasoning,
        response_a=str(data.get("response_a", "")).strip(),
        response_b=str(data.get("response_b", "")).strip(),
        conversation_status=_valid_status(data.get("conversation_status")),
        reasoning_state=reasoning_state,
        verification=verification,
    )
