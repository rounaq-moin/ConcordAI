"""Empathetic dual-response generation agent."""

from __future__ import annotations

from config import GROQ_MAX_TOKENS, RAG_FINAL_K
from models.schemas import LLMCoreOutput, Reasoning, RetrievedCase
from utils.llm import call_groq_json
from utils.scoring import fairness_score


SYSTEM_PROMPT = """You are an advanced AI conflict mediator.

Your purpose is not to force agreement. You must:
- deeply understand both perspectives
- reduce emotional escalation
- avoid taking sides
- validate both users without declaring either side simply right
- respect non-resolvable conflicts by focusing on boundaries and coexistence
- sound natural, calm, specific, and human
- keep both responses roughly equal in length and depth
- produce two materially different responses, one tailored to each user

Return only valid JSON."""


BAD_OPENERS = (
    "i can",
    "i understand",
    "i know",
    "i see",
    "i want",
    "i think",
    "i appreciate",
)


def _clean_rag_response(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.lower().startswith(BAD_OPENERS):
        return ""
    return cleaned[:80]


def _examples(cases: list[RetrievedCase]) -> str:
    if not cases:
        return "No sufficiently similar RAG examples were used for this turn."
    chunks = ["Relevant resolved examples for tone and strategy:"]
    for idx, case in enumerate(cases[:RAG_FINAL_K], 1):
        response_a = _clean_rag_response(case.response_a)
        response_b = _clean_rag_response(case.response_b)
        response_lines = ""
        if response_a:
            response_lines += f"Response A style: {response_a}\n"
        if response_b:
            response_lines += f"Response B style: {response_b}\n"
        chunks.append(
            f"""
Example {idx} [{case.conflict_type}, similarity={case.relevance_score}]
Strategy: {case.resolution_strategy}
{response_lines.rstrip()}
"""
        )
    return "\n".join(chunks)


def _resolvability_guidance(reasoning: Reasoning) -> str:
    return {
        "resolvable": "Guide toward a practical next step and potential agreement.",
        "partially_resolvable": "Reduce tension and identify workable middle ground without overpromising.",
        "non_resolvable": "Do not force agreement; guide toward respect, boundaries, and coexistence.",
    }.get(reasoning.resolvability, "Reduce tension while respecting uncertainty.")


def _build_prompt(
    *,
    text_a: str,
    text_b: str,
    reasoning: Reasoning,
    retrieved_cases: list[RetrievedCase],
    feedback: str | None,
) -> str:
    rejected = f"\nPREVIOUS ATTEMPT REJECTED. Fix this:\n{feedback}\n" if feedback else ""
    return f"""Generate two separate mediation responses.

CONFLICT:
User A: "{text_a}"
User B: "{text_b}"

REASONING:
- User A goal: {reasoning.user_a_goal}
- User B goal: {reasoning.user_b_goal}
- Conflict type: {reasoning.conflict_type}
- Resolvability: {reasoning.resolvability}
- Common ground: {reasoning.common_ground}
- Strategy: {reasoning.resolution_strategy}

CONFLICT TYPE DEFINITIONS:
- emotional: the core injury is a feeling: neglect, invisibility, betrayal, humiliation, abandonment, resentment.
- logical: the disagreement is about facts, evidence, data, process, deadlines, tradeoffs, risk, or decision quality.
- misunderstanding: both users are working from different interpretations of the same event, message, timing, ownership, or intent.
- value: the conflict is about identity, faith, moral belief, privacy boundaries, life priorities, family expectations, or non-negotiable principles.

RESOLVABILITY INSTRUCTIONS:
- resolvable: both responses should end with one concrete, actionable next step toward agreement.
- partially_resolvable: both responses should identify one workable compromise or boundary without promising full resolution.
- non_resolvable: do not suggest agreement or compromise. Focus on what each user needs respected and what coexistence or boundaries look like.

STRATEGY INTERPRETATION:
- Treat the strategy as a step-by-step execution plan, not a topic summary.
- The first sentence of each response must implement the first relevant action in the strategy.
- If the strategy says "validate feelings first", response_a must open by naming the specific feeling or impact from User A's words.
- If it says "acknowledge intent", response_b must open by naming what User B was trying to do or protect.
- Do not summarize the strategy. Execute it.

GUIDANCE:
{_resolvability_guidance(reasoning)}

RAG CONTEXT:
{_examples(retrieved_cases)}
{rejected}
REQUIREMENTS:
- When RAG context exists, explicitly apply one relevant retrieved strategy or tone pattern, but adapt it to this exact conflict instead of copying the example.
- Response A: speak directly to User A. Prefer starting from the impact on A, naming the unmet need, briefly acknowledging User B's constraint or intent, then suggesting one concrete next step A can ask for.
- Response B: speak directly to User B. Prefer starting from B's intent, constraint, or concern, then clearly acknowledging the impact on User A, then suggesting one concrete repair step B can offer.
- CRITICAL: response_a and response_b must use different structures. Do not mirror the same sentence order, same opening phrase, same transition, or same advice phrasing.
- Do not begin both responses with "You feel", "I understand", "It's understandable", or any shared template phrase.
- Do not use "To move forward" in both responses. Prefer natural, varied transitions.
- Never write response_a or response_b as if you are User A or User B speaking. Address each user as "you"; do not roleplay them in first person.
- Never use "Can we", "Let's", "We could", or "We should" because those make the mediator a participant.
- Do not narrate or report the other person's feelings to the user.
  Wrong: "The other person feels hurt when you do X."
  Right: "That choice has had an impact that is worth acknowledging."
- The concrete step must be something the addressed user can say or do themselves, not a joint action proposed by the mediator.
- Be specific to this exact conflict. Name the actual issue using the users' words rather than generic labels.
- Use warm, concise, human language. Avoid corporate, robotic, or overly perfect therapy-template phrasing.
- Both responses should be roughly equal in length and depth.

ENUM RULES:
- conversation_status must be exactly one of: Escalating, Stable, Improving, Resolved

Return ONLY valid JSON:
{{
  "response_a": "personalized response for User A only",
  "response_b": "personalized response for User B only",
  "conversation_status": "Improving"
}}"""


async def run(
    *,
    text_a: str,
    text_b: str,
    reasoning: Reasoning,
    retrieved_cases: list[RetrievedCase],
    feedback: str | None = None,
) -> LLMCoreOutput:
    async def generate(extra_feedback: str | None = None) -> dict:
        joined_feedback = "\n".join(part for part in (feedback, extra_feedback) if part)
        return await call_groq_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_prompt(
                        text_a=text_a,
                        text_b=text_b,
                        reasoning=reasoning,
                        retrieved_cases=retrieved_cases,
                        feedback=joined_feedback or None,
                    ),
                },
            ],
            source="reconciliation_agent",
            max_tokens=GROQ_MAX_TOKENS,
            temperature=0.5,
        )

    data = await generate()
    response_a = data["response_a"]
    response_b = data["response_b"]
    fscore = fairness_score(response_a, response_b)
    if fscore > 0.3:
        print(f"[reconciliation_agent] fairness warning: {fscore}")
        data = await generate(
            (
                f"Fairness score was {fscore}, which is too unbalanced. "
                "Regenerate with response_a and response_b closer in length and depth, "
                "while keeping them structurally different and specific."
            )
        )
        response_a = data["response_a"]
        response_b = data["response_b"]
    return LLMCoreOutput(
        reasoning=reasoning,
        response_a=response_a,
        response_b=response_b,
        conversation_status=data.get("conversation_status", "Stable"),
    )
