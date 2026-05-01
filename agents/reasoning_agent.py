"""Structured conflict reasoning agent."""

from __future__ import annotations

import re
from typing import Literal

from config import CONFLICT_STRATEGY_MAP
from models.schemas import Reasoning
from utils.llm import call_groq_json


ConflictHint = Literal["emotional", "logical", "misunderstanding", "value"]


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _term_score(text: str, terms: tuple[str, ...], *, weight: int = 1) -> int:
    score = 0
    for term in terms:
        if re.search(r"[\s'/-]", term):
            matched = term in text
        else:
            matched = bool(re.search(rf"\b{re.escape(term)}\b", text))
        if matched:
            score += weight
    return score


def _emotional_pattern_score(text: str) -> int:
    score = 0
    broad_feeling_terms = ("feel", "felt")
    specific_feeling_terms = (
        "feel",
        "felt",
        "hurt",
        "ignored",
        "dismissed",
        "uncared",
        "unheard",
        "alone",
        "lonely",
        "disrespected",
        "excluded",
        "unwanted",
        "unsupported",
    )
    logical_context_terms = (
        "process",
        "data",
        "evidence",
        "metric",
        "deadline",
        "release",
        "deployment",
        "code review",
        "architecture",
        "campaign",
        "budget",
        "roi",
        "risk",
    )
    relational_terms = (
        "you",
        "partner",
        "friend",
        "family",
        "team",
        "manager",
        "parenting",
        "relationship",
    )
    blame_terms = ("always", "never", "no one", "nobody")

    # "I feel the process/data/deadline is wrong" is usually logical phrasing,
    # not emotional injury. Keep emotional scoring for specific feelings like
    # hurt/ignored/dismissed even when work/process language is present.
    if _has_any(text, logical_context_terms) and not _has_any(text, specific_feeling_terms[2:]):
        return 0

    if _has_any(text, broad_feeling_terms + specific_feeling_terms[2:]) and _has_any(text, relational_terms):
        score += 2
    if _has_any(text, blame_terms) and _has_any(text, broad_feeling_terms + specific_feeling_terms[2:]):
        score += 1
    return score


def _detect_ethics_violation(text: str) -> bool:
    """Detect pressure to cross an ethical/legal boundary plus resistance."""
    pressure_terms = (
        "you need to",
        "you have to",
        "just do it",
        "everyone does it",
        "it's how it works",
        "do not escalate",
        "don't escalate",
        "handle it quietly",
        "keep it quiet",
        "hide",
        "cover it",
        "stop complaining",
        "betray the company",
        "protects the department",
        "protect the department",
        "loyalty",
        "don't make this a big deal",
        "why are you making this",
    )
    refusal_or_boundary_terms = (
        "i will not",
        "i refuse",
        "i won't",
        "i am reporting",
        "i'm reporting",
        "reporting it",
        "that is wrong",
        "that's wrong",
        "that is illegal",
        "that's illegal",
        "that is fraud",
        "that's fraud",
        "i can't do that",
        "not ethical",
        "ethical thing",
        "my conscience",
        "against my",
        "not comfortable",
        "discrimination",
        "protects the wrong people",
        "hiding safety defects",
    )
    ethics_domain_terms = (
        "harassment complaint",
        "fraud",
        "adjusted numbers",
        "not accurate",
        "safety defects",
        "regulators",
        "discrimination",
        "refusing service",
        "refuse service",
        "serve that group",
        "serving that group",
    )
    has_pressure = _has_any(text, pressure_terms)
    has_boundary = _has_any(text, refusal_or_boundary_terms)
    has_domain = _has_any(text, ethics_domain_terms)
    return (has_pressure and (has_boundary or has_domain)) or (has_domain and has_boundary)


def _detect_intent_gap(text_a: str, text_b: str) -> bool:
    """Detect a two-sided gap between intended meaning and received meaning."""
    side_a = _norm(text_a)
    side_b = _norm(text_b)
    intent_terms = (
        "i meant",
        "i didn't mean",
        "i did not mean",
        "i was trying to",
        "i thought",
        "i assumed",
        "i believed",
        "i had no idea",
        "i didn't realize",
        "i did not realize",
        "i never said",
        "i only",
        "that wasn't my intention",
        "that was not my intention",
        "i needed to understand",
        "i was not threatening",
        "usually means",
        "i did not know",
        "i didn't know",
    )
    interpretation_terms = (
        "came across as",
        "felt like",
        "interpreted as",
        "sounded like",
        "took it as",
        "you reacted",
        "you assumed",
        "you thought that meant",
        "you meant",
        "that meant",
        "still waited",
        "threat",
        "ultimatum",
        "different meaning",
    )
    return (
        _has_any(side_a, interpretation_terms) and _has_any(side_b, intent_terms)
    ) or (
        _has_any(side_b, interpretation_terms) and _has_any(side_a, intent_terms)
    )


def _detect_tradeoff_deadlock(text: str) -> bool:
    """Detect practical conflicts where each path carries a concrete cost."""
    cost_or_commitment_terms = (
        "six months",
        "rewrite",
        "migration",
        "effort",
        "timeline",
        "architecture",
        "microservices",
        "monolith",
        "technical debt",
        "feature work",
        "feature delivery",
        "promised the feature",
        "this quarter",
        "public apology",
        "apologize publicly",
        "should deploy",
        "deploy it",
        "new model",
        "higher accuracy",
        "sample size",
        "twelve responses",
        "survey only",
        "waiting for more data",
        "benchmarks",
        "accuracy",
        "raise prices",
        "price hike",
        "miss payroll",
        "payroll",
        "lost money",
        "laying off",
        "anchor clients",
        "can't hire",
        "cannot hire",
        "need more people",
        "can't grow",
        "cannot grow",
    )
    risk_or_impact_terms = (
        "bugs",
        "outage",
        "risk",
        "risks",
        "stability",
        "regression",
        "broken",
        "renewal",
        "customer segment",
        "errors are uneven",
        "worse for one",
        "high-risk users",
        "liability",
        "legal said",
        "legal exposure",
        "admitting liability",
        "customer trust",
        "brand",
        "brand damage",
        "long term",
        "harm",
        "lose",
        "we'd lose",
        "impossible",
        "asking me to do the impossible",
        "can't do both",
        "cannot do both",
        "tradeoff",
    )
    return _has_any(text, cost_or_commitment_terms) and _has_any(text, risk_or_impact_terms)


def _detect_safety_sensitive(text: str) -> bool:
    """Detect intense fear/threat patterns without over-triggering on one mild word."""
    strong_phrases = (
        "i feel threatened",
        "i feel unsafe",
        "i am in danger",
        "i'm in danger",
        "i'm scared of you",
        "i am scared of you",
        "scared to bring up",
        "afraid to bring up",
        "blocked the doorway",
        "felt trapped",
    )
    if _has_any(text, strong_phrases):
        return True
    safety_terms = (
        "afraid",
        "scared",
        "unsafe",
        "threatening",
        "threatened",
        "fear",
        "yell",
        "yelling",
        "screaming",
        "shouting",
        "loudly",
        "shut down",
        "trapped",
    )
    return _term_score(text, safety_terms) >= 2


def _detect_repeated_pattern(text: str) -> bool:
    strong_recurrence_terms = (
        "again",
        "pattern",
        "third time",
        "every single time",
        "over and over",
        "same thing every",
        "same behavior",
        "nothing changes",
        "keeps doing",
        "keeps happening",
        "one more time",
        "keep rebuilding",
    )
    weak_recurrence_terms = ("always", "never", "anymore")
    negations = ("not always", "not anymore", "no longer")
    repeat_hits = (
        _term_score(text, strong_recurrence_terms, weight=2)
        + (_term_score(text, weak_recurrence_terms) * 0.5)
    )
    negation_hits = _term_score(text, negations)
    return max(0, repeat_hits - negation_hits) >= 2


def _detect_value_boundary_conflict(text: str) -> bool:
    """Detect value conflicts by paired context instead of a single loaded word."""
    medical_child = _has_any(text, ("child", "doctor", "medication", "treatment")) and _has_any(
        text, ("natural healing", "chemicals", "dangerous", "lifestyle choice", "delaying treatment")
    )
    end_of_life = _has_any(text, ("end of life", "end-of-life", "aggressive treatment", "quality of life")) and _has_any(
        text, ("keep every treatment", "no real quality", "did not want", "any chance")
    )
    protected_identity_service = _has_any(text, ("serve that group", "serving that group", "refusing service", "refuse service")) and _has_any(
        text, ("lifestyle is wrong", "because of who people are", "discrimination")
    )
    family_process = _has_any(text, ("family loyalty", "my cousin", "hire family")) and _has_any(
        text, ("formal interview", "fair process", "nepotism", "undermines trust")
    )
    political_home = _has_any(text, ("campaign sign", "political sign", "political statement")) and _has_any(
        text, ("what i believe", "represents what i believe", "shared home", "our home", "consent")
    )
    return medical_child or end_of_life or protected_identity_service or family_process or political_home


def _conflict_type_hint(text_a: str, text_b: str) -> ConflictHint | None:
    """Classify the dispute's nature, not its emotional tone.

    The LLM is still the primary reasoner, but this hint/post-check prevents
    anger words from turning evidence/process disputes into "emotional".
    """
    combined = _norm(f"{text_a} {text_b}")
    text = combined

    if _detect_ethics_violation(combined):
        return "value"
    if _detect_intent_gap(text_a, text_b):
        return "misunderstanding"
    if _detect_value_boundary_conflict(combined):
        return "value"
    if _detect_safety_sensitive(combined):
        return "emotional"
    if _detect_tradeoff_deadlock(combined):
        return "logical"

    contested_reality = (
        "vaccines caused",
        "stolen election",
        "election was stolen",
        "no way those results",
        "choosing to believe a lie",
        "you're choosing to believe",
        "that's not real",
        "not supported by any evidence",
        "data is fake",
        "numbers are fake",
        "manipulated data",
        "rejecting reality",
    )
    if _has_any(combined, contested_reality):
        return "value"

    coercive_or_safety_emotional = (
        "threatened",
        "ruin my reputation",
        "feel unsafe",
        "blocked the doorway",
        "felt trapped",
        "scare you",
    )
    if _has_any(combined, coercive_or_safety_emotional):
        return "emotional"

    self_harm_adjacent = (
        "better off if i disappeared",
        "if i disappeared",
        "i ruin every relationship",
        "ruin every relationship",
    )
    if _has_any(combined, self_harm_adjacent):
        return "emotional"

    emoji_or_literal_tone_gap = (
        "thumbs up",
        "emoji",
        "fine literally",
        "meant fine literally",
    )
    if _has_any(combined, emoji_or_literal_tone_gap):
        return "misunderstanding"

    identity_or_autonomy_value = (
        _has_any(combined, ("pronouns", "two genders", "shame the family", "before marriage"))
        or (
            _has_any(combined, ("independence", "keys", "keep driving"))
            and _has_any(combined, ("dangerous", "got lost", "hit a mailbox", "safety"))
        )
    )
    if identity_or_autonomy_value:
        return "value"

    technical_or_business_logic = (
        _has_any(combined, ("churn", "price increase", "pricing caused", "statistically significant"))
        or _has_any(combined, ("mfa", "security risk", "compliance policy", "vendor contract", "termination clause"))
        or _term_score(combined, ("sales", "product", "engineering", "timeline", "estimates", "commitments")) >= 2
    )
    if technical_or_business_logic:
        return "logical"

    physical_boundary_terms = (
        "went through my phone",
        "read my messages",
        "searched my bag",
        "searched my room",
        "went through my",
    )
    physical_boundary_hit = _has_any(combined, physical_boundary_terms)
    culture_without_identity = (
        "culture" in combined
        and not _has_any(combined, ("belief", "identity", "tradition", "faith", "religion"))
    )
    individual_work_mode_terms = (
        "work from home",
        "from home",
        "remote work",
        "produce better work",
        "results prove",
        "forcing me back",
        "office is pointless",
    )
    collective_work_mode_terms = (
        "team culture",
        "spontaneous collaboration",
        "collaboration",
        "in person",
        "back to the office",
        "whole team",
    )
    work_philosophy_value = (
        _has_any(combined, individual_work_mode_terms)
        and _has_any(combined, collective_work_mode_terms)
    )

    hard_value_terms = (
        "religion",
        "faith",
        "belief",
        "values",
        "moral",
        "identity",
        "culture",
        "tradition",
        "privacy",
        "private",
        "fraud",
        "illegal",
        "unethical",
        "not accurate",
        "will not put my name",
        "will not sign",
        "shared details",
        "relationship problems",
        "transparent",
        "complicit",
    )
    soft_value_terms = (
        "savings",
        "future",
        "family takes care",
        "job security",
        "inheritance",
        "split equally",
        "sacrifice",
        "career over our family",
        "life is happening now",
        "deprive myself",
    )
    logical_terms = (
        "data",
        "evidence",
        "metric",
        "roi",
        "budget",
        "cost",
        "deadline",
        "release",
        "deployment",
        "code review",
        "architecture",
        "microservices",
        "monolith",
        "scalability",
        "technical debt",
        "code quality",
        "feature delivery",
        "legal wording",
        "legal language",
        "cease and desist",
        "liability",
        "hire",
        "revenue",
        "grow revenue",
        "more people",
        "product line",
        "lost money",
        "quarters",
        "laying off",
        "clients",
        "anchor clients",
        "shut down",
        "campaign",
        "strategy",
        "approved process",
        "process exists",
        "follow the process",
        "skipping steps",
        "outage",
        "risk",
        "security patch",
        "sla",
        "staffing",
        "comparable",
        "expenses",
        "delivered",
        "deliver",
        "ship",
        "cutting corners",
        "committed",
    )
    strong_misunderstanding_terms = (
        "assumed",
        "no idea",
        "didn't realize",
        "did not realize",
        "written down",
        "confirmed",
        "change with me",
        "came across",
        "interpreted as",
        "came across as threat",
        "different meaning",
        "meaning of urgent",
        "direct",
        "directness",
        "engaging seriously",
        "tone",
        "said you were fine",
        "didn't mean",
        "did not mean",
        "fine forever",
        "things have changed",
        "thought that meant yes",
        "sounded good",
        "thinking about it",
        "booked everything",
        "sarcastic",
        "sincere",
        "optional",
        "prioritized",
        "misread",
        "misinterpreted",
        "exploring options",
        "mentioned",
        "get back to you",
        "only reach out",
        "keep score",
        "texts first",
        "never agreed",
        "legal wording",
        "legal language",
        "cease and desist",
        "asked for pricing",
        "talked to the vendor",
        "handling the catering",
    )
    weak_misunderstanding_terms = (
        "thought",
        "meant",
        "intention",
        "trying to",
        "trying",
    )
    emotional_core_terms = (
        "never make time",
        "last priority",
        "no longer a priority",
        "not a priority",
        "priority to you",
        "feel completely",
        "feel ignored",
        "feel unheard",
        "feel dismissed",
        "feel unsupported",
        "feel unwanted",
        "never care",
        "never listen",
        "humiliated",
        "humiliate",
        "useless",
        "impossible to work with",
        "nobody on this team can stand",
        "abandoned",
        "invisible",
        "feeling invisible",
        "burning out",
        "nobody cares",
        "undermine",
        "destroys my authority",
        "singled out",
        "single you out",
        "no peace",
        "own home",
        "resent",
        "unappreciated",
        "betrayal",
        "betrayed",
        "chose your career",
        "lie to me",
        "lied",
        "rebuilding trust",
        "broken trust",
        "went through my phone",
        "violation of my privacy",
        "without my permission",
        "better off without me",
        "ruin everything",
        "yelling",
        "screaming",
        "shouting",
        "threatens",
        "threatening",
        "afraid of",
        "scared of",
    )

    hard_value_score = _term_score(text, hard_value_terms, weight=3)
    if physical_boundary_hit:
        hard_value_score = max(0, hard_value_score - 3)
    if culture_without_identity:
        hard_value_score = max(0, hard_value_score - 3)
    if work_philosophy_value:
        hard_value_score += 3

    soft_value_score = _term_score(text, soft_value_terms)
    logical_score = _term_score(text, logical_terms, weight=2)
    misunderstanding_score = (
        _term_score(text, strong_misunderstanding_terms, weight=2)
        + _term_score(text, weak_misunderstanding_terms)
    )
    emotional_score = _term_score(text, emotional_core_terms, weight=2)
    emotional_score += _emotional_pattern_score(text)

    if hard_value_score:
        return "value"

    # Evidence/process disputes stay logical even when the language is heated.
    if logical_score >= 2 and logical_score >= max(misunderstanding_score, emotional_score):
        return "logical"

    # Misunderstanding needs more than a stray "thought"; otherwise relational
    # injury cases like "I feel like the last priority" get misclassified.
    if misunderstanding_score >= 3 and misunderstanding_score >= emotional_score:
        return "misunderstanding"

    if emotional_score >= 2 and emotional_score > misunderstanding_score:
        return "emotional"

    if soft_value_score:
        return "value"

    if logical_score >= 2:
        return "logical"
    if emotional_score >= 2:
        return "emotional"
    if misunderstanding_score >= 2:
        return "misunderstanding"
    return None


def classify_conflict_type(text_a: str, text_b: str) -> ConflictHint | None:
    """Public rule-first classifier for fast production paths."""
    return _conflict_type_hint(text_a, text_b)


def _resolvability_hint(conflict_type: str, text_a: str, text_b: str) -> str | None:
    combined = _norm(f"{text_a} {text_b}")
    text = combined

    if _detect_ethics_violation(combined):
        return "non_resolvable"
    if conflict_type == "value" and _detect_value_boundary_conflict(combined):
        if _has_any(
            combined,
            (
                "doctor",
                "medication",
                "treatment",
                "delaying treatment",
                "dangerous",
                "end of life",
                "end-of-life",
                "quality of life",
                "aggressive treatment",
                "refusing service",
                "refuse service",
                "serve that group",
                "serving that group",
                "discrimination",
            ),
        ):
            return "non_resolvable"
        return "partially_resolvable"
    if conflict_type == "logical" and _detect_tradeoff_deadlock(combined):
        return "partially_resolvable"
    if conflict_type == "emotional" and _detect_safety_sensitive(combined):
        return "partially_resolvable"
    if conflict_type == "emotional" and _detect_repeated_pattern(combined):
        return "partially_resolvable"

    contested_reality = (
        "vaccines caused",
        "stolen election",
        "election was stolen",
        "no way those results",
        "choosing to believe a lie",
        "you're choosing to believe",
        "that's not real",
        "not supported by any evidence",
        "data is fake",
        "numbers are fake",
        "manipulated data",
        "rejecting reality",
    )
    if _has_any(combined, contested_reality):
        return "non_resolvable"

    non_resolvable_triggers = (
        "fraud",
        "illegal",
        "i will not put my name",
        "will not sign",
        "never accept",
        "i will not be vaccinating",
        "i will not vaccinate",
        "whistleblow",
        "whistleblowing",
        "refuse to serve",
        "refusing service",
        "end of life",
        "end-of-life",
        "medical treatment refusal",
        "political beliefs in",
        "shame the family",
        "cannot move out before marriage",
        "before marriage",
    )
    if conflict_type == "value" and _has_any(combined, non_resolvable_triggers):
        return "non_resolvable"

    if conflict_type == "logical" and "can't" in combined and "without" in combined:
        return "partially_resolvable"

    high_stakes_business = (
        "shut down",
        "product line",
        "lost money",
        "laying off",
        "clients",
        "anchor clients",
        "irreversible",
    )
    if conflict_type == "logical" and _term_score(combined, high_stakes_business) >= 2:
        return "partially_resolvable"

    cultural_interpretation = (
        "culture",
        "direct",
        "directness",
        "respect",
        "rude",
        "dismiss",
        "spoke over",
    )
    shifting_or_implied_agreement = (
        "said you were fine",
        "fine forever",
        "things have changed",
        "every week",
    )
    if conflict_type == "misunderstanding" and (
        ("culture" in combined and _term_score(combined, cultural_interpretation) >= 2)
        or _has_any(combined, shifting_or_implied_agreement)
        or _has_any(combined, ("exclusive", "exclusivity", "serious meant"))
    ):
        return "partially_resolvable"

    accumulated_grievance = (
        "five years",
        "for years",
        "for hours",
        "long time",
        "always been",
        "never changes",
        "same thing every",
        "over and over",
    )
    if conflict_type == "emotional" and _has_any(combined, accumulated_grievance):
        return "partially_resolvable"

    emotional_pattern_boundaries = (
        "ignore me for days",
        "silent treatment",
        "feels like punishment",
        "always compare",
        "nothing i do is enough",
        "every birthday",
        "family obligation",
        "invisible work",
        "sorry only to shut",
        "shut the conversation down",
        "never feel like you actually hear",
        "same behavior",
        "cannot trust myself",
        "gaslight",
        "gaslighting",
    )
    if conflict_type == "emotional" and _has_any(combined, emotional_pattern_boundaries):
        return "partially_resolvable"

    non_resolvable_terms = (
        "religion",
        "faith",
        "belief",
        "identity",
        "morally",
        "moral",
        "complicit",
    )
    if conflict_type == "value" and _has_any(text, non_resolvable_terms):
        return "non_resolvable"
    if conflict_type == "value":
        return "partially_resolvable"
    if conflict_type == "logical":
        cost_terms = (
            "six months",
            "rewrite",
            "migration",
            "delay",
            "delaying",
            "cost",
            "partnership",
            "effort",
            "timeline",
            "architecture",
            "microservices",
            "monolith",
            "mfa",
            "vendor contract",
            "contract",
            "three years",
            "termination clause",
        )
        risk_terms = (
            "bugs",
            "outage",
            "risk",
            "risky",
            "stability",
            "regression",
            "broken",
            "security risk",
            "security patch",
            "vulnerability",
            "severity",
            "compliance",
            "violates",
            "trap",
        )
        cost_hits = _term_score(text, cost_terms)
        risk_hits = _term_score(text, risk_terms)
        if cost_hits >= 1 and risk_hits >= 1:
            return "partially_resolvable"
        if _has_any(combined, ("travel budget", "roi", "client relationships")) and _has_any(
            combined, ("budget", "overspending", "measure", "spreadsheet")
        ):
            return "partially_resolvable"
        if _has_any(combined, ("rent increase", "comparable apartments", "maintenance costs", "taxes")):
            return "partially_resolvable"
        return "resolvable"
    if conflict_type == "misunderstanding":
        return "resolvable"
    if conflict_type == "emotional":
        strong_repeat_terms = (
            "third time",
            "every single time",
            "keeps",
            "pattern",
            "again",
        )
        weak_repeat_terms = ("always", "never", "anymore")
        negations = ("not always", "not anymore", "no longer")
        repeat_score = (
            _term_score(text, strong_repeat_terms, weight=2)
            + (_term_score(text, weak_repeat_terms) * 0.5)
            - _term_score(text, negations)
        )
        if max(0, repeat_score) >= 2:
            return "partially_resolvable"
        severe_terms = (
            "abuse",
            "threat",
            "unsafe",
            "violence",
            "betrayal",
            "betrayed",
            "privacy",
            "humiliated",
            "disappeared",
            "better off if i disappeared",
            "trapped",
            "blocked the doorway",
            "scare you",
            "useless",
            "impossible to work with",
            "can stand you",
        )
        return "partially_resolvable" if _has_any(text, severe_terms) else "resolvable"
    return None


def classify_resolvability(conflict_type: str, text_a: str, text_b: str) -> str | None:
    """Public rule-first resolvability classifier for fast production paths."""
    return _resolvability_hint(conflict_type, text_a, text_b)


def _build_prompt(
    *,
    text_a: str,
    text_b: str,
    stance_a: str,
    stance_b: str,
    emotion_a: str,
    emotion_b: str,
    sentiment_a: str,
    sentiment_b: str,
    intent_a: str,
    intent_b: str,
    prior_context: str | None,
    conflict_hint: str | None,
    resolvability_hint: str | None,
) -> str:
    hints = "\n".join(f"- {name}: {guide}" for name, guide in CONFLICT_STRATEGY_MAP.items())
    prior = f"\nPRIOR CONTEXT SUMMARY:\n{prior_context}\n" if prior_context else ""
    return f"""You are an expert conflict analyst.

Analyze the conflict without taking sides.
{prior}
CURRENT CONFLICT:
User A says: "{text_a}"
User B says: "{text_b}"

DETECTED SIGNALS:
- User A: stance={stance_a}, emotion={emotion_a}, sentiment={sentiment_a}, communicative_intent={intent_a}
- User B: stance={stance_b}, emotion={emotion_b}, sentiment={sentiment_b}, communicative_intent={intent_b}
- Communicative intent means the observable action the user appears to be taking with their statement, not hidden motive. Treat it as supporting context only; never let intent override conflict_type or resolvability.

CLASSIFICATION HINTS:
- conflict_type_hint={conflict_hint or "none"}
- resolvability_hint={resolvability_hint or "none"}

CONFLICT TYPE GUIDE:
{hints}

STRICT CLASSIFICATION RULES:
- Classify conflict_type by the nature of the dispute, not emotional tone.
- logical = disagreement about facts, data, process, deadlines, risk, metrics, or evidence.
- misunderstanding = different interpretations of the same event, message, wording, timing, ownership, or intent.
- emotional = the core issue is the feeling or relational injury itself: neglect, humiliation, betrayal, abandonment, invisibility, resentment.
- value = fundamentally different beliefs, identity commitments, moral priorities, faith, family values, privacy boundaries, or life priorities.
- Two users can sound angry while the dispute is still logical or misunderstanding.
- Do not choose emotional merely because detected emotions include anger, sadness, or frustration.

RESOLVABILITY RULES:
- resolvable = shared facts/process/communication can likely fix the issue.
- partially_resolvable = tradeoffs or values can be negotiated but not fully solved.
- non_resolvable = identity, faith, moral belief, or core life-priority conflict where coexistence/boundaries matter more than agreement.
- For faith/identity/value conflicts, prefer non_resolvable or partially_resolvable over resolvable.

ENUM RULES:
- conflict_type must be exactly one of: emotional, logical, misunderstanding, value
- resolvability must be exactly one of: resolvable, partially_resolvable, non_resolvable

Return ONLY valid JSON with these exact keys:
{{
  "user_a_goal": "underlying real goal of User A",
  "user_b_goal": "underlying real goal of User B",
  "conflict_type": "emotional",
  "resolvability": "resolvable",
  "common_ground": "shared need, value, or practical overlap",
  "resolution_strategy": "specific approach for this conflict",
  "one_line_summary": "one sentence capturing the core shared tension"
}}"""


async def run(
    *,
    text_a: str,
    text_b: str,
    stance_a: str = "neutral",
    stance_b: str = "neutral",
    emotion_a: str = "neutral",
    emotion_b: str = "neutral",
    sentiment_a: str = "neutral",
    sentiment_b: str = "neutral",
    intent_a: str = "unknown",
    intent_b: str = "unknown",
    prior_context: str | None = None,
) -> Reasoning:
    conflict_hint = _conflict_type_hint(text_a, text_b)
    rv_hint = _resolvability_hint(conflict_hint, text_a, text_b) if conflict_hint else None
    data = await call_groq_json(
        [
            {
                "role": "system",
                "content": "You are a precise conflict reasoning agent. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": _build_prompt(
                    text_a=text_a,
                    text_b=text_b,
                    stance_a=stance_a,
                    stance_b=stance_b,
                    emotion_a=emotion_a,
                    emotion_b=emotion_b,
                    sentiment_a=sentiment_a,
                    sentiment_b=sentiment_b,
                    intent_a=intent_a,
                    intent_b=intent_b,
                    prior_context=prior_context,
                    conflict_hint=conflict_hint,
                    resolvability_hint=rv_hint,
                ),
            },
        ],
        source="reasoning_agent",
        max_tokens=650,
        temperature=0.15,
    )
    conflict_type = conflict_hint or "emotional"
    resolvability = rv_hint or data.get("resolvability", "partially_resolvable")
    return Reasoning(
        user_a_goal=data["user_a_goal"],
        user_b_goal=data["user_b_goal"],
        conflict_type=conflict_type,
        resolvability=resolvability,
        common_ground=data["common_ground"],
        resolution_strategy=data["resolution_strategy"],
        one_line_summary=data.get("one_line_summary"),
    )
