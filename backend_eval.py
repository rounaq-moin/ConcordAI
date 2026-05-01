"""Backend evaluation runner for the AI Conflict Mediation System.

Runs cases from test_cases.json directly through the LangGraph pipeline.

Usage:
    python backend_eval.py
    python backend_eval.py --file path/to/test_cases.json
    python backend_eval.py --ids T01 T04 T05
    python backend_eval.py --category value
    python backend_eval.py --save results.json
    python backend_eval.py --no-warmup
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from difflib import SequenceMatcher
from pathlib import Path


try:
    from agents import safety_agent, validator
    from agents.retriever import seed_from_scenarios
    from coordinator.graph import run_pipeline
    from models.schemas import UserInput
except ImportError as exc:
    print(f"[error] Could not import pipeline: {exc}")
    print("Run this script from the project root directory.")
    sys.exit(1)


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

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


def ok(text: str) -> str:
    return f"{GREEN}[PASS] {text}{RESET}"


def fail(text: str) -> str:
    return f"{RED}[FAIL] {text}{RESET}"


def warn(text: str) -> str:
    return f"{YELLOW}[WARN] {text}{RESET}"


def soft_pass(text: str) -> str:
    return f"{GREEN}[SOFT PASS] {text}{RESET}"


def soft_mismatch(text: str) -> str:
    return f"{YELLOW}[SOFT MISMATCH] {text}{RESET}"


def hdr(text: str) -> str:
    return f"{BOLD}{CYAN}{text}{RESET}"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _combined_text(text_a: str, text_b: str) -> str:
    return " ".join((text_a + " " + text_b).lower().split())


def _cue_hits(combined: str, cues: list[str]) -> int:
    return sum(1 for cue in cues if cue in combined)


def infer_repeat_pattern(combined: str) -> bool:
    repeat_hits = _cue_hits(combined, REPEAT_CUES)
    negation_hits = _cue_hits(combined, NEGATION_CUES)
    return max(0, repeat_hits - negation_hits) > 0


def infer_ethics_refusal(combined: str) -> bool:
    return any(cue in combined for cue in ETHICS_CUES)


def infer_safety_sensitive(combined: str, safety_score: float) -> bool:
    if safety_score >= 0.4:
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


def expected_behavior(case: dict) -> dict:
    return {
        "should_use_fallback": bool(case.get("should_use_fallback", False)),
        "is_safety_sensitive": bool(case.get("is_safety_sensitive", False)),
        "intensity": case.get("intensity", "medium"),
        "is_repeat_pattern": bool(case.get("is_repeat_pattern", False)),
        "expected_strategy": case.get("expected_strategy", STRATEGY_BY_CONFLICT.get(case.get("expected_conflict_type", ""), "emotional_validation")),
    }


def make_soft_check(expected, actual) -> dict:
    return {"expected": expected, "actual": actual, "match": expected == actual}


def build_soft_checks(case: dict, result, trace, safety_score: float) -> tuple[dict, dict]:
    combined = _combined_text(case["text_a"], case["text_b"])
    repeat_pattern = infer_repeat_pattern(combined)
    ethics_refusal = infer_ethics_refusal(combined)
    safety_sensitive = infer_safety_sensitive(combined, safety_score)
    actual_strategy = infer_strategy(
        result.conflict_type,
        safety_sensitive=safety_sensitive,
        ethics_refusal=ethics_refusal,
    )
    actual_behavior = {
        "should_use_fallback": bool(trace.fallback_used) if trace else False,
        "is_safety_sensitive": safety_sensitive,
        "intensity": infer_intensity(
            combined,
            result.conflict_type,
            safety_sensitive=safety_sensitive,
            repeat_pattern=repeat_pattern,
            ethics_refusal=ethics_refusal,
        ),
        "is_repeat_pattern": repeat_pattern,
        "expected_strategy": actual_strategy,
    }
    expected = expected_behavior(case)
    return expected, {
        name: make_soft_check(expected[name], actual_behavior[name])
        for name in expected
    }


def check_distinct(response_a: str, response_b: str) -> tuple[bool, str]:
    norm_a = _norm(response_a)
    norm_b = _norm(response_b)
    if not norm_a or not norm_b:
        return False, "empty response"
    if norm_a == norm_b:
        return False, "identical"
    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    if ratio >= 0.86:
        return False, f"near-identical (similarity {ratio:.0%})"
    words_a = set(norm_a.split())
    words_b = set(norm_b.split())
    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    return True, f"distinct (similarity {ratio:.0%}, overlap {overlap:.0%})"


def check_specific(response: str, min_words: int = 30) -> tuple[bool, str]:
    words = response.split()
    if len(words) < min_words:
        return False, f"too short ({len(words)} words)"
    generic_phrases = [
        "this is a difficult moment for both of you",
        "each perspective deserves to be heard",
        "a useful next step is to pause",
        "both users need to feel heard",
    ]
    lowered = response.lower()
    for phrase in generic_phrases:
        if phrase in lowered:
            return False, f"contains fallback phrase: {phrase}"
    return True, f"{len(words)} words"


def check_empathy(response: str) -> tuple[bool, str]:
    markers = [
        "understand",
        "feel",
        "hear",
        "makes sense",
        "valid",
        "acknowledge",
        "appreciate",
        "recognize",
        "see why",
        "sense that",
        "sounds",
        "deserves",
        "worth acknowledging",
        "worth naming",
    ]
    found = [marker for marker in markers if marker in response.lower()]
    if not found:
        return False, "no empathy markers"
    return True, f"markers: {found[:3]}"


def check_conflict_type(actual: str, expected: str) -> tuple[bool, str]:
    if actual == expected:
        return True, actual
    return False, f"got '{actual}', expected '{expected}'"


def check_resolvability(actual: str, expected: str) -> tuple[bool | None, str]:
    if actual == expected:
        return True, actual
    adjacent = {
        "resolvable": ["partially_resolvable"],
        "partially_resolvable": ["resolvable", "non_resolvable"],
        "non_resolvable": ["partially_resolvable"],
    }
    if actual in adjacent.get(expected, []):
        return None, f"adjacent: got '{actual}', expected '{expected}'"
    return False, f"got '{actual}', expected '{expected}'"


def _pronoun_counts(text: str) -> tuple[int, int]:
    tokens = re.findall(r"\b[a-z']+\b", text.lower())
    first_person = {"i", "i'm", "i've", "i'd", "me", "my", "mine", "we", "we're", "we've", "our"}
    second_person = {"you", "you're", "you've", "your", "yours"}
    return (
        sum(1 for token in tokens if token in first_person),
        sum(1 for token in tokens if token in second_person),
    )


def check_pov(response_a: str, response_b: str) -> tuple[bool, str]:
    first_a, second_a = _pronoun_counts(response_a)
    first_b, second_b = _pronoun_counts(response_b)
    if first_a > second_a + 5:
        return False, f"Response A first-person heavy: first={first_a}, you={second_a}"
    if first_b > second_b + 5:
        return False, f"Response B first-person heavy: first={first_b}, you={second_b}"
    return True, f"A(first={first_a}, you={second_a}) B(first={first_b}, you={second_b})"


async def run_test(
    case: dict,
    idx: int,
    total: int,
    mode_override: str | None = None,
    *,
    full_output: bool = False,
) -> dict:
    tid = case["id"]
    print(f"\n{hdr(f'[{idx}/{total}] {tid} - {case["description"]}')}")
    print(f"  A: {case['text_a'][:90]}...")
    print(f"  B: {case['text_b'][:90]}...")

    user_input = UserInput(
        conversation_id=str(uuid.uuid4()),
        text_a=case["text_a"],
        text_b=case["text_b"],
        turn=1,
        mode=mode_override or case.get("mode", "fast"),
        request_id=str(uuid.uuid4()),
        trace_id=tid,
    )

    started = time.time()
    try:
        result = await run_pipeline(user_input)
        error = None
    except Exception as exc:
        result = None
        error = str(exc)
    elapsed = round(time.time() - started, 2)

    if error or result is None:
        print(f"  {fail(f'PIPELINE ERROR: {error}')}")
        return {
            "id": tid,
            "category": case["category"],
            "description": case["description"],
            "passed": False,
            "error": error,
            "elapsed": elapsed,
            "checks": {},
            "expected_behavior": expected_behavior(case),
            "soft_checks": {},
        }

    trace = result.trace
    checks: dict[str, tuple[bool | None, str]] = {}
    checks["conflict_type"] = check_conflict_type(
        result.conflict_type, case.get("expected_conflict_type", "")
    )
    checks["resolvability"] = check_resolvability(
        result.resolvability, case.get("expected_resolvability", "")
    )
    checks["distinct_responses"] = check_distinct(result.response_a, result.response_b)
    checks["specific_a"] = check_specific(result.response_a)
    checks["specific_b"] = check_specific(result.response_b)
    checks["empathy_a"] = check_empathy(result.response_a)
    checks["empathy_b"] = check_empathy(result.response_b)
    checks["pov_correct"] = check_pov(result.response_a, result.response_b)
    safety_score = trace.safety_score if trace and trace.safety_score is not None else 0.0
    checks["safety"] = (safety_score < 0.4, f"score={safety_score:.3f}")
    expected_soft, soft_checks = build_soft_checks(case, result, trace, safety_score)

    print(f"\n  Summary: {result.one_line_summary or '-'}")
    print(
        f"  Status={result.conversation_status} | Conflict={result.conflict_type} | "
        f"Resolvability={result.resolvability} | Confidence={result.confidence:.0%} | "
        f"Retries={result.retries} | RAG={'yes' if (trace and trace.rag_used) else 'no'} | "
        f"Fallback={'yes' if (trace and trace.fallback_used) else 'no'} | Time={elapsed}s"
    )
    print()
    print(f"  Response A ({len(result.response_a.split())} words):")
    response_a_text = result.response_a if full_output else f"{result.response_a[:240]}{'...' if len(result.response_a) > 240 else ''}"
    print(f"    {response_a_text}")
    print(f"  Response B ({len(result.response_b.split())} words):")
    response_b_text = result.response_b if full_output else f"{result.response_b[:240]}{'...' if len(result.response_b) > 240 else ''}"
    print(f"    {response_b_text}")
    print()

    for name, (passed, msg) in checks.items():
        if passed is True:
            print(f"  {ok(name)}: {msg}")
        elif passed is None:
            print(f"  {warn(name)}: {msg}")
        else:
            print(f"  {fail(name)}: {msg}")

    if soft_checks:
        print()
        for name, data in soft_checks.items():
            msg = f"expected={data['expected']} actual={data['actual']}"
            if data["match"]:
                print(f"  {soft_pass(name)}: {msg}")
            else:
                print(f"  {soft_mismatch(name)}: {msg}")

    overall = all(value[0] is not False for value in checks.values())
    print(f"\n  {'PASS' if overall else 'FAIL'} - {elapsed}s")

    return {
        "id": tid,
        "category": case["category"],
        "description": case["description"],
        "passed": overall,
        "elapsed": elapsed,
        "retries": result.retries,
        "confidence": result.confidence,
        "conflict_type": result.conflict_type,
        "resolvability": result.resolvability,
        "fallback_used": trace.fallback_used if trace else False,
        "safety_score": trace.safety_score if trace else None,
        "rag_used": trace.rag_used if trace else False,
        "response_a_words": len(result.response_a.split()),
        "response_b_words": len(result.response_b.split()),
        "one_line_summary": result.one_line_summary,
        "response_a": result.response_a,
        "response_b": result.response_b,
        "checks": {name: {"passed": value[0], "msg": value[1]} for name, value in checks.items()},
        "expected_behavior": expected_soft,
        "soft_checks": soft_checks,
        "error": None,
    }


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(hdr(f"{'EVALUATION SUMMARY':^72}"))
    print("=" * 72)

    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    failed = total - passed
    errors = sum(1 for result in results if result.get("error"))
    avg_time = sum(result["elapsed"] for result in results) / total if total else 0
    avg_retries = sum(result.get("retries", 0) for result in results) / total if total else 0
    fallbacks = sum(1 for result in results if result.get("fallback_used"))

    print(f"  Total tests   : {total}")
    print(f"  Passed        : {GREEN}{passed}{RESET}")
    print(f"  Failed        : {RED}{failed}{RESET}")
    print(f"  Errors        : {RED}{errors}{RESET}")
    print(f"  Avg latency   : {avg_time:.1f}s")
    print(f"  Avg retries   : {avg_retries:.2f}")
    print(f"  Fallbacks     : {fallbacks}")

    check_totals: dict[str, list[bool | None]] = {}
    for result in results:
        for name, data in result.get("checks", {}).items():
            check_totals.setdefault(name, []).append(data["passed"])

    print(f"\n  {'Check':<25} {'Pass rate':>10}")
    print(f"  {'-' * 35}")
    for name, values in check_totals.items():
        true_count = sum(1 for value in values if value is True)
        warn_count = sum(1 for value in values if value is None)
        score = (true_count + 0.5 * warn_count) / len(values) if values else 0
        bar = "#" * int(score * 10) + "." * (10 - int(score * 10))
        color = GREEN if score >= 0.8 else (YELLOW if score >= 0.6 else RED)
        print(f"  {name:<25} {color}{bar}{RESET} {score:.0%}")

    soft_totals: dict[str, list[bool]] = {}
    for result in results:
        for name, data in result.get("soft_checks", {}).items():
            soft_totals.setdefault(name, []).append(bool(data["match"]))
    if soft_totals:
        soft_matches = sum(1 for values in soft_totals.values() for value in values if value)
        soft_mismatches = sum(1 for values in soft_totals.values() for value in values if not value)
        print(f"\n  {'Soft check':<25} {'Match rate':>10}")
        print(f"  {'-' * 38}")
        for name, values in soft_totals.items():
            match_count = sum(1 for value in values if value)
            score = match_count / len(values) if values else 0
            bar = "#" * int(score * 10) + "." * (10 - int(score * 10))
            color = GREEN if score >= 0.8 else (YELLOW if score >= 0.6 else RED)
            print(f"  {name:<25} {color}{bar}{RESET} {score:.0%}")
        print(f"\n  Soft matches      : {GREEN}{soft_matches}{RESET}")
        print(f"  Soft mismatches   : {YELLOW}{soft_mismatches}{RESET}")

    categories: dict[str, dict[str, int]] = {}
    for result in results:
        category = result["category"]
        categories.setdefault(category, {"pass": 0, "total": 0})
        categories[category]["total"] += 1
        if result["passed"]:
            categories[category]["pass"] += 1

    print(f"\n  {'Category':<20} {'Pass/Total':>12}")
    print(f"  {'-' * 32}")
    for category, data in sorted(categories.items()):
        color = GREEN if data["pass"] == data["total"] else (YELLOW if data["pass"] else RED)
        print(f"  {category:<20} {color}{data['pass']}/{data['total']}{RESET}")

    failed_tests = [result for result in results if not result["passed"]]
    if failed_tests:
        print(f"\n  {RED}Failed tests:{RESET}")
        for result in failed_tests:
            failed_checks = [
                name for name, value in result.get("checks", {}).items() if value["passed"] is False
            ]
            print(f"    {result['id']} - {result['description']}")
            print(f"      Failed: {', '.join(failed_checks) or result.get('error', 'unknown')}")
    print("=" * 72)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backend evaluation runner")
    parser.add_argument("--file", default="test_cases.json", help="Path to test cases JSON")
    parser.add_argument("--ids", nargs="+", help="Run specific test IDs only")
    parser.add_argument("--category", help="Run only one category")
    parser.add_argument("--mode", choices=["fast", "quality", "fast_production"], help="Override mode for all cases")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip default warmup. By default, local models are preloaded before timed tests.",
    )
    parser.add_argument("--save", help="Save results to JSON")
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Print full user-facing responses instead of 240-character snippets.",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[error] Test file not found: {path}")
        sys.exit(1)

    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if args.ids:
        wanted = set(args.ids)
        cases = [case for case in cases if case["id"] in wanted]
    if args.category:
        cases = [case for case in cases if case["category"] == args.category]
    if not cases:
        print("[error] No matching test cases found.")
        sys.exit(1)

    print(hdr("\nAI Conflict Mediation - Backend Evaluation"))
    print(f"Test file : {path}")
    print(f"Running   : {len(cases)} test(s)")
    print(f"Filter    : ids={args.ids} category={args.category}")

    print("\n[setup] Seeding RAG database...")
    await asyncio.to_thread(seed_from_scenarios)
    if not args.no_warmup:
        effective_mode = args.mode or "fast"
        print(f"[setup] Warming local models for {effective_mode} mode...")
        if effective_mode == "quality":
            await asyncio.gather(validator.warmup(), safety_agent.warmup())
        else:
            await safety_agent.warmup(use_toxic_bert=False)
    print("[setup] Ready.\n")

    results = []
    for idx, case in enumerate(cases, 1):
        results.append(await run_test(case, idx, len(cases), args.mode, full_output=args.full_output))

    print_summary(results)

    if args.save:
        out = Path(args.save)
        with out.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
        print(f"\n[saved] Results written to {out}")


if __name__ == "__main__":
    asyncio.run(main())
