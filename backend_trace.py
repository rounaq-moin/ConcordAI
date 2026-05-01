"""Standalone verbose backend trace runner.

This file is intentionally NOT integrated into the FastAPI app or LangGraph.
Run it only when you want a transparent, step-by-step debug log of the backend
agents and tools.

Examples:
    python backend_trace.py
    python backend_trace.py --preset credit
    python backend_trace.py --text-a "You ignored me" --text-b "I was busy"
    python backend_trace.py --case T02
    python backend_trace.py --case-id T02
    python backend_trace.py --warmup --case T01
    python backend_trace.py --mode quality --show-raw
    python backend_trace.py --store-memory --case T09
    python backend_trace.py --save trace.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

IMPORT_ERROR: Exception | None = None

try:
    from agents import (
        critic_agent,
        production_agent,
        reasoning_agent,
        reconciliation_agent,
        retriever,
        safety_agent,
        validator,
    )
    from config import (
        CHROMA_PERSIST_DIR,
        DETOXIFY_THRESHOLD,
        EMBEDDING_MODEL,
        GROQ_MODEL,
        GROQ_MODEL_SEQUENCE,
        GROQ_TIMEOUT,
        PERSPECTIVE_MODEL,
        RAG_MIN_SCORE,
        RAG_TOP_K,
        TOXIC_BERT_MODEL,
        TOXIC_BERT_THRESHOLD,
        EMOTION_MODEL,
        get_groq_api_key,
        secret_fingerprint,
    )
    from models.schemas import ReflectionOutput, SafetyResult
    from utils import llm as llm_utils
    from utils.scoring import confidence_score, fairness_score
    from utils.text import sanitize_text
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc


PRESETS = {
    "credit": {
        "text_a": "You took credit for my idea in the meeting without mentioning my name.",
        "text_b": "I built on your idea significantly. The final version was mostly my work.",
    },
    "relationship": {
        "text_a": "You never make time for me anymore. I feel like I'm the last priority in your life.",
        "text_b": "I've been working extra hours to pay our bills. I thought you'd appreciate that, not resent me for it.",
    },
    "logical": {
        "text_a": "The data clearly shows our marketing strategy is failing. We need to pivot immediately.",
        "text_b": "The data is incomplete. We've only run this campaign for three weeks and you want to scrap everything.",
    },
    "misunderstanding": {
        "text_a": "You missed the client meeting. We agreed on 2pm and you just didn't show up.",
        "text_b": "I had it written down as 3pm. Nobody confirmed the change with me.",
    },
    "value": {
        "text_a": "I don't want our children raised with religious beliefs. Let them decide when they're adults.",
        "text_b": "Faith is central to who I am. Raising our children without it would feel like erasing part of our family.",
    },
}


class TraceLog:
    def __init__(self, *, show_raw: bool = False):
        self.show_raw = show_raw
        self.events: list[dict[str, Any]] = []
        self.started = time.time()

    def line(self, text: str = "") -> None:
        print(text)

    def section(self, title: str) -> None:
        self.line("\n" + "=" * 88)
        self.line(title)
        self.line("=" * 88)

    def step(self, title: str) -> None:
        self.line("\n" + "-" * 88)
        self.line(title)
        self.line("-" * 88)

    def event(self, name: str, payload: Any, *, raw: bool = False) -> None:
        self.events.append({"name": name, "payload": _jsonable(payload), "raw": raw})
        if raw and not self.show_raw:
            self.line(f"[{name}] <hidden raw output; use --show-raw>")
            return
        self.line(f"[{name}]")
        self.line(_pretty(payload))

    @contextmanager
    def timer(self, name: str):
        started = time.time()
        try:
            yield
        finally:
            elapsed = round(time.time() - started, 3)
            self.events.append({"name": f"{name}.timing", "payload": {"seconds": elapsed}})
            self.line(f"[timing] {name}: {elapsed}s")

    def save(self, path: str | None) -> None:
        if not path:
            return
        payload = {
            "elapsed_seconds": round(time.time() - self.started, 3),
            "events": self.events,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.line(f"\n[trace] saved to {path}")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _pretty(value: Any) -> str:
    return json.dumps(_jsonable(value), indent=2, ensure_ascii=False, default=str)


def _load_case(case_id: str) -> dict[str, str]:
    cases_path = Path("test_cases.json")
    if not cases_path.exists():
        raise FileNotFoundError("test_cases.json not found. Run from project root.")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    for case in cases:
        if case["id"] == case_id:
            return {"text_a": case["text_a"], "text_b": case["text_b"]}
    raise ValueError(f"Case id not found: {case_id}")


def _component_manifest() -> dict[str, Any]:
    return {
        "execution_model": {
            "runner": "backend_trace.py standalone manual trace runner",
            "integrated_into_api": False,
            "uses_langgraph": False,
            "note": (
                "This file does not change FastAPI, React, or coordinator/graph.py. "
                "The production app still uses LangGraph; this runner calls each backend "
                "agent directly so every input/output can be printed."
            ),
        },
        "tools_used": [
            {
                "tool": "ChromaDB PersistentClient",
                "purpose": "RAG vector store for similar conflict cases",
                "path": str(CHROMA_PERSIST_DIR),
            },
            {
                "tool": "SentenceTransformer embedding function",
                "model": EMBEDDING_MODEL,
                "purpose": "Embeds conflict text for retrieval",
            },
            {
                "tool": "transformers zero-shot-classification pipeline",
                "model": PERSPECTIVE_MODEL,
                "purpose": "Perspective / stance and communicative intent detection",
            },
            {
                "tool": "transformers text-classification pipeline",
                "model": EMOTION_MODEL,
                "purpose": "Emotion and sentiment detection",
            },
            {
                "tool": "Groq chat.completions",
                "model": GROQ_MODEL,
                "model_sequence": GROQ_MODEL_SEQUENCE,
                "purpose": "Reasoning, reconciliation, production, and critic agents",
                "timeout_seconds": GROQ_TIMEOUT,
            },
            {
                "tool": "Detoxify",
                "model": "original",
                "purpose": "Safety toxicity scoring",
                "threshold": DETOXIFY_THRESHOLD,
            },
            {
                "tool": "transformers text-classification pipeline",
                "model": TOXIC_BERT_MODEL,
                "purpose": "Second safety toxicity score in quality mode",
                "threshold": TOXIC_BERT_THRESHOLD,
            },
        ],
        "rag": {
            "top_k": RAG_TOP_K,
            "min_score": RAG_MIN_SCORE,
        },
        "groq_key": secret_fingerprint(get_groq_api_key()),
    }


@contextmanager
def patched_groq_logging(trace: TraceLog):
    """Patch agent-local Groq bindings to show raw/parsed LLM output."""

    originals: list[tuple[object, str, Callable]] = []

    async def traced_call_groq_json(
        messages: list[dict[str, str]],
        *,
        source: str,
        max_tokens: int,
        temperature: float | None = None,
        timeout: float = GROQ_TIMEOUT,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        trace.event(
            f"{source}.groq_request",
            {
                "model": GROQ_MODEL,
                "model_sequence": GROQ_MODEL_SEQUENCE,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout": timeout,
                "max_attempts": max_attempts,
                "messages": messages,
            },
        )
        with trace.timer(f"{source}.groq_call"):
            raw = await llm_utils.call_groq(
                messages,
                source=source,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                max_attempts=max_attempts,
            )
        trace.event(f"{source}.groq_raw_response", raw, raw=True)
        parsed = llm_utils.parse_json_object(raw, source)
        trace.event(f"{source}.groq_parsed_json", parsed)
        return parsed

    async def traced_call_groq_production(
        messages: list[dict[str, str]],
        *,
        source: str,
        max_tokens: int,
        temperature: float | None = None,
        attempt_timeouts: tuple[float, ...] = (6.0, 4.0),
        backoff: float = 2.0,
    ) -> dict[str, Any]:
        trace.event(
            f"{source}.groq_request",
            {
                "model": GROQ_MODEL,
                "model_sequence": GROQ_MODEL_SEQUENCE,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "attempt_timeouts": attempt_timeouts,
                "backoff": backoff,
                "messages": messages,
            },
        )
        with trace.timer(f"{source}.groq_call"):
            raw = await llm_utils._call_groq_production_raw(
                messages,
                source=source,
                max_tokens=max_tokens,
                temperature=temperature,
                attempt_timeouts=attempt_timeouts,
                backoff=backoff,
            )
        trace.event(f"{source}.groq_raw_response", raw, raw=True)
        parsed = llm_utils.parse_json_object(raw, source)
        trace.event(f"{source}.groq_parsed_json", parsed)
        return parsed

    for module in (reasoning_agent, reconciliation_agent, production_agent, critic_agent):
        if hasattr(module, "call_groq_json"):
            originals.append((module, "call_groq_json", getattr(module, "call_groq_json")))
            setattr(module, "call_groq_json", traced_call_groq_json)
        if hasattr(module, "call_groq_production"):
            originals.append((module, "call_groq_production", getattr(module, "call_groq_production")))
            setattr(module, "call_groq_production", traced_call_groq_production)

    try:
        yield
    finally:
        for module, attr, original in originals:
            setattr(module, attr, original)


async def run_trace(args: argparse.Namespace) -> None:
    if IMPORT_ERROR is not None:
        raise SystemExit(
            "[backend_trace] Project dependencies are not available in this Python environment.\n"
            "Run this with the same environment you use for the backend, for example:\n"
            "  .\\.venv\\Scripts\\python backend_trace.py --case T01\n"
            "  or activate your environment before running python backend_trace.py\n"
            f"Original import error: {IMPORT_ERROR}"
        )

    trace = TraceLog(show_raw=args.show_raw)
    text_a, text_b = _resolve_inputs(args)
    text_a = sanitize_text(text_a)
    text_b = sanitize_text(text_b)
    trace_id = args.trace_id or f"trace-{uuid.uuid4()}"

    trace.section("AI Conflict Mediation Backend Trace")
    trace.event(
        "run_metadata",
        {
            "trace_id": trace_id,
            "mode": args.mode,
            "show_raw": args.show_raw,
            "store_memory": args.store_memory,
        },
    )
    trace.event("component_manifest", _component_manifest())
    trace.event(
        "isolation_guarantee",
        {
            "current_system_disturbed": False,
            "memory_write_default": "skipped",
            "memory_write_requires": "--store-memory",
            "raw_llm_output_default": "hidden",
            "raw_llm_output_requires": "--show-raw",
        },
    )
    trace.event("user_input", {"text_a": text_a, "text_b": text_b})

    with patched_groq_logging(trace):
        if args.warmup:
            trace.step("Warmup. Optional Model Preload")
            trace.event(
                "warmup.input",
                {
                    "validator_models": [PERSPECTIVE_MODEL, EMOTION_MODEL],
                    "safety_models": ["Detoxify", "toxic-bert" if args.mode == "quality" else "toxic-bert skipped"],
                    "purpose": "Preload local CPU models before the timed agent trace.",
                },
            )
            with trace.timer("validator.warmup"):
                validator_status = await validator.warmup()
            with trace.timer("safety_agent.warmup"):
                safety_status = await safety_agent.warmup(use_toxic_bert=args.mode == "quality")
            trace.event("warmup.output", {"validator": validator_status, "safety": safety_status})
        else:
            trace.event(
                "warmup.skipped",
                {
                    "reason": "No --warmup flag was provided.",
                    "impact": "First run may include local model load time in agent timings.",
                },
            )

        trace.step("0. Seed RAG Store")
        with trace.timer("retriever.seed_from_scenarios"):
            seeded = await asyncio.to_thread(retriever.seed_from_scenarios)
        trace.event("retriever.seed_result", {"new_records": seeded})

        trace.step("1. Retriever Agent")
        trace.event(
            "retriever.input",
            {
                "text_a": text_a,
                "text_b": text_b,
                "tool": "ChromaDB + MiniLM embeddings",
            },
        )
        with trace.timer("retriever.retrieve"):
            rag = await asyncio.to_thread(retriever.retrieve, text_a, text_b)
        trace.event("retriever.output", rag)

        if args.mode == "fast_production":
            trace.step("2. Production Agent. Rule Reasoning + Single Groq Call")
            intent = validator.infer_intents(
                text_a,
                text_b,
                feedback="Intent inferred deterministically in fast_production trace mode.",
            )
            trace.event(
                "production.input",
                {
                    "text_a": text_a,
                    "text_b": text_b,
                    "retrieved_cases": rag.retrieved_cases,
                    "prior_context": None,
                    "intent": intent,
                    "tools": [
                        "rule-first conflict classifier",
                        "rule-first resolvability classifier",
                        "deterministic communicative intent inference",
                        "Groq chat.completions single call",
                    ],
                },
            )
            with trace.timer("production_agent.run"):
                llm_output = await production_agent.run(
                    text_a=text_a,
                    text_b=text_b,
                    retrieved_cases=rag.retrieved_cases,
                    prior_context=None,
                    intent_a=intent.user_a_intent,
                    intent_b=intent.user_b_intent,
                )
            reasoning = llm_output.reasoning
            trace.event("production.output", llm_output)
            trace.event(
                "production.metrics",
                {
                    "fairness_score": fairness_score(llm_output.response_a, llm_output.response_b),
                    "response_a_words": len(llm_output.response_a.split()),
                    "response_b_words": len(llm_output.response_b.split()),
                    "verification": llm_output.verification,
                    "reasoning_state": llm_output.reasoning_state,
                    "llm_calls": (
                        llm_output.verification.get("llm_calls", 1)
                        if isinstance(llm_output.verification, dict)
                        else 1
                    ),
                },
            )

            trace.step("3. Safety Agent")
            trace.event(
                "safety.input",
                {
                    "response_a": llm_output.response_a,
                    "response_b": llm_output.response_b,
                    "tools": ["Detoxify", "toxic-bert skipped"],
                },
            )
            with trace.timer("safety_agent.check"):
                safety = await safety_agent.check(
                    llm_output.response_a,
                    llm_output.response_b,
                    use_toxic_bert=False,
                )
            trace.event("safety.output", safety)

            trace.step("4. Output Validator")
            output_validation = {
                "skipped": True,
                "reason": "fast_production mode uses safety plus deterministic critic pre-check only.",
                "safety_approved": safety.approved,
            }
            trace.event("output_validator.skipped", output_validation)

            trace.step("5. Critic Agent")
            basic_failure = critic_agent.basic_response_check(llm_output.response_a, llm_output.response_b)
            if basic_failure:
                critic = basic_failure
                trace.event("critic.basic_check_failed", critic)
            elif isinstance(llm_output.verification, dict) and not llm_output.verification.get("passed", True):
                critic = ReflectionOutput(
                    approved=False,
                    is_fair=False,
                    is_empathetic=False,
                    is_unbiased=True,
                    is_specific=False,
                    is_non_escalatory=True,
                    skipped=True,
                    reason="; ".join(str(issue) for issue in llm_output.verification.get("issues", [])),
                    suggestion="Production verification failed after repair.",
                )
                trace.event("critic.production_verification_failed", critic)
            else:
                critic = ReflectionOutput(
                    approved=True,
                    is_fair=True,
                    is_empathetic=True,
                    is_unbiased=True,
                    skipped=True,
                    reason="Skipped in fast_production trace mode after deterministic checks.",
                )
                trace.event("critic.skipped", critic)

            trace.step("6. Final Assembly")
            safety_score = max(safety.detoxify_score, safety.toxic_bert_score)
            confidence = confidence_score(
                critic_approved=critic.approved,
                retries=0,
                safety_score=safety_score,
                fallback_used=False,
                critic_skipped=critic.skipped,
                production_mode=True,
            )
            final = {
                "trace_id": trace_id,
                "response_a": llm_output.response_a,
                "response_b": llm_output.response_b,
                "conversation_status": llm_output.conversation_status,
                "conflict_type": reasoning.conflict_type,
                "resolvability": reasoning.resolvability,
                "one_line_summary": reasoning.one_line_summary,
                "confidence": confidence,
                "fairness_score": fairness_score(llm_output.response_a, llm_output.response_b),
                "safety_score": safety_score,
                "critic_approved": critic.approved,
                "rag_used": rag.rag_used,
                "retrieved_cases": len(rag.retrieved_cases),
                "llm_calls": (
                    llm_output.verification.get("llm_calls", 1)
                    if isinstance(llm_output.verification, dict)
                    else 1
                ),
                "production_checks": llm_output.verification,
                "reasoning_state": llm_output.reasoning_state,
            }
            trace.event("final.output", final)

            trace.step("7. Memory Write Decision")
            if args.store_memory and safety.approved and critic.approved:
                with trace.timer("retriever.store_case"):
                    stored = await asyncio.to_thread(
                        retriever.store_case,
                        conversation_id=trace_id,
                        text_a=text_a,
                        text_b=text_b,
                        conflict_type=reasoning.conflict_type,
                        resolution_strategy=reasoning.resolution_strategy,
                        response_a=llm_output.response_a,
                        response_b=llm_output.response_b,
                    )
                trace.event("memory.output", {"stored": stored})
            else:
                reasons = []
                if not args.store_memory:
                    reasons.append("read-only logging mode; pass --store-memory to write")
                if not safety.approved:
                    reasons.append("safety agent did not approve")
                if not critic.approved:
                    reasons.append("critic agent did not approve")
                trace.event(
                    "memory.skipped",
                    {
                        "stored": False,
                        "reasons": reasons,
                        "safe_to_rerun": True,
                    },
                )

            trace.save(args.save)
            return

        trace.step("2. Perspective + Emotion + Intent Validator")
        if args.mode == "fast" and args.skip_local_validation:
            stance = _neutral_stance()
            emotion = _neutral_emotion()
            intent = _neutral_intent()
            trace.event(
                "validator.skipped",
                {
                    "reason": "fast mode with --skip-local-validation enabled",
                    "stance": stance,
                    "emotion": emotion,
                    "intent": intent,
                },
            )
        else:
            neutral_safety = SafetyResult(approved=True)
            trace.event(
                "validator.input",
                {
                    "text_a": text_a,
                    "text_b": text_b,
                    "response_a": "",
                    "response_b": "",
                    "safety_result": neutral_safety,
                    "tools": [PERSPECTIVE_MODEL, EMOTION_MODEL],
                },
            )
            with trace.timer("validator.validate_input"):
                validation_input = await validator.validate(
                    text_a=text_a,
                    text_b=text_b,
                    response_a="",
                    response_b="",
                    safety_result=neutral_safety,
                )
            trace.event("validator.output", validation_input)
            stance = validation_input.stance
            emotion = validation_input.emotion
            intent = validation_input.intent

        trace.step("3. Reasoning Agent")
        trace.event(
            "reasoning.input",
            {
                "text_a": text_a,
                "text_b": text_b,
                "stance_a": stance.user_a_stance,
                "stance_b": stance.user_b_stance,
                "emotion_a": emotion.user_a_emotion,
                "emotion_b": emotion.user_b_emotion,
                "sentiment_a": emotion.user_a_sentiment,
                "sentiment_b": emotion.user_b_sentiment,
                "intent_a": intent.user_a_intent,
                "intent_b": intent.user_b_intent,
            },
        )
        with trace.timer("reasoning_agent.run"):
            reasoning = await reasoning_agent.run(
                text_a=text_a,
                text_b=text_b,
                stance_a=stance.user_a_stance,
                stance_b=stance.user_b_stance,
                emotion_a=emotion.user_a_emotion,
                emotion_b=emotion.user_b_emotion,
                sentiment_a=emotion.user_a_sentiment,
                sentiment_b=emotion.user_b_sentiment,
                intent_a=intent.user_a_intent,
                intent_b=intent.user_b_intent,
                prior_context=None,
            )
        trace.event("reasoning.output", reasoning)

        trace.step("4. Reconciliation Agent")
        trace.event(
            "reconciliation.input",
            {
                "text_a": text_a,
                "text_b": text_b,
                "reasoning": reasoning,
                "retrieved_cases": rag.retrieved_cases,
                "feedback": None,
            },
        )
        with trace.timer("reconciliation_agent.run"):
            llm_output = await reconciliation_agent.run(
                text_a=text_a,
                text_b=text_b,
                reasoning=reasoning,
                retrieved_cases=rag.retrieved_cases,
                feedback=None,
            )
        trace.event("reconciliation.output", llm_output)
        trace.event(
            "reconciliation.metrics",
            {
                "fairness_score": fairness_score(llm_output.response_a, llm_output.response_b),
                "response_a_words": len(llm_output.response_a.split()),
                "response_b_words": len(llm_output.response_b.split()),
            },
        )

        trace.step("5. Safety Agent")
        trace.event(
            "safety.input",
            {
                "response_a": llm_output.response_a,
                "response_b": llm_output.response_b,
                "tools": ["Detoxify", "toxic-bert" if args.mode == "quality" else "toxic-bert skipped"],
            },
        )
        with trace.timer("safety_agent.check"):
            safety = await safety_agent.check(
                llm_output.response_a,
                llm_output.response_b,
                use_toxic_bert=args.mode == "quality",
            )
        trace.event("safety.output", safety)

        trace.step("6. Output Validator")
        if args.mode == "fast" and args.skip_output_validation:
            output_validation = {
                "skipped": True,
                "reason": "fast mode with --skip-output-validation enabled",
                "safety_approved": safety.approved,
            }
            trace.event("output_validator.skipped", output_validation)
        else:
            trace.event(
                "output_validator.input",
                {
                    "text_a": text_a,
                    "text_b": text_b,
                    "response_a": llm_output.response_a,
                    "response_b": llm_output.response_b,
                    "safety_result": safety,
                },
            )
            with trace.timer("validator.validate_output"):
                output_validation = await validator.validate(
                    text_a=text_a,
                    text_b=text_b,
                    response_a=llm_output.response_a,
                    response_b=llm_output.response_b,
                    safety_result=safety,
                )
            trace.event("output_validator.output", output_validation)

        trace.step("7. Critic Agent")
        basic_failure = critic_agent.basic_response_check(llm_output.response_a, llm_output.response_b)
        if basic_failure:
            critic = basic_failure
            trace.event("critic.basic_check_failed", critic)
        elif args.mode == "fast" and args.skip_critic:
            critic = ReflectionOutput(
                approved=True,
                is_fair=True,
                is_empathetic=True,
                is_unbiased=True,
                skipped=True,
                reason="Skipped in fast trace mode.",
            )
            trace.event("critic.skipped", critic)
        else:
            trace.event(
                "critic.input",
                {
                    "text_a": text_a,
                    "text_b": text_b,
                    "response_a": llm_output.response_a,
                    "response_b": llm_output.response_b,
                    "conflict_type": reasoning.conflict_type,
                    "resolvability": reasoning.resolvability,
                },
            )
            with trace.timer("critic_agent.critique"):
                critic = await critic_agent.critique(
                    text_a=text_a,
                    text_b=text_b,
                    response_a=llm_output.response_a,
                    response_b=llm_output.response_b,
                    conflict_type=reasoning.conflict_type,
                    resolvability=reasoning.resolvability,
                )
            trace.event("critic.output", critic)

        trace.step("8. Final Assembly")
        safety_score = max(safety.detoxify_score, safety.toxic_bert_score)
        confidence = confidence_score(
            critic_approved=critic.approved,
            retries=0,
            safety_score=safety_score,
            fallback_used=False,
        )
        final = {
            "trace_id": trace_id,
            "response_a": llm_output.response_a,
            "response_b": llm_output.response_b,
            "conversation_status": llm_output.conversation_status,
            "conflict_type": reasoning.conflict_type,
            "resolvability": reasoning.resolvability,
            "one_line_summary": reasoning.one_line_summary,
            "confidence": confidence,
            "fairness_score": fairness_score(llm_output.response_a, llm_output.response_b),
            "safety_score": safety_score,
            "critic_approved": critic.approved,
            "rag_used": rag.rag_used,
            "retrieved_cases": len(rag.retrieved_cases),
        }
        trace.event("final.output", final)

        trace.step("9. Memory Write Decision")
        if args.store_memory and safety.approved and critic.approved:
            with trace.timer("retriever.store_case"):
                stored = await asyncio.to_thread(
                    retriever.store_case,
                    conversation_id=trace_id,
                    text_a=text_a,
                    text_b=text_b,
                    conflict_type=reasoning.conflict_type,
                    resolution_strategy=reasoning.resolution_strategy,
                    response_a=llm_output.response_a,
                    response_b=llm_output.response_b,
                )
            trace.event("memory.output", {"stored": stored})
        else:
            reasons = []
            if not args.store_memory:
                reasons.append("read-only logging mode; pass --store-memory to write")
            if not safety.approved:
                reasons.append("safety agent did not approve")
            if not critic.approved:
                reasons.append("critic agent did not approve")
            trace.event(
                "memory.skipped",
                {
                    "stored": False,
                    "reasons": reasons,
                    "safe_to_rerun": True,
                },
            )

    trace.save(args.save)


def _neutral_stance():
    from models.schemas import StanceResult

    return StanceResult(user_a_stance="neutral", user_b_stance="neutral", response_consistent=True)


def _neutral_emotion():
    from models.schemas import EmotionResult

    return EmotionResult(
        user_a_emotion="neutral",
        user_b_emotion="neutral",
        user_a_sentiment="neutral",
        user_b_sentiment="neutral",
        response_empathetic=True,
    )


def _neutral_intent():
    from models.schemas import IntentResult

    return IntentResult(user_a_intent="unknown", user_b_intent="unknown", confidence_a=0.0, confidence_b=0.0)


def _resolve_inputs(args: argparse.Namespace) -> tuple[str, str]:
    if args.case_id:
        case = _load_case(args.case_id)
        return case["text_a"], case["text_b"]
    if args.text_a or args.text_b:
        if not args.text_a or not args.text_b:
            raise ValueError("Both --text-a and --text-b are required for custom input.")
        return args.text_a, args.text_b
    preset = PRESETS.get(args.preset)
    if not preset:
        raise ValueError(f"Unknown preset: {args.preset}. Choices: {', '.join(PRESETS)}")
    return preset["text_a"], preset["text_b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone transparent backend trace runner")
    parser.add_argument("--preset", default="credit", choices=sorted(PRESETS), help="Built-in input preset")
    parser.add_argument("--case-id", "--case", dest="case_id", help="Use a case from test_cases.json, e.g. T02")
    parser.add_argument("--text-a", help="Custom User A input")
    parser.add_argument("--text-b", help="Custom User B input")
    parser.add_argument("--mode", choices=["fast", "quality", "fast_production"], default="quality", help="Trace mode")
    parser.add_argument("--show-raw", action="store_true", help="Print raw Groq responses")
    parser.add_argument("--store-memory", action="store_true", help="Write the final case to Chroma memory")
    parser.add_argument("--warmup", action="store_true", help="Preload local validator and safety models before tracing")
    parser.add_argument("--save", help="Save full trace events to JSON")
    parser.add_argument("--trace-id", help="Custom trace id")
    parser.add_argument(
        "--skip-local-validation",
        action="store_true",
        help="Skip input validator in fast mode. Default trace shows validators unless this is set.",
    )
    parser.add_argument(
        "--skip-output-validation",
        action="store_true",
        help="Skip output validator in fast mode.",
    )
    parser.add_argument(
        "--skip-critic",
        action="store_true",
        help="Skip LLM critic in fast mode after deterministic checks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_trace(parse_args()))
