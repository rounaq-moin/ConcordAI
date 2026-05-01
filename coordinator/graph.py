"""LangGraph multi-agent orchestrator."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents import (
    critic_agent,
    production_agent,
    reasoning_agent,
    reconciliation_agent,
    retriever,
    safety_agent,
    validator,
)
from config import CONTEXT_TURNS, MAX_RETRIES
from models.schemas import (
    AgentTrace,
    EmotionResult,
    FinalOutput,
    IntentResult,
    LLMCoreOutput,
    Reasoning,
    ReflectionOutput,
    RetrieverOutput,
    SafetyResult,
    StanceResult,
    UserInput,
    ValidationOutput,
)
from utils.observability import record_request
from utils.scoring import confidence_score, fairness_score


_CONTEXT: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=CONTEXT_TURNS))
_COMPILED_GRAPH = None


class PipelineState(TypedDict):
    user_input: UserInput
    retriever_output: Optional[RetrieverOutput]
    stance_result: Optional[StanceResult]
    emotion_result: Optional[EmotionResult]
    intent_result: Optional[IntentResult]
    reasoning_output: Optional[Reasoning]
    llm_output: Optional[LLMCoreOutput]
    safety_result: Optional[SafetyResult]
    validation_output: Optional[ValidationOutput]
    critic_output: Optional[ReflectionOutput]
    retry_count: int
    feedback: Optional[str]
    final_output: Optional[FinalOutput]
    stored_to_memory: bool
    fallback_used: bool
    start_time: float


def _prior_context(conversation_id: str) -> str | None:
    turns = list(_CONTEXT.get(conversation_id, []))
    return "\n".join(turns[-CONTEXT_TURNS:]) if turns else None


def _remember(final: FinalOutput) -> None:
    summary = final.one_line_summary or "No summary available"
    _CONTEXT[final.conversation_id].append(
        f"Turn {final.turn}: status={final.conversation_status}; "
        f"type={final.conflict_type}; resolvability={final.resolvability}; summary={summary}"
    )


def _dummy_reasoning() -> Reasoning:
    return Reasoning(
        user_a_goal="to feel heard and treated fairly",
        user_b_goal="to be understood without being blamed",
        conflict_type="emotional",
        resolvability="partially_resolvable",
        common_ground="both users want the interaction to become less tense",
        resolution_strategy="validate both perspectives and slow the conversation down",
        one_line_summary="Both users need to feel heard before the conflict can move forward.",
    )


def _fallback_reasoning_from_input(inp: UserInput) -> Reasoning:
    conflict_type = reasoning_agent.classify_conflict_type(inp.text_a, inp.text_b) or "emotional"
    resolvability = (
        reasoning_agent.classify_resolvability(conflict_type, inp.text_a, inp.text_b)
        or "partially_resolvable"
    )
    summaries = {
        "logical": "Both users are weighing evidence, risk, and timing differently.",
        "misunderstanding": "Both users may be working from different assumptions about what happened or what was understood.",
        "value": "Both users are protecting different values that need to be respected without forcing agreement.",
        "emotional": "Both users need the emotional impact and the intent behind the behavior to be heard.",
    }
    strategies = {
        "logical": "separate facts from urgency, define missing evidence, and agree on a concrete checkpoint",
        "misunderstanding": "clarify what each person believed happened, acknowledge impact, and set a confirmation rule",
        "value": "name the values on both sides and move toward boundaries or coexistence rather than forced agreement",
        "emotional": "validate the impact first, then acknowledge intent and ask for one concrete repair",
    }
    return Reasoning(
        user_a_goal="to have their concern taken seriously",
        user_b_goal="to have their intent or constraint understood",
        conflict_type=conflict_type,
        resolvability=resolvability,
        common_ground="both users want the conflict handled more constructively",
        resolution_strategy=strategies.get(conflict_type, strategies["emotional"]),
        one_line_summary=summaries.get(conflict_type, summaries["emotional"]),
    )


def _dynamic_fallback_output(inp: UserInput, reasoning: Reasoning | None = None) -> LLMCoreOutput:
    reason = reasoning or _fallback_reasoning_from_input(inp)
    response_a, response_b = production_agent.build_fallback_responses(
        reason.conflict_type,
        reason.resolvability,
        text_a=inp.text_a,
        text_b=inp.text_b,
    )
    return LLMCoreOutput(
        reasoning=reason,
        response_a=response_a,
        response_b=response_b,
        conversation_status="Stable",
    )


def _fallback_output(reasoning: Reasoning | None = None, inp: UserInput | None = None) -> LLMCoreOutput:
    if inp is not None:
        return _dynamic_fallback_output(inp, reasoning)
    reason = reasoning or _dummy_reasoning()
    response_a, response_b = production_agent.build_fallback_responses(
        reason.conflict_type,
        reason.resolvability,
    )
    return LLMCoreOutput(
        reasoning=reason,
        response_a=response_a,
        response_b=response_b,
        conversation_status="Stable",
    )


def _safety_score(safety: SafetyResult | None) -> float:
    if not safety:
        return 0.0
    return max(safety.detoxify_score, safety.toxic_bert_score)


def _build_trace(state: PipelineState) -> AgentTrace:
    inp = state["user_input"]
    stance = state.get("stance_result")
    emotion = state.get("emotion_result")
    intent = state.get("intent_result")
    reason = state.get("reasoning_output")
    out = state.get("llm_output")
    critic = state.get("critic_output")
    safety = state.get("safety_result")
    retr = state.get("retriever_output")
    elapsed = round(time.time() - state["start_time"], 3)

    return AgentTrace(
        request_id=inp.request_id,
        trace_id=inp.trace_id,
        mode=inp.mode,
        stance_a=stance.user_a_stance if stance else None,
        stance_b=stance.user_b_stance if stance else None,
        emotion_a=emotion.user_a_emotion if emotion else None,
        emotion_b=emotion.user_b_emotion if emotion else None,
        sentiment_a=emotion.user_a_sentiment if emotion else None,
        sentiment_b=emotion.user_b_sentiment if emotion else None,
        intent_a=intent.user_a_intent if intent else None,
        intent_b=intent.user_b_intent if intent else None,
        intent_confidence_a=intent.confidence_a if intent else None,
        intent_confidence_b=intent.confidence_b if intent else None,
        conflict_type=reason.conflict_type if reason else None,
        resolvability=reason.resolvability if reason else None,
        user_a_goal=reason.user_a_goal if reason else None,
        user_b_goal=reason.user_b_goal if reason else None,
        common_ground=reason.common_ground if reason else None,
        resolution_strategy=reason.resolution_strategy if reason else None,
        one_line_summary=reason.one_line_summary if reason else None,
        fairness_score=fairness_score(out.response_a, out.response_b) if out else None,
        critic_approved=critic.approved if critic else None,
        critic_feedback=critic.reason if critic and not critic.approved else None,
        safety_score=_safety_score(safety),
        selected_strategy=(
            out.reasoning_state.get("selected_strategy")
            if out and isinstance(out.reasoning_state, dict)
            else None
        ),
        reasoning_uncertainty=(
            out.reasoning_state.get("reasoning_uncertainty")
            if out and isinstance(out.reasoning_state, dict)
            else None
        ),
        production_checks=out.verification if out else None,
        repair_applied=(
            bool(out.verification.get("repair_applied"))
            if out and isinstance(out.verification, dict)
            else False
        ),
        retrieved_cases=len(retr.retrieved_cases) if retr else 0,
        rag_used=retr.rag_used if retr else False,
        fallback_used=state.get("fallback_used", False),
        processing_time_seconds=elapsed,
        retries=state.get("retry_count", 0),
    )


async def node_retrieve(state: PipelineState) -> PipelineState:
    inp = state["user_input"]
    try:
        result = await asyncio.to_thread(retriever.retrieve, inp.text_a, inp.text_b)
        print(
            f"[graph:{inp.trace_id}] retrieve cases={len(result.retrieved_cases)} "
            f"top={result.top_score} rag_used={result.rag_used}"
        )
        return {**state, "retriever_output": result}
    except Exception as exc:
        print(f"[graph:{inp.trace_id}] retrieval skipped: {exc}")
        return {**state, "retriever_output": RetrieverOutput(rag_used=False)}


async def node_validate_input(state: PipelineState) -> PipelineState:
    inp = state["user_input"]
    neutral_safety = SafetyResult(approved=True)
    result = await validator.validate(
        text_a=inp.text_a,
        text_b=inp.text_b,
        response_a="",
        response_b="",
        safety_result=neutral_safety,
    )
    print(
        f"[graph:{inp.trace_id}] input signals "
        f"stance=({result.stance.user_a_stance},{result.stance.user_b_stance}) "
        f"emotion=({result.emotion.user_a_emotion},{result.emotion.user_b_emotion}) "
        f"intent=({result.intent.user_a_intent},{result.intent.user_b_intent})"
    )
    return {
        **state,
        "stance_result": result.stance,
        "emotion_result": result.emotion,
        "intent_result": result.intent,
    }


async def node_retrieve_and_validate(state: PipelineState) -> PipelineState:
    """Run RAG retrieval and input validation concurrently."""
    inp = state["user_input"]
    if inp.mode in {"fast", "fast_production"}:
        try:
            retrieve_output = await asyncio.to_thread(retriever.retrieve, inp.text_a, inp.text_b)
            print(
                f"[graph:{inp.trace_id}] retrieve cases={len(retrieve_output.retrieved_cases)} "
                f"top={retrieve_output.top_score} rag_used={retrieve_output.rag_used}"
            )
        except Exception as exc:
            print(f"[graph:{inp.trace_id}] retrieval skipped: {exc}")
            retrieve_output = RetrieverOutput(rag_used=False)
        print(f"[graph:{inp.trace_id}] input validation skipped in {inp.mode} mode")
        intent = validator.infer_intents(
            inp.text_a,
            inp.text_b,
            feedback=f"Input intent inferred deterministically in {inp.mode} mode.",
        )
        return {
            **state,
            "retriever_output": retrieve_output,
            "stance_result": StanceResult(
                user_a_stance="neutral",
                user_b_stance="neutral",
                response_consistent=True,
                feedback=f"Input validation skipped in {inp.mode} mode.",
            ),
            "emotion_result": EmotionResult(
                user_a_emotion="neutral",
                user_b_emotion="neutral",
                user_a_sentiment="neutral",
                user_b_sentiment="neutral",
                response_empathetic=True,
                feedback=f"Input validation skipped in {inp.mode} mode.",
            ),
            "intent_result": intent,
        }

    neutral_safety = SafetyResult(approved=True)

    retrieve_task = asyncio.to_thread(retriever.retrieve, inp.text_a, inp.text_b)
    validate_task = validator.validate(
        text_a=inp.text_a,
        text_b=inp.text_b,
        response_a="",
        response_b="",
        safety_result=neutral_safety,
    )

    retrieve_result, validate_result = await asyncio.gather(
        retrieve_task,
        validate_task,
        return_exceptions=True,
    )

    if isinstance(retrieve_result, Exception):
        print(f"[graph:{inp.trace_id}] retrieval skipped: {retrieve_result}")
        retrieve_output = RetrieverOutput(rag_used=False)
    else:
        retrieve_output = retrieve_result
        print(
            f"[graph:{inp.trace_id}] retrieve cases={len(retrieve_output.retrieved_cases)} "
            f"top={retrieve_output.top_score} rag_used={retrieve_output.rag_used}"
        )

    if isinstance(validate_result, Exception):
        print(f"[graph:{inp.trace_id}] input validation skipped: {validate_result}")
        stance = StanceResult(
            user_a_stance="neutral",
            user_b_stance="neutral",
            response_consistent=True,
            feedback=f"Input validation skipped: {validate_result}",
        )
        emotion = EmotionResult(
            user_a_emotion="neutral",
            user_b_emotion="neutral",
            user_a_sentiment="neutral",
            user_b_sentiment="neutral",
            response_empathetic=True,
            feedback=f"Input validation skipped: {validate_result}",
        )
        intent = validator.infer_intents(
            inp.text_a,
            inp.text_b,
            feedback=f"Input intent inferred after validation skip: {validate_result}",
        )
    else:
        stance = validate_result.stance
        emotion = validate_result.emotion
        intent = validate_result.intent
        print(
            f"[graph:{inp.trace_id}] input signals "
            f"stance=({stance.user_a_stance},{stance.user_b_stance}) "
            f"emotion=({emotion.user_a_emotion},{emotion.user_b_emotion}) "
            f"intent=({intent.user_a_intent},{intent.user_b_intent})"
        )

    return {
        **state,
        "retriever_output": retrieve_output,
        "stance_result": stance,
        "emotion_result": emotion,
        "intent_result": intent,
    }


async def node_reason(state: PipelineState) -> PipelineState:
    inp = state["user_input"]
    stance = state.get("stance_result")
    emotion = state.get("emotion_result")
    intent = state.get("intent_result")
    try:
        reasoning = await reasoning_agent.run(
            text_a=inp.text_a,
            text_b=inp.text_b,
            stance_a=stance.user_a_stance if stance else "neutral",
            stance_b=stance.user_b_stance if stance else "neutral",
            emotion_a=emotion.user_a_emotion if emotion else "neutral",
            emotion_b=emotion.user_b_emotion if emotion else "neutral",
            sentiment_a=emotion.user_a_sentiment if emotion else "neutral",
            sentiment_b=emotion.user_b_sentiment if emotion else "neutral",
            intent_a=intent.user_a_intent if intent else "unknown",
            intent_b=intent.user_b_intent if intent else "unknown",
            prior_context=inp.prior_context,
        )
        print(
            f"[graph:{inp.trace_id}] reason type={reasoning.conflict_type} "
            f"resolvability={reasoning.resolvability}"
        )
        return {**state, "reasoning_output": reasoning}
    except Exception as exc:
        print(f"[graph:{inp.trace_id}] reasoning failed, using dummy reasoning: {exc}")
        return {
            **state,
            "reasoning_output": _dummy_reasoning(),
        }


async def node_production(state: PipelineState) -> PipelineState:
    inp = state["user_input"]
    cases = state["retriever_output"].retrieved_cases if state.get("retriever_output") else []
    intent = state.get("intent_result")
    try:
        output = await production_agent.run(
            text_a=inp.text_a,
            text_b=inp.text_b,
            retrieved_cases=cases,
            prior_context=inp.prior_context,
            intent_a=intent.user_a_intent if intent else "unknown",
            intent_b=intent.user_b_intent if intent else "unknown",
        )
        print(
            f"[graph:{inp.trace_id}] production type={output.reasoning.conflict_type} "
            f"status={output.conversation_status}"
        )
        return {
            **state,
            "reasoning_output": output.reasoning,
            "llm_output": output,
            "feedback": None,
        }
    except Exception as exc:
        print(f"[graph:{inp.trace_id}] production failed, using fallback: {exc}")
        reasoning = _fallback_reasoning_from_input(inp)
        return {
            **state,
            "reasoning_output": reasoning,
            "llm_output": _fallback_output(reasoning, inp),
            "feedback": None,
            "fallback_used": True,
        }


async def node_reconcile(state: PipelineState) -> PipelineState:
    inp = state["user_input"]
    if state.get("fallback_used"):
        return {**state, "llm_output": _fallback_output(state.get("reasoning_output"), inp), "feedback": None}

    cases = state["retriever_output"].retrieved_cases if state.get("retriever_output") else []
    try:
        output = await reconciliation_agent.run(
            text_a=inp.text_a,
            text_b=inp.text_b,
            reasoning=state["reasoning_output"] or _dummy_reasoning(),
            retrieved_cases=cases,
            feedback=state.get("feedback"),
        )
        print(f"[graph:{inp.trace_id}] reconcile status={output.conversation_status}")
        return {**state, "llm_output": output, "feedback": None}
    except Exception as exc:
        print(f"[graph:{inp.trace_id}] reconciliation failed, using fallback: {exc}")
        return {
            **state,
            "llm_output": _fallback_output(state.get("reasoning_output"), inp),
            "feedback": None,
            "fallback_used": True,
        }


async def node_safety(state: PipelineState) -> PipelineState:
    inp = state["user_input"]
    out = state["llm_output"] or _fallback_output(state.get("reasoning_output"), inp)
    result = await safety_agent.check(
        out.response_a,
        out.response_b,
        use_toxic_bert=inp.mode == "quality",
    )
    print(f"[graph:{inp.trace_id}] safety approved={result.approved} score={_safety_score(result):.3f}")
    return {**state, "safety_result": result}


async def node_validate_output(state: PipelineState) -> PipelineState:
    inp = state["user_input"]
    out = state["llm_output"] or _fallback_output(state.get("reasoning_output"), inp)
    if inp.mode in {"fast", "fast_production"}:
        result = ValidationOutput(
            stance=state.get("stance_result")
            or StanceResult(user_a_stance="neutral", user_b_stance="neutral", response_consistent=True),
            emotion=state.get("emotion_result")
            or EmotionResult(
                user_a_emotion="neutral",
                user_b_emotion="neutral",
                user_a_sentiment="neutral",
                user_b_sentiment="neutral",
                response_empathetic=True,
            ),
            intent=state.get("intent_result") or validator.infer_intents(inp.text_a, inp.text_b),
            safety=state.get("safety_result") or SafetyResult(approved=True),
            overall_passed=(state.get("safety_result") or SafetyResult(approved=True)).approved,
            aggregated_feedback=(state.get("safety_result") or SafetyResult(approved=True)).feedback,
        )
        print(f"[graph:{inp.trace_id}] output validation skipped in {inp.mode} mode")
        return {**state, "validation_output": result}

    result = await validator.validate(
        text_a=inp.text_a,
        text_b=inp.text_b,
        response_a=out.response_a,
        response_b=out.response_b,
        safety_result=state.get("safety_result") or SafetyResult(approved=True),
    )
    print(f"[graph:{inp.trace_id}] output validation passed={result.overall_passed}")
    return {**state, "validation_output": result}


async def node_critic(state: PipelineState) -> PipelineState:
    inp = state["user_input"]
    if state.get("fallback_used"):
        return {
            **state,
            "critic_output": ReflectionOutput(
                approved=True,
                is_fair=True,
                is_empathetic=True,
                is_unbiased=True,
                skipped=True,
                reason="Skipped because fallback response was used.",
            ),
        }

    out = state["llm_output"] or _fallback_output(state.get("reasoning_output"), inp)
    basic_failure = critic_agent.basic_response_check(out.response_a, out.response_b)
    if basic_failure:
        return {**state, "critic_output": basic_failure}

    if inp.mode == "fast_production" and isinstance(out.verification, dict) and not out.verification.get("passed", True):
        issues = out.verification.get("issues") or ["production verification failed"]
        return {
            **state,
            "critic_output": ReflectionOutput(
                approved=False,
                is_fair=False,
                is_empathetic=False,
                is_unbiased=True,
                is_specific=False,
                is_non_escalatory=True,
                skipped=True,
                reason="; ".join(str(issue) for issue in issues),
                suggestion="Use fallback because fast production verification failed after repair.",
            ),
        }

    if inp.mode in {"fast", "fast_production"}:
        return {
            **state,
            "critic_output": ReflectionOutput(
                approved=True,
                is_fair=True,
                is_empathetic=True,
                is_unbiased=True,
                skipped=True,
                reason=f"Skipped in {inp.mode} mode.",
            ),
        }

    reasoning = state.get("reasoning_output") or out.reasoning
    try:
        result = await critic_agent.critique(
            text_a=inp.text_a,
            text_b=inp.text_b,
            response_a=out.response_a,
            response_b=out.response_b,
            conflict_type=reasoning.conflict_type,
            resolvability=reasoning.resolvability,
        )
        print(f"[graph:{inp.trace_id}] critic approved={result.approved}")
        return {**state, "critic_output": result}
    except Exception as exc:
        print(f"[graph:{inp.trace_id}] critic skipped after failure: {exc}")
        return {
            **state,
            "critic_output": ReflectionOutput(
                approved=True,
                is_fair=True,
                is_empathetic=True,
                is_unbiased=True,
                skipped=True,
                reason=f"Critic skipped after failure: {exc}",
            ),
        }


async def node_retry(state: PipelineState) -> PipelineState:
    new_count = state["retry_count"] + 1
    parts: list[str] = []
    validation = state.get("validation_output")
    if validation and validation.aggregated_feedback:
        parts.append(validation.aggregated_feedback)
    critic = state.get("critic_output")
    if critic and not critic.approved:
        parts.append(critic.suggestion or critic.reason or "Improve fairness, empathy, and specificity.")

    feedback = "\n".join(parts) if parts else "Improve fairness, empathy, safety, and specificity."
    print(f"[graph:{state['user_input'].trace_id}] retry {new_count}/{MAX_RETRIES}")
    return {**state, "retry_count": new_count, "feedback": feedback}


async def node_fallback(state: PipelineState) -> PipelineState:
    inp = state["user_input"]
    print(f"[graph:{inp.trace_id}] final fallback activated")
    return {
        **state,
        "llm_output": _fallback_output(state.get("reasoning_output"), inp),
        "safety_result": SafetyResult(approved=True),
        "critic_output": ReflectionOutput(
            approved=True,
            is_fair=True,
            is_empathetic=True,
            is_unbiased=True,
            skipped=True,
            reason="Fallback response used after retries.",
        ),
        "fallback_used": True,
        "stored_to_memory": False,
    }


async def node_memory(state: PipelineState) -> PipelineState:
    inp = state["user_input"]
    if state.get("fallback_used"):
        print(f"[graph:{inp.trace_id}] memory skip: fallback_used")
        return {**state, "stored_to_memory": False}
    safety = state.get("safety_result")
    if not safety or not safety.approved:
        print(f"[graph:{inp.trace_id}] memory skip: safety_not_approved")
        return {**state, "stored_to_memory": False}
    critic = state.get("critic_output")
    if not critic or not critic.approved:
        print(f"[graph:{inp.trace_id}] memory skip: critic_not_approved")
        return {**state, "stored_to_memory": False}
    out = state.get("llm_output")
    if not out:
        print(f"[graph:{inp.trace_id}] memory skip: production_checks_failed")
        return {**state, "stored_to_memory": False}
    if isinstance(out.verification, dict) and not out.verification.get("passed", True):
        print(f"[graph:{inp.trace_id}] memory skip: production_checks_failed")
        return {**state, "stored_to_memory": False}
    conf = confidence_score(
        critic_approved=critic.approved,
        retries=state.get("retry_count", 0),
        safety_score=max(safety.detoxify_score, safety.toxic_bert_score),
        fallback_used=state.get("fallback_used", False),
        critic_skipped=critic.skipped,
        production_mode=inp.mode == "fast_production",
    )
    if conf < 0.75:
        print(f"[graph:{inp.trace_id}] memory skip: confidence_below_threshold conf={conf:.3f}")
        return {**state, "stored_to_memory": False}

    reasoning = state.get("reasoning_output") or out.reasoning
    try:
        stored = await asyncio.to_thread(
            retriever.store_case,
            conversation_id=inp.conversation_id,
            text_a=inp.text_a,
            text_b=inp.text_b,
            conflict_type=reasoning.conflict_type,
            resolution_strategy=reasoning.resolution_strategy,
            response_a=out.response_a,
            response_b=out.response_b,
        )
        return {**state, "stored_to_memory": stored}
    except Exception as exc:
        print(f"[graph:{inp.trace_id}] memory write skipped: {exc}")
        return {**state, "stored_to_memory": False}


async def node_finalize(state: PipelineState) -> PipelineState:
    inp = state["user_input"]
    out = state["llm_output"] or _fallback_output(state.get("reasoning_output"), inp)
    reasoning = state.get("reasoning_output") or out.reasoning
    trace = _build_trace(state)
    critic = state.get("critic_output")
    approved = critic.approved if critic else True
    confidence = confidence_score(
        critic_approved=approved,
        retries=state["retry_count"],
        safety_score=trace.safety_score or 0.0,
        fallback_used=state.get("fallback_used", False),
        critic_skipped=critic.skipped if critic else False,
        production_mode=inp.mode == "fast_production",
    )

    final = FinalOutput(
        conversation_id=inp.conversation_id,
        turn=inp.turn,
        request_id=inp.request_id or "",
        trace_id=inp.trace_id or "",
        response_a=out.response_a,
        response_b=out.response_b,
        conversation_status=out.conversation_status,
        conflict_type=reasoning.conflict_type,
        resolvability=reasoning.resolvability,
        one_line_summary=reasoning.one_line_summary,
        confidence=confidence,
        retries=state["retry_count"],
        stored_to_memory=state.get("stored_to_memory", False),
        processing_time_seconds=trace.processing_time_seconds,
        trace=trace,
    )
    _remember(final)
    record_request(
        latency=final.processing_time_seconds or 0.0,
        retries=final.retries,
        fallback_used=trace.fallback_used,
        memory_written=final.stored_to_memory,
        resolvability=final.resolvability,
    )
    print(
        f"[graph:{inp.trace_id}] done status={final.conversation_status} "
        f"confidence={final.confidence} retries={final.retries}"
    )
    return {**state, "final_output": final}


def route_after_validation(state: PipelineState) -> str:
    if state["user_input"].mode == "fast_production":
        safety = state.get("safety_result")
        validation = state.get("validation_output")
        if (safety and not safety.approved) or (validation and not validation.overall_passed):
            return "fallback"
        return "critic"

    safety = state.get("safety_result")
    validation = state.get("validation_output")
    if safety and not safety.approved:
        return "retry" if state["retry_count"] < MAX_RETRIES else "fallback"
    if validation and not validation.overall_passed:
        return "retry" if state["retry_count"] < MAX_RETRIES else "critic"
    return "critic"


def route_after_critic(state: PipelineState) -> str:
    critic = state.get("critic_output")
    if state.get("fallback_used") or not critic or critic.approved:
        return "memory"
    if state["user_input"].mode == "fast_production":
        return "fallback"
    return "retry" if state["retry_count"] < MAX_RETRIES else "fallback"


def route_after_retrieve_and_validate(state: PipelineState) -> str:
    return "production" if state["user_input"].mode == "fast_production" else "reason"


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("retrieve_and_validate", node_retrieve_and_validate)
    graph.add_node("reason", node_reason)
    graph.add_node("production", node_production)
    graph.add_node("reconcile", node_reconcile)
    graph.add_node("safety", node_safety)
    graph.add_node("validate_output", node_validate_output)
    graph.add_node("critic", node_critic)
    graph.add_node("retry", node_retry)
    graph.add_node("fallback", node_fallback)
    graph.add_node("memory", node_memory)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("retrieve_and_validate")
    graph.add_conditional_edges(
        "retrieve_and_validate",
        route_after_retrieve_and_validate,
        {"production": "production", "reason": "reason"},
    )
    graph.add_edge("production", "safety")
    graph.add_edge("reason", "reconcile")
    graph.add_edge("reconcile", "safety")
    graph.add_edge("safety", "validate_output")
    graph.add_conditional_edges(
        "validate_output",
        route_after_validation,
        {"retry": "retry", "critic": "critic", "fallback": "fallback"},
    )
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"retry": "retry", "memory": "memory", "fallback": "fallback"},
    )
    graph.add_edge("retry", "reconcile")
    graph.add_edge("fallback", "finalize")
    graph.add_edge("memory", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


async def run_pipeline(user_input: UserInput) -> FinalOutput:
    trace_id = user_input.trace_id or str(uuid.uuid4())
    enriched = user_input.model_copy(
        update={
            "trace_id": trace_id,
            "prior_context": user_input.prior_context or _prior_context(user_input.conversation_id),
        }
    )
    initial: PipelineState = {
        "user_input": enriched,
        "retriever_output": None,
        "stance_result": None,
        "emotion_result": None,
        "intent_result": None,
        "reasoning_output": None,
        "llm_output": None,
        "safety_result": None,
        "validation_output": None,
        "critic_output": None,
        "retry_count": 0,
        "feedback": None,
        "final_output": None,
        "stored_to_memory": False,
        "fallback_used": False,
        "start_time": time.time(),
    }
    result = await get_graph().ainvoke(initial)
    return result["final_output"]


def run_pipeline_sync(user_input: UserInput) -> FinalOutput:
    return asyncio.run(run_pipeline(user_input))
