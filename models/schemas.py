"""Pydantic contracts shared by all agents and API endpoints."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


Mode = Literal["quality", "fast", "fast_production"]
ConflictType = Literal["emotional", "logical", "misunderstanding", "value"]
Resolvability = Literal["resolvable", "partially_resolvable", "non_resolvable"]
ConversationStatus = Literal["Escalating", "Stable", "Improving", "Resolved"]
CommunicativeIntent = Literal[
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
]


class UserInput(BaseModel):
    conversation_id: str
    text_a: str = Field(..., description="Raw text from User A")
    text_b: str = Field(..., description="Raw text from User B")
    turn: int = Field(default=1, ge=1)
    mode: Mode = "quality"
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    prior_context: Optional[str] = None


class RetrievedCase(BaseModel):
    conflict_type: str
    text_a: str
    text_b: str
    resolution_strategy: str
    response_a: str
    response_b: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class RetrieverOutput(BaseModel):
    retrieved_cases: list[RetrievedCase] = Field(default_factory=list)
    query_embedding_used: bool = True
    top_score: Optional[float] = None
    rag_used: bool = False


class StanceResult(BaseModel):
    user_a_stance: Literal["support", "attack", "neutral"]
    user_b_stance: Literal["support", "attack", "neutral"]
    response_consistent: bool = True
    confidence_a: Optional[float] = None
    confidence_b: Optional[float] = None
    feedback: Optional[str] = None


class EmotionResult(BaseModel):
    user_a_emotion: str
    user_b_emotion: str
    user_a_sentiment: Literal["positive", "negative", "neutral"] = "neutral"
    user_b_sentiment: Literal["positive", "negative", "neutral"] = "neutral"
    response_empathetic: bool = True
    confidence_a: Optional[float] = None
    confidence_b: Optional[float] = None
    feedback: Optional[str] = None


class IntentResult(BaseModel):
    user_a_intent: CommunicativeIntent = "unknown"
    user_b_intent: CommunicativeIntent = "unknown"
    confidence_a: Optional[float] = None
    confidence_b: Optional[float] = None
    feedback: Optional[str] = None


class Reasoning(BaseModel):
    user_a_goal: str
    user_b_goal: str
    conflict_type: ConflictType
    resolvability: Resolvability = "partially_resolvable"
    common_ground: str
    resolution_strategy: str
    one_line_summary: Optional[str] = None


class LLMCoreOutput(BaseModel):
    reasoning: Reasoning
    response_a: str
    response_b: str
    conversation_status: ConversationStatus = "Stable"
    reasoning_state: Optional[dict[str, Any]] = None
    verification: Optional[dict[str, Any]] = None


class SafetyResult(BaseModel):
    approved: bool = True
    detoxify_score: float = Field(default=0.0, ge=0.0, le=1.0)
    toxic_bert_score: float = Field(default=0.0, ge=0.0, le=1.0)
    flagged_categories: list[str] = Field(default_factory=list)
    feedback: Optional[str] = None


class ValidationOutput(BaseModel):
    stance: StanceResult
    emotion: EmotionResult
    intent: IntentResult
    safety: SafetyResult
    overall_passed: bool
    aggregated_feedback: Optional[str] = None


class ReflectionOutput(BaseModel):
    approved: bool
    is_fair: bool
    is_empathetic: bool
    is_unbiased: bool
    is_specific: bool = True
    is_non_escalatory: bool = True
    reason: Optional[str] = None
    suggestion: Optional[str] = None
    skipped: bool = False


class MemoryEntry(BaseModel):
    conversation_id: str
    turn: int
    text_a: str
    text_b: str
    conflict_type: str
    resolution_strategy: str
    response_a: str
    response_b: str
    conversation_status: str
    validation_passed: bool
    critic_approved: bool


class AgentTrace(BaseModel):
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    mode: Mode = "quality"
    stance_a: Optional[str] = None
    stance_b: Optional[str] = None
    emotion_a: Optional[str] = None
    emotion_b: Optional[str] = None
    sentiment_a: Optional[str] = None
    sentiment_b: Optional[str] = None
    intent_a: Optional[str] = None
    intent_b: Optional[str] = None
    intent_confidence_a: Optional[float] = None
    intent_confidence_b: Optional[float] = None
    conflict_type: Optional[str] = None
    resolvability: Optional[str] = None
    user_a_goal: Optional[str] = None
    user_b_goal: Optional[str] = None
    common_ground: Optional[str] = None
    resolution_strategy: Optional[str] = None
    one_line_summary: Optional[str] = None
    fairness_score: Optional[float] = None
    critic_approved: Optional[bool] = None
    critic_feedback: Optional[str] = None
    safety_score: Optional[float] = None
    selected_strategy: Optional[str] = None
    reasoning_uncertainty: Optional[float] = None
    production_checks: Optional[dict[str, Any]] = None
    repair_applied: bool = False
    retrieved_cases: int = 0
    rag_used: bool = False
    fallback_used: bool = False
    processing_time_seconds: Optional[float] = None
    retries: int = 0


class FinalOutput(BaseModel):
    conversation_id: str
    turn: int
    request_id: str
    trace_id: str
    response_a: str
    response_b: str
    conversation_status: ConversationStatus
    conflict_type: str
    resolvability: str = "partially_resolvable"
    one_line_summary: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    retries: int = 0
    stored_to_memory: bool = False
    processing_time_seconds: Optional[float] = None
    trace: Optional[AgentTrace] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    fallback_used: bool = False
    trace_id: Optional[str] = None


class ErrorEnvelope(BaseModel):
    error: ErrorDetail
