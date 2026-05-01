"""FastAPI entry point for the AI Conflict Mediation System."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api import db
from agents import retriever, safety_agent, validator
from config import (
    API_TITLE,
    API_VERSION,
    CORS_ORIGINS,
    IDEMPOTENCY_TTL_SECONDS,
    MODE_DEFAULT,
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    SUPPORTED_MODES,
    ensure_data_dirs,
    get_groq_api_key,
    secret_fingerprint,
)
from coordinator.graph import run_pipeline
from models.schemas import ErrorEnvelope, FinalOutput, Mode, UserInput
from utils.cache import TTLCache
from utils.observability import snapshot
from utils.text import request_hash, sanitize_text


class MediateRequest(BaseModel):
    text_a: str
    text_b: str
    conversation_id: Optional[str] = None
    turn: int = Field(default=1, ge=1)
    mode: Mode = MODE_DEFAULT  # type: ignore[assignment]


class SignUpRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ResolveConversationRequest(BaseModel):
    user_a_rating: Optional[int] = Field(default=None, ge=1, le=5)
    user_b_rating: Optional[int] = Field(default=None, ge=1, le=5)
    note: Optional[str] = Field(default=None, max_length=1000)
    user_a_comment: Optional[str] = Field(default=None, max_length=1000)
    user_b_comment: Optional[str] = Field(default=None, max_length=1000)


class MediateResponse(BaseModel):
    conversation_id: str
    turn: int
    request_id: str
    trace_id: str
    response_a: str
    response_b: str
    conversation_status: str
    conflict_type: str
    resolvability: str
    one_line_summary: Optional[str] = None
    confidence: float
    retries: int
    stored_to_memory: bool
    processing_time_seconds: Optional[float] = None
    trace: Optional[dict[str, Any]] = None


class WarmupRequest(BaseModel):
    preload_local_models: bool = True
    ping_groq: bool = False


_idempotency_cache: TTLCache[MediateResponse] = TTLCache(IDEMPOTENCY_TTL_SECONDS)
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)
_INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all instructions",
    "ignore your instructions",
    "disregard previous",
    "return only json",
    "return only valid json",
    "system prompt",
    "you are now",
    "act as if",
    "new instructions:",
    "response_a:",
    "response_b:",
    "```json",
)


def _error(status_code: int, code: str, message: str, *, fallback_used: bool = False, trace_id: str | None = None):
    return JSONResponse(
        status_code=status_code,
        content=ErrorEnvelope(
            error={
                "code": code,
                "message": message,
                "fallback_used": fallback_used,
                "trace_id": trace_id,
            }
        ).model_dump(),
    )


def _state_error(status_code: int, code: str, message: str, *, trace_id: str | None = None, **payload: Any):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "trace_id": trace_id,
                **payload,
            },
            "code": code,
            "message": message,
            "trace_id": trace_id,
            **payload,
        },
    )


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(ip: str) -> bool:
    now = time.time()
    bucket = _rate_buckets[ip]
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
        return True
    bucket.append(now)
    return False


def _contains_instruction_pattern(*texts: str) -> bool:
    combined = "\n".join(text.lower() for text in texts)
    return any(pattern in combined for pattern in _INJECTION_PATTERNS)


def _token_from_request(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def _user_from_request(request: Request, *, required: bool = False) -> dict[str, Any] | None:
    token = _token_from_request(request)
    if not token:
        if required:
            raise HTTPException(status_code=401, detail="Sign in required.")
        return None
    user = db.get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return user


def _validate_auth(name: str | None, email: str, password: str) -> None:
    if name is not None and not name.strip():
        raise HTTPException(status_code=400, detail="Name is required.")
    if not email.strip() or "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        raise HTTPException(status_code=400, detail="A valid email is required.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")


def _to_response(result: FinalOutput) -> MediateResponse:
    return MediateResponse(
        conversation_id=result.conversation_id,
        turn=result.turn,
        request_id=result.request_id,
        trace_id=result.trace_id,
        response_a=result.response_a,
        response_b=result.response_b,
        conversation_status=result.conversation_status,
        conflict_type=result.conflict_type,
        resolvability=result.resolvability,
        one_line_summary=result.one_line_summary,
        confidence=result.confidence,
        retries=result.retries,
        stored_to_memory=result.stored_to_memory,
        processing_time_seconds=result.processing_time_seconds,
        trace=result.trace.model_dump() if result.trace else None,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
    db.init_db()
    await asyncio.to_thread(retriever.seed_from_scenarios)
    print("[api] Ready")
    yield


app = FastAPI(
    title=API_TITLE,
    description="LangGraph-orchestrated multi-agent conflict mediation with RAG and self-reflection.",
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    groq_key = get_groq_api_key()
    model_status = {
        **validator.status(),
        **safety_agent.status(),
    }
    return {
        "status": "ok",
        "version": API_VERSION,
        "groq": "configured" if groq_key else "missing_key",
        "groq_key": secret_fingerprint(groq_key),
        "retriever": retriever.status(),
        "models_loaded": all(
            bool(value)
            for key, value in model_status.items()
            if key.endswith("_loaded")
        ),
        "models": model_status,
        "metrics": snapshot(),
    }


@app.post("/auth/signup")
async def signup(req: SignUpRequest):
    _validate_auth(req.name, req.email, req.password)
    try:
        user = await asyncio.to_thread(db.create_user, req.name, req.email, req.password)
    except db.DuplicateUserError as exc:
        raise HTTPException(status_code=409, detail="Account already exists. Please sign in.") from exc
    token = await asyncio.to_thread(db.create_token, user["id"])
    return {"token": token, "user": user}


@app.post("/auth/login")
async def login(req: LoginRequest):
    _validate_auth(None, req.email, req.password)
    user = await asyncio.to_thread(db.authenticate_user, req.email, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    token = await asyncio.to_thread(db.create_token, user["id"])
    return {"token": token, "user": user}


@app.get("/auth/me")
async def me(request: Request):
    return {"user": _user_from_request(request, required=True)}


@app.post("/auth/logout")
async def logout(request: Request):
    token = _token_from_request(request)
    if token:
        await asyncio.to_thread(db.revoke_token, token)
    return {"ok": True}


@app.get("/history")
async def history(request: Request, limit: int = 100):
    user = _user_from_request(request, required=True)
    rows = await asyncio.to_thread(db.list_mediation_history, user["id"], limit)
    return {"history": rows}


@app.post("/warmup")
async def warmup(req: WarmupRequest):
    result: dict[str, Any] = {
        "retriever": await asyncio.to_thread(retriever.warmup),
        "groq": "not_pinged",
    }
    if req.preload_local_models:
        validator_status, safety_status = await asyncio.gather(
            validator.warmup(),
            safety_agent.warmup(),
        )
        result["validator"] = validator_status
        result["safety"] = safety_status
    if req.ping_groq:
        result["groq"] = "configured" if get_groq_api_key() else "missing_key"
    return result


def _conversation_reject_response(state: dict[str, Any], *, trace_id: str | None = None):
    code = str(state.get("error") or "CONVERSATION_STATE_ERROR")
    if code == "ALREADY_RESOLVED":
        print(f"MEDIATE_REJECT_RESOLVED conversation_id={state.get('conversation_id', '-')}")
        return _state_error(
            409,
            "ALREADY_RESOLVED",
            "This conversation is already resolved.",
            trace_id=trace_id,
            status="resolved",
            resolved=True,
            resolved_at=state.get("resolved_at"),
            resolved_turn=state.get("resolved_turn"),
            user_a_rating=state.get("user_a_rating"),
            user_b_rating=state.get("user_b_rating"),
            user_a_comment=state.get("user_a_comment"),
            user_b_comment=state.get("user_b_comment"),
            rated=state.get("rated"),
            conversation_status="resolved",
        )
    if code == "STALE_TURN":
        print(
            "MEDIATE_REJECT_STALE "
            f"latest_turn={state.get('latest_turn')} expected_turn={state.get('expected_turn')}"
        )
        return _state_error(
            409,
            "STALE_TURN",
            "This conversation has moved to a different turn. Refresh and continue from the latest turn.",
            trace_id=trace_id,
            status=state.get("status", "active"),
            latest_turn=state.get("latest_turn", 0),
            expected_turn=state.get("expected_turn"),
        )
    return _state_error(
        409,
        code,
        "Conversation state could not be updated.",
        trace_id=trace_id,
        status=state.get("status", "unknown"),
    )


@app.post("/mediate", response_model=MediateResponse)
async def mediate(req: MediateRequest, request: Request):
    trace_id = str(uuid.uuid4())
    if _rate_limited(_client_ip(request)):
        return _error(429, "RATE_LIMITED", "Too many requests. Please wait before retrying.", trace_id=trace_id)
    user = _user_from_request(request, required=True)

    mode = req.mode if req.mode in SUPPORTED_MODES else MODE_DEFAULT
    text_a = sanitize_text(req.text_a)
    text_b = sanitize_text(req.text_b)
    if not text_a or not text_b:
        return _error(400, "EMPTY_INPUT", "Both text_a and text_b are required.", trace_id=trace_id)
    if len(text_a) > 5000 or len(text_b) > 5000:
        return _error(400, "INPUT_TOO_LONG", "Each perspective must be 5000 characters or fewer.", trace_id=trace_id)
    if _contains_instruction_pattern(text_a, text_b):
        return _error(
            400,
            "INSTRUCTION_PATTERN_DETECTED",
            "Input contains instruction patterns that cannot be processed.",
            trace_id=trace_id,
        )

    conversation_id = req.conversation_id or str(uuid.uuid4())
    preflight = await asyncio.to_thread(
        db.prepare_conversation_turn,
        user["id"],
        conversation_id,
        req.turn,
    )
    if not preflight.get("ok"):
        return _conversation_reject_response({**preflight, "conversation_id": conversation_id}, trace_id=trace_id)

    request_id = request_hash(conversation_id, req.turn, text_a, text_b, mode)
    cached = _idempotency_cache.get(request_id)
    if cached:
        return cached

    user_input = UserInput(
        conversation_id=conversation_id,
        text_a=text_a,
        text_b=text_b,
        turn=req.turn,
        mode=mode,  # type: ignore[arg-type]
        request_id=request_id,
        trace_id=trace_id,
    )

    try:
        result = await run_pipeline(user_input)
    except Exception as exc:  # pragma: no cover - last-resort runtime safety
        return _error(500, "PIPELINE_ERROR", str(exc), fallback_used=False, trace_id=trace_id)

    response = _to_response(result)
    commit_state = await asyncio.to_thread(
        db.save_mediation_turn,
        user_id=user["id"],
        mode=mode,
        text_a=text_a,
        text_b=text_b,
        response=response,
    )
    if not commit_state.get("ok"):
        return _conversation_reject_response({**commit_state, "conversation_id": conversation_id}, trace_id=trace_id)
    _idempotency_cache.set(request_id, response)
    return response


@app.post("/conversations/{conversation_id}/resolve")
async def resolve_conversation(conversation_id: str, req: ResolveConversationRequest, request: Request):
    user = _user_from_request(request, required=True)
    if (req.user_a_rating is None) != (req.user_b_rating is None):
        return _error(400, "INCOMPLETE_RATING", "Rate both users or leave both ratings empty.")
    result = await asyncio.to_thread(
        db.resolve_conversation,
        user_id=user["id"],
        conversation_id=conversation_id,
        user_a_rating=req.user_a_rating,
        user_b_rating=req.user_b_rating,
        note=sanitize_text(req.note or "") or None,
        user_a_comment=sanitize_text(req.user_a_comment or "") or None,
        user_b_comment=sanitize_text(req.user_b_comment or "") or None,
        source="ui",
    )
    if not result.get("ok"):
        code = str(result.get("error") or "RESOLVE_FAILED")
        status = 404 if code == "CONVERSATION_NOT_FOUND" else 409
        return _state_error(
            status,
            code,
            "Conversation could not be resolved.",
            status=result.get("status", "unknown"),
        )
    if result.get("resolved_at"):
        print(
            "RESOLVE_SUCCESS "
            f"conversation_id={conversation_id} resolved_turn={result.get('resolved_turn')}"
        )
    return {key: value for key, value in result.items() if key != "ok"}
