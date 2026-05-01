"""Shared Groq and JSON parsing helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from email.utils import parsedate_to_datetime
from typing import Any

from groq import AsyncGroq

from config import (
    GROQ_ACCOUNT_COOLDOWN_SECONDS,
    GROQ_MAX_ATTEMPTS,
    GROQ_MODEL,
    GROQ_MODEL_SEQUENCE,
    GROQ_PRODUCTION_MAX_ATTEMPTS,
    GROQ_TEMPERATURE,
    GROQ_TIMEOUT,
    get_groq_api_accounts,
    get_groq_api_key,
)


_client: AsyncGroq | None = None
_client_key: str | None = None
_production_clients: dict[str, AsyncGroq] = {}
_production_cooldowns: dict[str, float] = {}
_production_disabled_keys: set[str] = set()


def _get_client() -> AsyncGroq:
    global _client, _client_key
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env before running LLM agents.")
    if _client is None or _client_key != api_key:
        _client = AsyncGroq(api_key=api_key)
        _client_key = api_key
    return _client


def _key_id(label: str, api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    return f"{label}:{digest}"


def _key_fingerprint(api_key: str) -> str:
    if not api_key:
        return "empty"
    return f"{api_key[:4]}...{api_key[-4:]}:{len(api_key)}"


def _get_production_client(key_id: str, api_key: str) -> AsyncGroq:
    client = _production_clients.get(key_id)
    if client is None:
        client = AsyncGroq(api_key=api_key)
        _production_clients[key_id] = client
    return client


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _headers(exc: Exception) -> Any:
    headers = getattr(exc, "headers", None)
    if headers is None and getattr(exc, "response", None) is not None:
        headers = getattr(exc.response, "headers", None)
    return headers or {}


def _header_get(headers: Any, name: str) -> str | None:
    if not headers:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        return getter(name) or getter(name.lower()) or getter(name.upper())
    return None


def _retry_after_seconds(exc: Exception) -> int | None:
    value = _header_get(_headers(exc), "retry-after")
    if not value:
        return None
    text = str(value).strip()
    try:
        return max(1, int(float(text)))
    except ValueError:
        pass

    total = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([smhd])", text.lower()):
        multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        total += float(amount) * multiplier
    if total > 0:
        return max(1, int(total))

    try:
        parsed = parsedate_to_datetime(text)
        return max(1, int(parsed.timestamp() - time.time()))
    except (TypeError, ValueError, OverflowError):
        return None


def _error_kind(exc: Exception) -> str:
    status = _status_code(exc)
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if isinstance(exc, asyncio.TimeoutError) or "timeout" in name:
        return "timeout"
    if status == 429 or "ratelimit" in name or "rate_limit" in message or "rate limit" in message:
        return "rate_limit"
    if status in {401, 403} or "authentication" in name or "permission" in name:
        return "auth"
    if status is not None and status >= 500:
        return "server_error"
    return "api_error"


def _active_key_counts(now: float) -> tuple[int, int]:
    accounts = get_groq_api_accounts()
    active = 0
    skipped = 0
    for label, api_key in accounts:
        account_id = _key_id(label, api_key)
        if account_id in _production_disabled_keys or now < _production_cooldowns.get(account_id, 0):
            skipped += 1
        else:
            active += 1
    return active, skipped


def parse_json_object(raw: str, source: str) -> dict[str, Any]:
    """Parse model JSON robustly from fenced or extra-text responses."""
    cleaned = re.sub(r"```(?:json)?", "", raw or "", flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("`").strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError(f"[{source}] Could not parse valid JSON from model output:\n{raw}")


async def call_groq(
    messages: list[dict[str, str]],
    *,
    source: str,
    max_tokens: int,
    temperature: float | None = None,
    timeout: float = GROQ_TIMEOUT,
    max_attempts: int | None = None,
) -> str:
    """Call Groq with bounded retries and timeout."""
    last_error: Exception | None = None
    attempts = max(1, max_attempts or GROQ_MAX_ATTEMPTS)
    models = GROQ_MODEL_SEQUENCE or [GROQ_MODEL]
    for attempt in range(attempts):
        model = models[min(attempt, len(models) - 1)]
        try:
            client = _get_client()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=GROQ_TEMPERATURE if temperature is None else temperature,
                ),
                timeout=timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - network/runtime behavior
            last_error = RuntimeError(f"model={model}: {type(exc).__name__}: {str(exc).strip() or repr(exc)}")
            if attempt < attempts - 1:
                await asyncio.sleep(min(2**attempt, 2))
    if last_error is None:
        detail = "unknown error"
    else:
        message = str(last_error).strip()
        detail = f"{type(last_error).__name__}: {message or repr(last_error)}"
    raise RuntimeError(f"[{source}] Groq call failed after {attempts} attempts: {detail}")


async def call_groq_json(
    messages: list[dict[str, str]],
    *,
    source: str,
    max_tokens: int,
    temperature: float | None = None,
    timeout: float = GROQ_TIMEOUT,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    raw = await call_groq(
        messages,
        source=source,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        max_attempts=max_attempts,
    )
    return parse_json_object(raw, source)


async def _call_groq_production_raw(
    messages: list[dict[str, str]],
    *,
    source: str,
    max_tokens: int,
    temperature: float | None = None,
    attempt_timeouts: tuple[float, ...] = (6.0, 4.0),
    backoff: float = 2.0,
) -> str:
    """Call Groq for fast production paths with isolated asymmetric timing and key failover."""
    last_error: Exception | None = None
    accounts = get_groq_api_accounts()
    if not accounts:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env before running LLM agents.")
    models = GROQ_MODEL_SEQUENCE or [GROQ_MODEL]
    timeouts = tuple(attempt_timeouts) or (6.0,)
    max_attempts = max(1, GROQ_PRODUCTION_MAX_ATTEMPTS)
    attempts = 0
    cooldown_skipped_key_ids: set[str] = set()
    attempted_key_ids: set[str] = set()
    last_model = "-"

    for label, api_key in accounts:
        account_id = _key_id(label, api_key)
        now = time.time()
        cooldown_until = _production_cooldowns.get(account_id, 0)
        if account_id in _production_disabled_keys:
            active, skipped = _active_key_counts(now)
            print(
                "groq_failover "
                f"account={label} fingerprint={_key_fingerprint(api_key)} reason=disabled "
                f"active_keys={active} skipped={skipped}"
            )
            continue
        if now < cooldown_until:
            cooldown_skipped_key_ids.add(account_id)
            active, skipped = _active_key_counts(now)
            remaining = max(1, int(cooldown_until - now))
            print(
                "groq_failover "
                f"account={label} fingerprint={_key_fingerprint(api_key)} reason=cooldown "
                f"cooldown={remaining}s active_keys={active} skipped={skipped}"
            )
            continue

        for model_index, model in enumerate(models):
            if attempts >= max_attempts:
                break
            timeout = timeouts[min(model_index, len(timeouts) - 1)]
            attempts += 1
            attempted_key_ids.add(account_id)
            last_model = model
            try:
                client = _get_production_client(account_id, api_key)
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=GROQ_TEMPERATURE if temperature is None else temperature,
                    ),
                    timeout=timeout,
                )
                print(f"groq_summary attempts={attempts} success={label}:{model} fallback_used=false")
                return response.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover - network/runtime behavior
                kind = _error_kind(exc)
                last_error = RuntimeError(
                    f"account={label}: model={model}: timeout={timeout}: "
                    f"{type(exc).__name__}: {str(exc).strip() or repr(exc)}"
                )
                now = time.time()
                if kind == "rate_limit":
                    cooldown = _retry_after_seconds(exc) or GROQ_ACCOUNT_COOLDOWN_SECONDS
                    _production_cooldowns[account_id] = now + cooldown
                    active, skipped = _active_key_counts(now)
                    print(
                        "groq_failover "
                        f"account={label} fingerprint={_key_fingerprint(api_key)} model={model} "
                        f"reason=rate_limit cooldown={cooldown}s active_keys={active} skipped={skipped}"
                    )
                    break
                if kind == "auth":
                    _production_disabled_keys.add(account_id)
                    active, skipped = _active_key_counts(now)
                    print(
                        "groq_failover "
                        f"account={label} fingerprint={_key_fingerprint(api_key)} model={model} "
                        f"reason=auth_disabled active_keys={active} skipped={skipped}"
                    )
                    break

                active, skipped = _active_key_counts(now)
                print(
                    "groq_failover "
                    f"account={label} fingerprint={_key_fingerprint(api_key)} model={model} "
                    f"reason={kind} active_keys={active} skipped={skipped}"
                )
                if attempts < max_attempts and (model_index < len(models) - 1 or label != accounts[-1][0]):
                    await asyncio.sleep(backoff)

        if attempts >= max_attempts:
            break

    if last_error is None:
        detail = "unknown error"
    else:
        message = str(last_error).strip()
        detail = f"{type(last_error).__name__}: {message or repr(last_error)}"
    print(
        "[groq_failover] all_keys_exhausted "
        f"keys_configured={len(accounts)} "
        f"keys_skipped={len(cooldown_skipped_key_ids)} "
        f"keys_attempted={len(attempted_key_ids)} "
        f"last_model={last_model} "
        "falling_back_to=deterministic"
    )
    print(f"groq_summary attempts={attempts} success=none fallback_used=true")
    raise RuntimeError(f"[{source}] Groq production call failed after {attempts} attempts: {detail}")


async def call_groq_production(
    messages: list[dict[str, str]],
    *,
    source: str,
    max_tokens: int,
    temperature: float | None = None,
    attempt_timeouts: tuple[float, ...] = (6.0, 4.0),
    backoff: float = 2.0,
) -> dict[str, Any]:
    raw = await _call_groq_production_raw(
        messages,
        source=source,
        max_tokens=max_tokens,
        temperature=temperature,
        attempt_timeouts=attempt_timeouts,
        backoff=backoff,
    )
    return parse_json_object(raw, source)
