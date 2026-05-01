"""Text normalization and request identity helpers."""

from __future__ import annotations

import hashlib
import html
import re


_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def sanitize_text(value: str) -> str:
    """Strip simple HTML/script markup and collapse whitespace."""
    unescaped = html.unescape(value or "")
    no_tags = _TAG_RE.sub(" ", unescaped)
    return _SPACE_RE.sub(" ", no_tags).strip()


def normalize_for_hash(value: str) -> str:
    return sanitize_text(value).lower()


def stable_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(normalize_for_hash(part).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def request_hash(conversation_id: str, turn: int, text_a: str, text_b: str, mode: str) -> str:
    return stable_hash(conversation_id, str(turn), text_a, text_b, mode)


def case_hash(text_a: str, text_b: str) -> str:
    return stable_hash(text_a, text_b)

