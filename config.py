"""Central configuration for the AI Conflict Mediation System."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"
CHROMA_PERSIST_DIR = DATA_DIR / "conflict_store"

# Make the project-local .env authoritative. Without override=True, a stale
# machine-level GROQ_API_KEY can silently beat the key the user just set here.
load_dotenv(PROJECT_ROOT / ".env", override=True)


# LLM -------------------------------------------------------------------------

def _clean_secret(value: str) -> str:
    cleaned = (value or "").strip().strip('"').strip("'").strip()
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip()
    return cleaned


GROQ_API_KEY = _clean_secret(os.getenv("GROQ_API_KEY", ""))
GROQ_BACKUP_API_KEYS = [
    key
    for key in (_clean_secret(value) for value in os.getenv("GROQ_BACKUP_API_KEYS", "").split(","))
    if key
][:2]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("GROQ_FALLBACK_MODELS", "llama-3.3-70b-versatile").split(",")
    if model.strip()
]
GROQ_MODEL_SEQUENCE = list(dict.fromkeys([GROQ_MODEL, *GROQ_FALLBACK_MODELS]))
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.25"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "900"))
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "12"))
GROQ_MAX_ATTEMPTS = int(os.getenv("GROQ_MAX_ATTEMPTS", "3"))
GROQ_ACCOUNT_COOLDOWN_SECONDS = int(os.getenv("GROQ_ACCOUNT_COOLDOWN_SECONDS", "1200"))
GROQ_PRODUCTION_MAX_ATTEMPTS = int(os.getenv("GROQ_PRODUCTION_MAX_ATTEMPTS", "6"))


# Runtime modes ---------------------------------------------------------------

MODE_DEFAULT = os.getenv("MODE_DEFAULT", "quality")
SUPPORTED_MODES = {"quality", "fast", "fast_production"}


# RAG -------------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_FINAL_K = int(os.getenv("RAG_FINAL_K", "3"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.60"))


# Local classifiers -----------------------------------------------------------

PERSPECTIVE_MODEL = os.getenv(
    "PERSPECTIVE_MODEL",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
)
EMOTION_MODEL = os.getenv(
    "EMOTION_MODEL",
    "j-hartmann/emotion-english-distilroberta-base",
)
TOXIC_BERT_MODEL = os.getenv("TOXIC_BERT_MODEL", "unitary/toxic-bert")

POSITIVE_EMOTIONS = {
    "admiration",
    "amusement",
    "approval",
    "caring",
    "desire",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
}


# Mediation controls ----------------------------------------------------------

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
DETOXIFY_THRESHOLD = float(os.getenv("DETOXIFY_THRESHOLD", "0.40"))
TOXIC_BERT_THRESHOLD = float(os.getenv("TOXIC_BERT_THRESHOLD", "0.50"))
INTENT_CONFIDENCE = float(os.getenv("INTENT_CONFIDENCE", "0.60"))
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "2"))
IDEMPOTENCY_TTL_SECONDS = int(os.getenv("IDEMPOTENCY_TTL_SECONDS", "90"))

CONFLICT_STRATEGY_MAP = {
    "emotional": "validate feelings first, then gently reframe the impact and next step",
    "logical": "clarify evidence, constraints, and tradeoffs without personalizing the disagreement",
    "misunderstanding": "name the communication gap and give both users a non-blaming path forward",
    "value": "identify shared values while respecting that full agreement may not be realistic",
}


# API controls ----------------------------------------------------------------

API_TITLE = "AI Conflict Mediation System"
API_VERSION = "1.0.0"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))


def ensure_data_dirs() -> None:
    """Create runtime data directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


def get_groq_api_key() -> str:
    """Read the current project-local Groq key.

    This intentionally reloads .env so a changed key is picked up after a
    server restart, and so health/debug paths do not rely on stale imports.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    return _clean_secret(os.getenv("GROQ_API_KEY", ""))


def get_groq_api_accounts() -> list[tuple[str, str]]:
    """Return ordered Groq key entries for demo failover.

    The primary key is always tried first. At most two backup keys are used, and
    duplicates are ignored so a copied key does not waste attempts.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    primary = _clean_secret(os.getenv("GROQ_API_KEY", ""))
    backups = [
        key
        for key in (_clean_secret(value) for value in os.getenv("GROQ_BACKUP_API_KEYS", "").split(","))
        if key
    ][:2]

    accounts: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(label: str, key: str) -> None:
        if key and key not in seen:
            accounts.append((label, key))
            seen.add(key)

    add("primary", primary)
    for index, key in enumerate(backups, start=1):
        add(f"backup_{index}", key)
    return accounts


def secret_fingerprint(value: str) -> dict[str, object]:
    """Return non-sensitive shape information for debugging secret loading."""
    if not value:
        return {"loaded": False, "prefix": "", "suffix": "", "length": 0, "looks_like_groq": False}
    return {
        "loaded": True,
        "prefix": value[:4],
        "suffix": value[-4:],
        "length": len(value),
        "looks_like_groq": value.startswith("gsk_"),
    }
