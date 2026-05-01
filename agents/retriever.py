"""ChromaDB RAG retriever and memory writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    RAG_MIN_SCORE,
    RAG_TOP_K,
    SCENARIOS_PATH,
    ensure_data_dirs,
)
from models.schemas import RetrievedCase, RetrieverOutput
from utils.text import case_hash


_client: chromadb.PersistentClient | None = None
_collection: Any | None = None


def _get_collection():
    global _client, _collection
    ensure_data_dirs()
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    _collection = _client.get_or_create_collection(
        name="conflict_cases",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _build_document(
    text_a: str,
    text_b: str,
    conflict_type: str = "",
    resolution_strategy: str = "",
    response_a: str = "",
    response_b: str = "",
) -> str:
    return (
        f"User A: {text_a.strip()} | "
        f"User B: {text_b.strip()} | "
        f"Conflict: {conflict_type} | "
        f"Strategy: {resolution_strategy} | "
        f"Resolution A: {response_a[:180]} | "
        f"Resolution B: {response_b[:180]}"
    )


def _build_query(text_a: str, text_b: str) -> str:
    return f"User A: {text_a.strip()} | User B: {text_b.strip()}"


def _similarity_from_distance(distance: float) -> float:
    # Chroma cosine distance is usually 0..2. This maps it to a stable 0..1 score.
    return round(max(0.0, min(1.0, 1.0 - (distance / 2.0))), 3)


def seed_from_scenarios(path: Path = SCENARIOS_PATH) -> int:
    collection = _get_collection()
    if not path.exists():
        print(f"[retriever] scenarios file missing: {path}")
        return 0

    with path.open("r", encoding="utf-8") as file:
        scenarios = json.load(file)

    existing_ids = set()
    existing_count = collection.count()
    if existing_count:
        existing = collection.get(include=[])
        existing_ids = set(existing.get("ids", []))

    docs: list[str] = []
    metas: list[dict[str, Any]] = []
    ids: list[str] = []

    for scenario in scenarios:
        sid = scenario.get("id") or f"seed_{case_hash(scenario['text_a'], scenario['text_b'])}"
        if sid in existing_ids:
            continue

        resolved = scenario.get("resolved_example", {})
        response_a = resolved.get("response_a", scenario.get("response_a", ""))
        response_b = resolved.get("response_b", scenario.get("response_b", ""))
        conflict_type = scenario.get("conflict_type", "unknown")
        strategy = scenario.get("resolution_strategy", "")

        docs.append(
            _build_document(
                scenario["text_a"],
                scenario["text_b"],
                conflict_type,
                strategy,
                response_a,
                response_b,
            )
        )
        metas.append(
            {
                "source": "seed",
                "case_hash": case_hash(scenario["text_a"], scenario["text_b"]),
                "conflict_type": conflict_type,
                "resolution_strategy": strategy,
                "response_a": response_a,
                "response_b": response_b,
                "text_a": scenario["text_a"],
                "text_b": scenario["text_b"],
            }
        )
        ids.append(sid)

    if docs:
        collection.add(documents=docs, metadatas=metas, ids=ids)
        print(f"[retriever] Seeded {len(docs)} scenarios")
    else:
        print(f"[retriever] Already seeded ({existing_count} records)")
    return len(docs)


def retrieve(text_a: str, text_b: str) -> RetrieverOutput:
    collection = _get_collection()
    if collection.count() == 0:
        seed_from_scenarios()

    count = collection.count()
    if count == 0:
        return RetrieverOutput(retrieved_cases=[], top_score=None, rag_used=False)

    results = collection.query(
        query_texts=[_build_query(text_a, text_b)],
        n_results=min(RAG_TOP_K, count),
        include=["metadatas", "distances"],
    )

    cases: list[RetrievedCase] = []
    top_score = None
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for meta, distance in zip(metadatas, distances):
        similarity = _similarity_from_distance(float(distance))
        top_score = similarity if top_score is None else max(top_score, similarity)
        if similarity < RAG_MIN_SCORE:
            continue
        cases.append(
            RetrievedCase(
                conflict_type=meta.get("conflict_type", "unknown"),
                text_a=meta.get("text_a", ""),
                text_b=meta.get("text_b", ""),
                resolution_strategy=meta.get("resolution_strategy", ""),
                response_a=meta.get("response_a", ""),
                response_b=meta.get("response_b", ""),
                relevance_score=similarity,
            )
        )

    return RetrieverOutput(
        retrieved_cases=cases,
        top_score=top_score,
        rag_used=bool(cases),
    )


def store_case(
    *,
    conversation_id: str,
    text_a: str,
    text_b: str,
    conflict_type: str,
    resolution_strategy: str,
    response_a: str,
    response_b: str,
) -> bool:
    """Store approved resolutions, deduped by normalized input pair hash."""
    collection = _get_collection()
    digest = case_hash(text_a, text_b)
    entry_id = f"mem_{digest[:24]}"
    existing = collection.get(ids=[entry_id], include=[])
    if existing.get("ids"):
        print(f"[retriever] Memory duplicate skipped: {entry_id}")
        return False

    metadata = {
        "source": "memory",
        "conversation_id": conversation_id,
        "case_hash": digest,
        "conflict_type": conflict_type,
        "resolution_strategy": resolution_strategy,
        "response_a": response_a,
        "response_b": response_b,
        "text_a": text_a,
        "text_b": text_b,
    }
    collection.add(
        documents=[_build_document(text_a, text_b, conflict_type, resolution_strategy, response_a, response_b)],
        metadatas=[metadata],
        ids=[entry_id],
    )
    print(f"[retriever] Stored memory case: {entry_id}")
    return True


def warmup() -> dict[str, Any]:
    seeded = seed_from_scenarios()
    collection = _get_collection()
    return {"seeded": seeded, "collection_count": collection.count()}


def status() -> dict[str, Any]:
    try:
        collection = _get_collection()
        return {"ready": True, "count": collection.count()}
    except Exception as exc:  # pragma: no cover - runtime dependency behavior
        return {"ready": False, "error": str(exc)}

