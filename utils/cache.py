"""Small in-memory TTL cache for idempotency."""

from __future__ import annotations

import time
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, T]] = {}

    def get(self, key: str) -> Optional[T]:
        self._prune()
        item = self._store.get(key)
        if not item:
            return None
        created_at, value = item
        if time.time() - created_at > self.ttl_seconds:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: T) -> None:
        self._prune()
        self._store[key] = (time.time(), value)

    def _prune(self) -> None:
        now = time.time()
        expired = [k for k, (created_at, _) in self._store.items() if now - created_at > self.ttl_seconds]
        for key in expired:
            self._store.pop(key, None)

