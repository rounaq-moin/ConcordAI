"""Minimal in-process observability counters."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Metrics:
    total_requests: int = 0
    total_latency: float = 0.0
    total_retries: int = 0
    fallback_count: int = 0
    memory_writes: int = 0
    resolvability_counts: dict[str, int] = field(default_factory=dict)


_metrics = Metrics()
_lock = Lock()


def record_request(
    *,
    latency: float,
    retries: int,
    fallback_used: bool,
    memory_written: bool,
    resolvability: str,
) -> None:
    with _lock:
        _metrics.total_requests += 1
        _metrics.total_latency += latency
        _metrics.total_retries += retries
        if fallback_used:
            _metrics.fallback_count += 1
        if memory_written:
            _metrics.memory_writes += 1
        _metrics.resolvability_counts[resolvability] = _metrics.resolvability_counts.get(resolvability, 0) + 1


def snapshot() -> dict:
    with _lock:
        total = max(_metrics.total_requests, 1)
        return {
            "total_requests": _metrics.total_requests,
            "avg_latency": round(_metrics.total_latency / total, 3),
            "retry_rate": round(_metrics.total_retries / total, 3),
            "fallback_rate": round(_metrics.fallback_count / total, 3),
            "memory_writes": _metrics.memory_writes,
            "resolvability_counts": dict(_metrics.resolvability_counts),
        }

