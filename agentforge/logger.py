"""
Structured trace logger for agentforge.

Each run_agent call gets a unique trace_id. Every sub-step (intent classification,
retrieval, generation, tool calls) logs with that trace_id so you can reconstruct
the full chain of events for a single user request.

The Span context manager measures wall-clock duration and auto-logs on exit.

compute_latency_percentiles reads the log file and returns P50/P95 latency
breakdowns per span name — useful for identifying bottlenecks.
"""
import json
import time
import uuid
import statistics
from datetime import datetime, timezone

from agentforge.config import AGENT_LOG_FILE


def generate_trace_id() -> str:
    """Short unique ID for grouping all events in one run_agent call."""
    return uuid.uuid4().hex[:12]


def log_event(
    event_type: str,
    payload: dict,
    trace_id: str = None,
    duration_ms: float = None,
):
    """Append a structured JSON event to the log file.

    Args:
        event_type: Name of the event (e.g. "intent_classification", "docs_qa_retrieve").
        payload:    Arbitrary key-value data about the event.
        trace_id:   Optional trace ID linking this event to a run_agent call.
        duration_ms: Optional wall-clock duration of this step in milliseconds.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
    }
    if trace_id is not None:
        record["trace_id"] = trace_id
    if duration_ms is not None:
        record["duration_ms"] = round(duration_ms, 1)
    record["payload"] = payload

    with open(AGENT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class Span:
    """Context manager that measures duration and logs on exit.

    Usage:
        with Span("intent_classification", trace_id=tid) as s:
            result = classify_intent(user_input)
            s.payload = {"intent": result["intent"]}

    On exit, log_event is called with the span name, payload, trace_id,
    and the measured duration_ms.
    """

    def __init__(self, name: str, trace_id: str = None, payload: dict = None):
        self.name = name
        self.trace_id = trace_id
        self.payload = payload or {}
        self._start: float = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self._start) * 1000
        if exc_type is not None:
            self.payload["error"] = str(exc_val)
        log_event(self.name, self.payload, trace_id=self.trace_id, duration_ms=elapsed)
        return False


def compute_latency_percentiles(log_path: str = None) -> dict:
    """Read the log file and compute P50/P95 latency per event type.

    Returns:
        {
            "intent_classification": {"count": 40, "p50_ms": 120.3, "p95_ms": 310.5},
            "docs_qa_retrieve":     {"count": 25, "p50_ms": 350.0, "p95_ms": 820.0},
            ...
        }
    Only events with a duration_ms field are included.
    """
    path = log_path or AGENT_LOG_FILE
    durations: dict[str, list[float]] = {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d = record.get("duration_ms")
                if d is not None:
                    event = record.get("event", "unknown")
                    durations.setdefault(event, []).append(float(d))
    except FileNotFoundError:
        return {}

    result = {}
    for event, times in sorted(durations.items()):
        times.sort()
        n = len(times)
        result[event] = {
            "count": n,
            "p50_ms": round(statistics.median(times), 1),
            "p95_ms": round(times[int(n * 0.95)] if n >= 2 else times[-1], 1),
        }
    return result
