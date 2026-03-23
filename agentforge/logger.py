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


# ---------------------------------------------------------------------------
# Step 14 — Cost tracking
# ---------------------------------------------------------------------------

# Per-token pricing in USD. Update when model pricing changes.
# Source: https://openai.com/api/pricing/
MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
    "gpt-4o":      {"prompt": 2.50 / 1_000_000, "completion": 10.0 / 1_000_000},
    "gpt-4":       {"prompt": 30.0 / 1_000_000, "completion": 60.0 / 1_000_000},
    "gpt-3.5-turbo": {"prompt": 0.50 / 1_000_000, "completion": 1.50 / 1_000_000},
}


def log_token_usage(
    response,
    operation: str,
    trace_id: str = None,
    model: str = None,
):
    """Extract token usage from an OpenAI response and log it with cost estimate.

    Wrapped in try/except so cost tracking never breaks core agent functionality.

    Args:
        response:  The ChatCompletion response object (must have .usage).
        operation: Human-readable label (e.g. "intent_classification", "docs_qa_generate").
        trace_id:  Optional trace ID for linking to a run_agent call.
        model:     Model name override. If None, reads from response.model.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        model_name = model or str(getattr(response, "model", "unknown"))
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens

        costs = MODEL_COSTS.get(model_name, MODEL_COSTS.get("gpt-4o-mini"))
        cost_usd = (prompt_tokens * costs["prompt"]) + (completion_tokens * costs["completion"])

        log_event("token_usage", {
            "operation": operation,
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost_usd, 6),
        }, trace_id=trace_id)
    except Exception:
        pass


def compute_cost_summary(log_path: str = None) -> dict:
    """Read the log file and aggregate token usage and cost per operation.

    Returns:
        {
            "by_operation": {
                "intent_classification": {"calls": 10, "prompt_tokens": 2000, "completion_tokens": 500, "cost_usd": 0.0006},
                ...
            },
            "total": {"calls": 50, "prompt_tokens": 15000, "completion_tokens": 5000, "cost_usd": 0.012}
        }
    """
    path = log_path or AGENT_LOG_FILE
    ops: dict[str, dict] = {}

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
                if record.get("event") != "token_usage":
                    continue
                p = record.get("payload", {})
                op = p.get("operation", "unknown")
                if op not in ops:
                    ops[op] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
                ops[op]["calls"] += 1
                ops[op]["prompt_tokens"] += p.get("prompt_tokens", 0)
                ops[op]["completion_tokens"] += p.get("completion_tokens", 0)
                ops[op]["cost_usd"] += p.get("cost_usd", 0.0)
    except FileNotFoundError:
        return {"by_operation": {}, "total": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}}

    total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    for v in ops.values():
        total["calls"] += v["calls"]
        total["prompt_tokens"] += v["prompt_tokens"]
        total["completion_tokens"] += v["completion_tokens"]
        total["cost_usd"] += v["cost_usd"]
    total["cost_usd"] = round(total["cost_usd"], 6)
    for v in ops.values():
        v["cost_usd"] = round(v["cost_usd"], 6)

    return {"by_operation": dict(sorted(ops.items())), "total": total}


def compute_trace_cost(trace_id: str, log_path: str = None) -> dict:
    """Get the total cost and token breakdown for a single trace (one run_agent call).

    Returns:
        {"calls": 3, "prompt_tokens": 800, "completion_tokens": 200,
         "cost_usd": 0.0003, "operations": ["intent_classification", "docs_qa_generate", ...]}
    """
    path = log_path or AGENT_LOG_FILE
    result = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "operations": []}

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
                if record.get("event") != "token_usage":
                    continue
                if record.get("trace_id") != trace_id:
                    continue
                p = record.get("payload", {})
                result["calls"] += 1
                result["prompt_tokens"] += p.get("prompt_tokens", 0)
                result["completion_tokens"] += p.get("completion_tokens", 0)
                result["cost_usd"] += p.get("cost_usd", 0.0)
                result["operations"].append(p.get("operation", "unknown"))
    except FileNotFoundError:
        pass

    result["cost_usd"] = round(result["cost_usd"], 6)
    return result


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
