"""Normalize the model's ``reply`` field to display text (issue #49).

The ACT and REACT pipelines ask the model for JSON with a ``reply`` string.
``response_format={"type": "json_object"}`` guarantees valid JSON, not field
types, so a model sometimes returns ``reply`` as an object or a list. Passed
through raw, that breaks three things downstream:

- the front-end shows the Python repr of a dict;
- the output guardrail (``main._scan_output``) skips non-strings, so PII
  inside the reply is never redacted;
- the front-end stores the reply in the conversation history, and every
  later turn that sends history fails with an API 400 (``content`` must be
  a string).

So every pipeline reads ``reply`` through :func:`reply_text`, which always
returns a string.
"""
import json
from typing import Any, Optional

from agentforge.logger import log_event


def reply_text(value: Any, *, source: str, trace_id: Optional[str] = None) -> str:
    """Return ``value`` as display text. A string passes through unchanged.

    ``None`` (a missing or null reply) becomes ``""``. A dict or list becomes
    indented JSON, which keeps the model's content readable instead of
    dropping it. Anything else becomes ``str(value)``. Every non-string is
    logged with its TYPE only — never the value, which can hold user data.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    log_event("reply_not_string", {"source": source, "type": type(value).__name__},
              trace_id=trace_id)
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False)
    return str(value)
