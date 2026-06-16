"""Output guardrail — redact structured PII from the agent's final reply (issue #22).

WHY this exists (the OUTPUT placement point):
    Gap E guards tool-call OUTPUT; the input and retrieval guards cover the other two
    points. This is the fourth: the agent's OWN final reply. The agent stores user
    facts in memory and pulls in tool / document / web text, so a reply can
    inadvertently carry PII — an email, a card number, a leaked secret. For a
    multi-user product, leaking one user's data in a reply is a real harm. This module
    scans the OUTGOING reply and redacts structured PII before it reaches the user.

SCOPE (honest — a guard is only as good as what it actually catches):
    This catches *structured* PII via deterministic regex. The DEFAULT (core) set is
    the spans rarely legitimate in a reply — a few high-signal secret/API-key shapes
    (OpenAI ``sk-``, AWS ``AKIA``, ``Bearer`` tokens), email, SSN, and credit-card-
    shaped digit runs. IP and US-phone are OPT-IN (the ``aggressive`` set / the
    ``AGENT_OUTPUT_GUARDRAIL_AGGRESSIVE`` flag) because they false-positive on
    legitimate technical answers (an answer that mentions ``8.8.8.8`` is not a leak).
    It does NOT do ML-grade named-entity recognition (person names, postal addresses)
    or toxicity — those need a model. That upgrade is ticketed (Presidio / LLM Guard /
    a transformers toxicity classifier), deliberately NOT bundled here.

WHY regex and not LLM Guard (precedent — see workspace memory feedback_tool_egress_safety):
    LLM Guard is the production framework, but it is heavyweight — it drags in Presidio
    + spaCy model downloads + multiple HF models, the same LlamaFirewall-class footprint
    we deliberately rejected for gap E. The standing call is: build the thin, local,
    zero-egress thing now and ticket the framework upgrade. Regex PII is fully local,
    deterministic, has no new dependency, and makes no network call — strictly safe.

EGRESS POSTURE: zero. Pure stdlib ``re``. ``tests/test_output_guardrail_egress.py``
    statically asserts this module imports no remote-LLM / network client, mirroring
    the gap-E egress invariant.

REDACT, don't refuse: the right UX for PII in an otherwise-useful answer is to mask the
    sensitive span (``[REDACTED_EMAIL]``), not to drop the whole reply.

STREAMING LIMITATION (documented): on streaming paths the tokens are already sent, so
    this can only run post-hoc. ``run_agent`` applies it to the NON-streaming string
    returns; scanning streamed output (buffer-then-emit, or non-streaming the PII-risky
    paths) is a ticketed follow-up — same shape as the citation guardrail's limitation.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from agentforge.config import AGENT_OUTPUT_GUARDRAIL_AGGRESSIVE

logger = logging.getLogger(__name__)

# Ordered (label, pattern) pairs. Order matters within the active list: redact the most
# specific / highest-signal spans first so a broad one can't chew up a digit run that is
# really part of a card/SSN. Each pattern is intentionally conservative to limit false
# positives on ordinary prose.
#
# CORE — on by default: spans that are essentially never legitimate in a reply.
_CORE_PATTERNS: List[Tuple[str, "re.Pattern"]] = [
    ("SECRET", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),                 # OpenAI-style key
    ("SECRET", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),                    # AWS access key id
    ("SECRET", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b")),       # bearer token
    ("EMAIL",  re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("SSN",    re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Credit-card-shaped: 13-16 digits in groups of 4 (spaces/dashes) or solid.
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
]
# EXTENDED — opt-in (aggressive): these false-positive on legitimate technical answers
# (a real answer may legitimately contain an IP address or a phone number).
_EXTENDED_PATTERNS: List[Tuple[str, "re.Pattern"]] = [
    # US phone: optional +1, area code, 7 digits, common separators.
    ("PHONE",  re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("IP",     re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


@dataclass
class OutputScanResult:
    """Result of scanning a reply for PII.

    ``redacted_text`` is the reply with every detected span replaced by
    ``[REDACTED_<TYPE>]``. ``types`` lists the distinct PII categories found (no
    values — never log the actual PII). ``count`` is the number of spans redacted.
    """
    redacted_text: str
    count: int = 0
    types: List[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.count > 0


def scan_output(text: str, aggressive: bool = None) -> OutputScanResult:
    """Redact structured PII from ``text``. Pure, local, deterministic.

    ``aggressive`` adds the IP + phone patterns (which false-positive on legitimate
    technical answers); when ``None`` it follows ``AGENT_OUTPUT_GUARDRAIL_AGGRESSIVE``.
    Returns an :class:`OutputScanResult`. Empty/blank text is returned unchanged with a
    zero count. Never raises and never logs the PII values themselves.
    """
    if not text:
        return OutputScanResult(text or "", 0, [])

    if aggressive is None:
        aggressive = AGENT_OUTPUT_GUARDRAIL_AGGRESSIVE
    patterns = _CORE_PATTERNS + (_EXTENDED_PATTERNS if aggressive else [])

    redacted = text
    total = 0
    types_found: List[str] = []
    for label, pattern in patterns:
        # Count matches against the progressively-redacted text so an earlier, more
        # specific redaction removes the span before a broader pattern can re-match it.
        matches = pattern.findall(redacted)
        if matches:
            n = len(matches)
            total += n
            if label not in types_found:
                types_found.append(label)
            redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return OutputScanResult(redacted, total, types_found)
