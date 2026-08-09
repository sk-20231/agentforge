"""Output guardrail — redact structured PII from the agent's final reply (issue #22).

WHY this exists (the OUTPUT placement point):
    Gap E guards tool-call OUTPUT; the input and retrieval guards cover the other two
    points. This is the fourth: the agent's OWN final reply. The agent stores user
    facts in memory and pulls in tool / document / web text, so a reply can
    inadvertently carry PII — an email, a card number, a leaked secret. For a
    multi-user product, leaking one user's data in a reply is a real harm. This module
    scans the OUTGOING reply and redacts structured PII before it reaches the user.

TWO ENGINES, one interface (issue #24):
    ``scan_output`` dispatches to one of two detectors behind a single result type:

    1. REGEX (always available, zero-dep): deterministic patterns for *structured* PII —
       high-signal secret/API-key shapes (OpenAI ``sk-``, AWS ``AKIA``, ``Bearer``),
       email, SSN, and credit-card-shaped digit runs. It CANNOT see person names or
       postal addresses (those have no fixed shape), and its card pattern has no Luhn
       check (it over-redacts long digit runs).

    2. PRESIDIO + GLiNER (optional ``[pii]`` extra): the production PII framework. A
       GLiNER zero-shot NER model catches PERSON and ADDRESS (the regex gap); Presidio's
       checksum-validated recognizers handle EMAIL / SSN / CREDIT_CARD (Luhn fixes the
       regex over-redaction); our API-key shapes stay as a custom recognizer. CORE set:
       PERSON, ADDRESS, EMAIL, SSN, CREDIT_CARD, SECRET. IP + phone are OPT-IN
       (``aggressive`` / ``AGENT_OUTPUT_GUARDRAIL_AGGRESSIVE``) — they false-positive on
       legitimate technical answers (``8.8.8.8`` in an answer is not a leak).

    ``AGENT_OUTPUT_GUARDRAIL_ENGINE`` ("auto" | "regex" | "presidio") picks the engine;
    "auto" uses Presidio when the deps are present and **degrades to regex** on any
    load/scan failure — so the guard is never weaker than the regex floor.

EGRESS POSTURE: the regex engine is pure stdlib ``re`` (zero egress). The Presidio
    engine runs **fully locally** — spaCy + a GLiNER model do forward passes on
    downloaded weights, with NO API calls at scan time; the GLiNER weights are a
    one-time HuggingFace download (telemetry disabled). ``[pii]`` is optional, imported
    lazily in one builder, so a fresh clone without it still runs on regex.
    ``tests/test_output_guardrail_egress.py`` statically asserts this module imports no
    remote-LLM / network client, mirroring the gap-E egress invariant.

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

from agentforge.config import (
    AGENT_OUTPUT_GUARDRAIL_AGGRESSIVE,
    AGENT_OUTPUT_GUARDRAIL_ENGINE,
    AGENT_OUTPUT_GUARDRAIL_MODEL_ID,
    AGENT_OUTPUT_GUARDRAIL_THRESHOLD,
)

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


def _scan_regex(text: str, aggressive: bool) -> OutputScanResult:
    """Regex engine: redact *structured* PII. Pure, local, deterministic, zero-dep.

    ``text`` is assumed non-empty; ``aggressive`` adds the IP + phone patterns. This is
    the always-available floor — the dispatcher falls back here if Presidio is absent.
    """
    patterns = _CORE_PATTERNS + (_EXTENDED_PATTERNS if aggressive else [])
    redacted = text
    total = 0
    types_found: List[str] = []
    for label, pattern in patterns:
        # Count matches against the progressively-redacted text so an earlier, more
        # specific redaction removes the span before a broader pattern can re-match it.
        matches = pattern.findall(redacted)
        if matches:
            total += len(matches)
            if label not in types_found:
                types_found.append(label)
            redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return OutputScanResult(redacted, total, types_found)


# --------------------------------------------------------------------------- #
# Presidio + GLiNER engine — the ONLY place that touches the Presidio/ML API.  #
# Mirrors agentforge.guardrail: lazy tri-state cache, fail-safe (degrade to    #
# regex), engine isolated to one builder so swapping it is a one-function edit. #
# --------------------------------------------------------------------------- #

# Presidio entity name -> our redaction label. CORE is on by default; EXTENDED is the
# opt-in (aggressive) set, kept to IP + phone for parity with the regex engine (the
# wider Presidio zoo — IBAN/crypto/passport/etc. — is a deliberate, eval-driven later).
_CORE_ENTITIES = {
    "PERSON": "PERSON",
    "ADDRESS": "ADDRESS",
    "EMAIL_ADDRESS": "EMAIL",
    "US_SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "SECRET": "SECRET",
}
_EXTENDED_ENTITIES = {
    "PHONE_NUMBER": "PHONE",
    "IP_ADDRESS": "IP",
}
# GLiNER (zero-shot NER) owns only the freeform spans regex can't see. Limiting it to
# person + address — rather than generic "location"/"organization" — keeps it from
# redacting ordinary place/company mentions ("Paris", "Microsoft") in a normal answer.
_GLINER_LABEL_MAP = {"person": "PERSON", "name": "PERSON", "address": "ADDRESS"}


def _build_presidio():
    """Build the Presidio (analyzer, anonymizer) pair once. Raises if ``[pii]`` deps
    are missing or the GLiNER model can't load — the caller catches that and degrades
    to regex. Inference is fully local; the only network is the one-time GLiNER weight
    download (telemetry disabled here)."""
    import os
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import GLiNERRecognizer
    from presidio_anonymizer import AnonymizerEngine

    # A small spaCy pipeline supplies tokens/lemmas/context; GLiNER does the NER.
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    })
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())

    # GLiNER for PERSON/ADDRESS, then drop spaCy's own NER so it doesn't also emit
    # (weaker) PERSON/LOCATION detections competing with GLiNER.
    gliner = GLiNERRecognizer(
        model_name=AGENT_OUTPUT_GUARDRAIL_MODEL_ID,
        entity_mapping=_GLINER_LABEL_MAP,
        flat_ner=False, multi_label=True,
        threshold=AGENT_OUTPUT_GUARDRAIL_THRESHOLD, map_location="cpu",
    )
    analyzer.registry.add_recognizer(gliner)
    try:
        analyzer.registry.remove_recognizer("SpacyRecognizer")
    except Exception:  # name varies across versions; not fatal if absent
        logger.debug("output guardrail: no SpacyRecognizer to remove")

    # Our API-key / token shapes have no Presidio built-in — add them as a custom
    # high-confidence pattern recognizer (reuses the regex-engine patterns).
    analyzer.registry.add_recognizer(PatternRecognizer(
        supported_entity="SECRET",
        name="SecretKeyRecognizer",
        patterns=[
            Pattern("openai_key", r"\bsk-[A-Za-z0-9]{20,}\b", 0.9),
            Pattern("aws_access_key", r"\bAKIA[0-9A-Z]{16}\b", 0.9),
            Pattern("bearer_token", r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b", 0.9),
        ],
    ))
    return analyzer, AnonymizerEngine()


_presidio = None
_presidio_state = "uninit"  # uninit -> try once; ready -> use; failed -> stay on regex


def _get_presidio():
    """Return the cached (analyzer, anonymizer), building once. None if unavailable."""
    global _presidio, _presidio_state
    if _presidio_state == "ready":
        return _presidio
    if _presidio_state == "failed":
        return None
    try:
        _presidio = _build_presidio()
        _presidio_state = "ready"
        logger.info("Output guardrail ready (Presidio + GLiNER, local).")
    except Exception as exc:  # missing [pii] deps, model not downloaded, etc.
        _presidio = None
        _presidio_state = "failed"
        logger.warning(
            "Output guardrail Presidio engine unavailable (%s); falling back to regex.",
            exc,
        )
    return _presidio


def reset_engine_cache() -> None:
    """Drop the cached Presidio engine so the next scan rebuilds it. For tests only."""
    global _presidio, _presidio_state
    _presidio = None
    _presidio_state = "uninit"


def _scan_presidio(text: str, aggressive: bool) -> OutputScanResult:
    """Presidio engine: detect via analyzer, redact via anonymizer. Returns our
    :class:`OutputScanResult`. Raises on scan failure (caller degrades to regex)."""
    from presidio_anonymizer.entities import OperatorConfig

    analyzer, anonymizer = _get_presidio()
    entity_map = dict(_CORE_ENTITIES)
    if aggressive:
        entity_map.update(_EXTENDED_ENTITIES)

    results = analyzer.analyze(
        text=text, language="en",
        entities=list(entity_map.keys()),
        score_threshold=AGENT_OUTPUT_GUARDRAIL_THRESHOLD,
    )
    if not results:
        return OutputScanResult(text, 0, [])

    operators = {
        ent: OperatorConfig("replace", {"new_value": f"[REDACTED_{label}]"})
        for ent, label in entity_map.items()
    }
    anonymized = anonymizer.anonymize(
        text=text, analyzer_results=results, operators=operators)

    # Distinct OUR-labels, in order of first appearance (sorted by span start).
    types_found: List[str] = []
    for r in sorted(results, key=lambda x: x.start):
        label = entity_map.get(r.entity_type, r.entity_type)
        if label not in types_found:
            types_found.append(label)
    return OutputScanResult(anonymized.text, len(results), types_found)


def scan_output(text: str, aggressive: bool = None,
                engine: str = None) -> OutputScanResult:
    """Redact PII from ``text`` using the configured engine.

    Dispatches per ``AGENT_OUTPUT_GUARDRAIL_ENGINE``: "auto" prefers Presidio + GLiNER
    (names/addresses + checksum-validated cards) and **degrades to regex** on any
    failure; "regex" forces the regex floor; "presidio" forces Presidio (degrades to
    regex only if the engine truly can't load). ``aggressive`` adds the IP/phone opt-in
    set; when ``None`` it follows ``AGENT_OUTPUT_GUARDRAIL_AGGRESSIVE``. Empty/blank
    text returns unchanged. Never raises and never logs the PII values themselves.

    ``engine`` overrides the configured engine for ONE call; ``None`` (the default)
    keeps the previous behaviour, so every existing caller is unaffected. It exists
    for callers that scan many tiny strings on a hot path and must not pay for a
    model forward pass each time — see ``scan_structured``.
    """
    if not text:
        return OutputScanResult(text or "", 0, [])
    if aggressive is None:
        aggressive = AGENT_OUTPUT_GUARDRAIL_AGGRESSIVE
    engine = (engine or AGENT_OUTPUT_GUARDRAIL_ENGINE).lower()

    if engine in ("auto", "presidio") and _get_presidio() is not None:
        try:
            return _scan_presidio(text, aggressive)
        except Exception as exc:  # any scan-time failure -> never weaker than regex
            logger.warning("Output guardrail Presidio scan failed (%s); using regex.", exc)
    return _scan_regex(text, aggressive)


# ---------------------------------------------------------------------------
# Structured (dict / list) scanning — Step 21b.1
# ---------------------------------------------------------------------------

# Bounds on the walk itself, independent of the per-string cap the caller sets.
# Tool arguments are shallow and small in practice; these stop a pathological
# model-generated value from writing an unbounded record to the trace log.
_STRUCTURED_MAX_ITEMS = 20
_STRUCTURED_MAX_DEPTH = 4


@dataclass
class StructuredScanResult:
    """Result of scanning a structured value (dict / list / scalar) for PII.

    ``value`` has the SAME shape as the input, with every string redacted and
    capped. ``count`` / ``types`` mirror :class:`OutputScanResult` (categories
    only — never the values). ``truncated`` is True when any cap bit: a string
    was cut, a sequence was clipped, or the walk hit its depth limit.
    """
    value: object
    count: int = 0
    types: List[str] = field(default_factory=list)
    truncated: bool = False


def scan_structured(value, aggressive: bool = None, engine: str = "regex",
                    max_chars: int = 500,
                    max_items: int = _STRUCTURED_MAX_ITEMS,
                    max_depth: int = _STRUCTURED_MAX_DEPTH) -> StructuredScanResult:
    """Redact + bound a structured value so it is safe to write to a log.

    WHY the strings and not the serialised blob: redacting a JSON string can drop
    a replacement across a quote and leave the record unparseable. Walking the
    structure and scanning each string VALUE keeps the shape intact by
    construction. Dict KEYS are never scanned — they come from the tool's own
    JSON Schema, so they are vocabulary, not user data.

    WHY ``engine`` defaults to "regex": this runs on a hot path over strings that
    are typically a handful of characters ("Paris"). A GLiNER forward pass per
    call would add latency for almost no yield, and NER precision on 6-character
    strings is exactly where the #27 false-positive tax is worst. The regex floor
    is deterministic, free, and still catches the categories that actually matter
    in a log file — secrets, emails, SSNs, cards. Pass ``engine=None`` to follow
    the configured engine instead.

    Numbers are scanned as text (a card number can arrive as an int, and the
    regex floor only sees strings); a redacted number therefore comes back as a
    string — the shape change is itself the signal. Booleans and None pass
    through untouched.

    ``max_chars <= 0`` disables the per-string cap (the sequence and depth caps
    always apply). Callers that want to skip scanning altogether should not call
    this at all.
    """
    types: List[str] = []
    state = {"count": 0, "truncated": False}

    def note(result: OutputScanResult) -> None:
        state["count"] += result.count
        for t in result.types:
            if t not in types:
                types.append(t)

    def walk(node, depth: int):
        if depth > max_depth:
            state["truncated"] = True
            return "[depth-capped]"

        if isinstance(node, str):
            scanned = scan_output(node, aggressive=aggressive, engine=engine)
            note(scanned)
            text = scanned.redacted_text
            if max_chars > 0 and len(text) > max_chars:
                state["truncated"] = True
                return text[:max_chars] + "...[truncated]"
            return text

        # bool is a subclass of int — check it first so True doesn't become "True".
        if isinstance(node, bool) or node is None:
            return node

        if isinstance(node, (int, float)):
            scanned = scan_output(str(node), aggressive=aggressive, engine=engine)
            if scanned.count:
                note(scanned)
                return scanned.redacted_text
            return node

        if isinstance(node, dict):
            out = {}
            for i, (k, v) in enumerate(node.items()):
                if i >= max_items:
                    state["truncated"] = True
                    break
                out[str(k)] = walk(v, depth + 1)
            return out

        if isinstance(node, (list, tuple)):
            out = []
            for i, v in enumerate(node):
                if i >= max_items:
                    state["truncated"] = True
                    break
                out.append(walk(v, depth + 1))
            return out

        # Anything else (an object the model somehow produced) becomes its repr,
        # scanned and capped like any other string rather than trusted verbatim.
        return walk(repr(node), depth)

    scanned_value = walk(value, 0)
    return StructuredScanResult(scanned_value, state["count"], types,
                                state["truncated"])
