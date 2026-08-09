"""
Configuration for the agent. Reads from environment variables with sensible defaults.
Optional: create a .env file or set env vars (OPENAI_API_KEY, OPENAI_MODEL, etc.).
"""
import os
import sys

# Optional: load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# OpenAI
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # None = use default

# Paths
AGENT_MEMORY_DIR = os.environ.get("AGENT_MEMORY_DIR", "memory")
AGENT_LOG_FILE = os.environ.get("AGENT_LOG_FILE", "agent_logs.jsonl")
# Single JSON file where the document corpus (chunks + embeddings) is stored for RAG.
AGENT_CORPUS_FILE = os.environ.get("AGENT_CORPUS_FILE", "corpus.json")
# Tool "pins": baseline fingerprints of each untrusted MCP tool's definition,
# recorded on first sight (trust-on-first-use). The gateway compares against these
# on later turns to detect rug pulls (a tool's definition silently changing after
# we trusted it). Runtime data — gitignored. (Step 17e gap C.)
AGENT_TOOL_PINS_FILE = os.environ.get("AGENT_TOOL_PINS_FILE", "tool_pins.json")

# Maximum estimated tokens allowed in conversation history before trimming.
# Conservative default leaves room for system prompts + RAG chunks (which can
# consume 1,000–2,000 tokens on DOCS_QA calls) within most model context windows.
HISTORY_TOKEN_BUDGET = int(os.environ.get("HISTORY_TOKEN_BUDGET", "2000"))
# Hard cap on the compaction running-summary length (tokens) — see the
# AGENT_CONTEXT_COMPACTION_ENABLED block below (defined after _env_flag). Also
# reserved out of the history budget so summary + kept-recent turns still fit.
COMPACTION_SUMMARY_MAX_TOKENS = int(os.environ.get("AGENT_COMPACTION_SUMMARY_MAX_TOKENS", "256"))

# ReAct observation compression threshold, in characters (issue #8). A tool
# observation longer than this is compressed by one query-focused LLM call
# before it enters the loop's message history — otherwise every later step
# re-sends the raw chunk (cost grows per step, attention degrades). Small
# observations pass through untouched; 0 disables compression entirely.
REACT_OBS_COMPRESS_THRESHOLD = int(os.environ.get("REACT_OBS_COMPRESS_THRESHOLD", "2500"))

# Content guardrail (Step 17e gap E) — a meaning-reading injection/jailbreak
# classifier (a local HuggingFace model, run directly via `transformers`; ProtectAI
# DeBERTa by default, Meta Prompt Guard 2 optional) layered ON TOP of the gateway's
# deterministic guards. Those check the *form* of text (control chars, HTML, SSRF
# URLs, tool fingerprints); this reads its *intent*. It runs inside gw.call() on
# untrusted tool OUTPUT, before the nonce wrap.
#
# Why the model directly and not the LlamaFirewall framework: LlamaFirewall is the
# production wrapper, but it hard-depends on codeshield→semgrep, which has no native
# Windows build, and it drags in openai/typer we don't use. The classifier is just a
# HuggingFace model, so we load it straight from `transformers` — same capability,
# smaller + Windows-clean footprint, and a smaller egress surface (better for the
# no-data-egress rule). See workspace memory feedback_tool_egress_safety.
#
# Egress posture: the model runs FULLY LOCALLY (no API calls). `transformers` /
# `torch` are an OPTIONAL dependency — imported lazily; if absent the guardrail
# no-ops and the agent runs unchanged.
def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean env var ('1/true/yes' = True), falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

# CONTEXT ENGINEERING — compaction (Step 18a). When history exceeds the budget,
# the default (Step 2) behaviour was to DELETE the oldest turns outright, losing
# their information forever. With compaction ON, those oldest turns are instead
# summarised into a single running summary message (names / IDs / decisions /
# open tasks preserved) that rides at the front of the history — so the agent
# keeps the gist of the whole conversation, not just the last few turns.
# Compaction only fires when history is OVER budget (zero added cost on the
# common path) and FAILS SAFE: if the summary LLM call errors or returns empty,
# it degrades to the original delete-oldest trim (never worse than Step 2).
# (Cap on the summary length lives in COMPACTION_SUMMARY_MAX_TOKENS, above.)
AGENT_CONTEXT_COMPACTION_ENABLED = _env_flag("AGENT_CONTEXT_COMPACTION_ENABLED", True)

# Master switch. On by default, but a no-op unless `transformers`/`torch` are installed.
AGENT_GUARDRAIL_ENABLED = _env_flag("AGENT_GUARDRAIL_ENABLED", True)
# Which local injection classifier to load (a HuggingFace repo id, downloaded once,
# run locally). Default: ProtectAI's DeBERTa-v3 prompt-injection model — UNGATED
# (Apache-2.0), so no access-approval queue. To use Meta's Prompt Guard 2 instead
# (gated: requires approved HF access), set this to
# "meta-llama/Llama-Prompt-Guard-2-86M". The label handling in guardrail.py is
# model-agnostic (it keys off SAFE/BENIGN vs INJECTION/MALICIOUS), so either works.
AGENT_GUARDRAIL_MODEL_ID = os.environ.get(
    "AGENT_GUARDRAIL_MODEL_ID", "ProtectAI/deberta-v3-base-prompt-injection-v2"
)
# Malicious-probability cutoff at/above which output is treated as an attack.
# Raise it to cut false positives, lower it to be stricter (Meta suggests tuning).
AGENT_GUARDRAIL_THRESHOLD = float(os.environ.get("AGENT_GUARDRAIL_THRESHOLD", "0.5"))
# When the scanner CANNOT run (model missing / load or scan error): fail OPEN
# (allow the output through, log loudly) by default — the nonce wrap + HITL gate
# remain as backstops, and a flaky model load shouldn't brick every tool call.
# Set true to fail CLOSED (withhold the output) for higher-security deployments.
AGENT_GUARDRAIL_FAIL_CLOSED = _env_flag("AGENT_GUARDRAIL_FAIL_CLOSED", False)
# Scan first-party (trusted) server output too? Default false: only untrusted
# third-party output is scanned, the same scope as the SSRF URL guard.
AGENT_GUARDRAIL_SCAN_TRUSTED = _env_flag("AGENT_GUARDRAIL_SCAN_TRUSTED", False)

# --- Guardrail PLACEMENT POINTS beyond tool-call output (issue #22) ----------
# Gap E above guards the tool-call OUTPUT point. These three flags extend the same
# meaning-level classifier (and, for output, LLM Guard) to the other points of the
# four-placement-points model: INPUT / RETRIEVAL / OUTPUT. Each defaults ON but is a
# no-op unless the underlying model deps are installed (fail-open), exactly like
# gap E — so a fresh clone without `transformers`/LLM Guard runs unchanged.
#
# INPUT: scan the user's message for prompt-injection / jailbreak at the run_agent
# entry, before intent classification. NOTE this is the *direct*-injection guard and
# the user is also the legitimate command channel, so it false-positives on role-play
# ("act as ...") — the deepset FP class. Tune with AGENT_GUARDRAIL_THRESHOLD.
AGENT_INPUT_GUARDRAIL_ENABLED = _env_flag("AGENT_INPUT_GUARDRAIL_ENABLED", True)
# RETRIEVAL: scan each RAG chunk at INGEST and skip flagged ones (defense-in-depth).
# The retrieval-time spotlight WRAP of chunks is unconditional (it is just correct),
# so this flag governs only the ingest-time classifier scan.
AGENT_RAG_GUARDRAIL_ENABLED = _env_flag("AGENT_RAG_GUARDRAIL_ENABLED", True)
# OUTPUT: redact PII from the agent's final reply before it is returned (see
# agentforge.output_guardrail). Default-on CORE set = secrets / email / SSN /
# credit-card / PERSON / ADDRESS. The AGGRESSIVE flag additionally redacts IP + phone,
# which false-positive on legitimate technical answers, so they are OPT-IN.
AGENT_OUTPUT_GUARDRAIL_ENABLED = _env_flag("AGENT_OUTPUT_GUARDRAIL_ENABLED", True)
AGENT_OUTPUT_GUARDRAIL_AGGRESSIVE = _env_flag("AGENT_OUTPUT_GUARDRAIL_AGGRESSIVE", False)
# ENGINE selects the detector (issue #24). "auto" = use the Presidio + GLiNER ML engine
# if the optional `[pii]` deps are installed (catches names/addresses via NER + checksum-
# validated cards), else the always-available zero-dep regex. "regex" forces the regex
# engine; "presidio" prefers Presidio (the difference from "auto" is intent, not behaviour
# — tests use it to assert the Presidio path). The regex layer is the FAIL-SAFE for both
# Presidio modes: a load failure or a mid-scan error degrades to regex, never to no guard.
AGENT_OUTPUT_GUARDRAIL_ENGINE = os.environ.get("AGENT_OUTPUT_GUARDRAIL_ENGINE", "auto").lower()
# GLiNER NER model for the Presidio engine. Default is the ungated, Apache-2.0,
# commercial-OK PII model (person / address / etc.). Pinned so we never silently pull a
# CC-BY-NC variant. Local inference only; weights are a one-time HuggingFace download.
AGENT_OUTPUT_GUARDRAIL_MODEL_ID = os.environ.get(
    "AGENT_OUTPUT_GUARDRAIL_MODEL_ID", "urchade/gliner_multi_pii-v1"
)
# Confidence cutoff for the Presidio engine: detections below this score are dropped
# (filters the low-confidence NER/pattern noise observed at <0.1).
AGENT_OUTPUT_GUARDRAIL_THRESHOLD = float(
    os.environ.get("AGENT_OUTPUT_GUARDRAIL_THRESHOLD", "0.4")
)

# --- MODEL ROUTING (Step 28) -------------------------------------------------
# A routing layer that sends each request to a model sized to its difficulty:
# the easy majority to a small/cheap model, the hard minority to a frontier
# model. The difficulty estimator is FREE — we reuse the intent label that
# classify_intent() already produces every turn (see agentforge/router.py).
#
# Two tiers. FRONTIER DEFAULTS TO THE SAME MODEL AS SMALL, so a fresh clone's
# cost and behaviour are UNCHANGED until you deliberately set a real frontier
# model (e.g. AGENT_MODEL_FRONTIER=gpt-4o). That is the backward-compat
# guarantee: routing is live and logged, but a no-op cost-wise out of the box.
# Small defaults to OPENAI_MODEL so the existing single-model default carries.
AGENT_MODEL_ROUTING_ENABLED = _env_flag("AGENT_MODEL_ROUTING_ENABLED", True)
MODEL_TIER_SMALL = os.environ.get("AGENT_MODEL_SMALL", OPENAI_MODEL)
MODEL_TIER_FRONTIER = os.environ.get("AGENT_MODEL_FRONTIER", MODEL_TIER_SMALL)
# Which intents are "hard" and escalate to the frontier tier. REACT is multi-step
# reasoning/planning — the one intent the roadmap calls genuinely hard. Everything
# else (single tool-calls, RAG answers, memory, plain answers) stays on small.
# Comma-separated env override lets you widen the set (e.g. "REACT,DOCS_QA")
# without a code change.
ROUTING_HARD_INTENTS = frozenset(
    i.strip().upper()
    for i in os.environ.get("AGENT_ROUTING_HARD_INTENTS", "REACT").split(",")
    if i.strip()
)


# MCP servers the agent connects to at runtime to discover and call tools.
#
# Follows the cross-vendor standard "mcpServers" shape (Claude Desktop / Cursor /
# VS Code): a named map where each entry declares HOW TO LAUNCH the server and
# whether we trust it. Tool names are NOT listed here — they are discovered via
# tools/list at runtime.
#
#   command : executable to spawn the server (stdio transport)
#   args    : arguments passed to that executable, in order
#   env     : extra environment variables for THIS server only (optional;
#             merged over the inherited default environment)
#   trusted : True for our own first-party servers; False for a third-party server
#             we did not write. The gateway treats an untrusted server's output as
#             untrusted data and guards its URL arguments against SSRF (Step 17d).
#   requires_approval : when True, every call to this server's tools is gated
#             behind human confirmation (Step 17f — least-privilege / Rule of Two).
#             DEFAULTS TO ``not trusted``: a third-party server is gated unless
#             explicitly relaxed; our first-party read-only servers are not.
#             We own this flag deliberately — MCP's self-declared tool annotations
#             (readOnlyHint / openWorldHint) come from the server and are
#             untrusted input, so they don't get to decide.
#
# Add a new server by adding an entry here — no other code changes needed.
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).parent.parent
_MCP_DIR = _REPO_ROOT / "mcp_servers"
MCP_SERVERS: dict = {
    "wikipedia": {"command": sys.executable, "args": [str(_MCP_DIR / "wikipedia_server.py")], "trusted": True},
    "weather":   {"command": sys.executable, "args": [str(_MCP_DIR / "weather_server.py")],   "trusted": True},
    "news":      {"command": sys.executable, "args": [str(_MCP_DIR / "news_server.py")],       "trusted": True},
    # Third-party: Anthropic reference web-fetch server. Launched via uvx and
    # pinned to a specific PyPI version (supply-chain hygiene). Untrusted: the
    # gateway wraps its output and blocks internal/private URLs (SSRF guard).
    #
    # The `--with "mcp<2"` bound is NOT redundant with the `--from` pin. uvx
    # builds this server its own isolated environment and resolves the server's
    # TRANSITIVE dependencies fresh — so pinning the server alone still let the
    # mcp SDK float to 2.0.0, which renamed McpError -> MCPError and made the
    # pinned server crash on import. Discovery then failed on every turn and the
    # gateway failed open: the agent silently ran without the fetch tool.
    # Pinning what you depend on is not enough when it depends on something else.
    "fetch":     {"command": "uvx",
                  "args": ["--from", "mcp-server-fetch==2026.6.4", "--with", "mcp<2", "mcp-server-fetch"],
                  "trusted": False},
}
