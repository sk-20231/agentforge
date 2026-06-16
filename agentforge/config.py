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
# OUTPUT: redact structured PII from the agent's final reply before it is returned
# (local regex; see agentforge.output_guardrail). Default-on CORE set = secrets / email
# / SSN / credit-card (rarely legitimate in a reply). The AGGRESSIVE flag additionally
# redacts IP + phone, which false-positive on legitimate technical answers, so they are
# OPT-IN. (ML-grade NER for names/addresses + toxicity = a ticketed LLM Guard upgrade.)
AGENT_OUTPUT_GUARDRAIL_ENABLED = _env_flag("AGENT_OUTPUT_GUARDRAIL_ENABLED", True)
AGENT_OUTPUT_GUARDRAIL_AGGRESSIVE = _env_flag("AGENT_OUTPUT_GUARDRAIL_AGGRESSIVE", False)

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
    "fetch":     {"command": "uvx", "args": ["--from", "mcp-server-fetch==2026.6.4", "mcp-server-fetch"], "trusted": False},
}
