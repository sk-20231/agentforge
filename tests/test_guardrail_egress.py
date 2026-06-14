"""Egress invariant for the content guardrail (Step 17e gap E).

The guardrail must run the Prompt Guard 2 classifier **fully locally** via
``transformers`` and must NEVER import a remote-LLM client (openai / together /
raw HTTP). Adopting any third-party tool, we never let work data leave the machine
(see workspace memory feedback_tool_egress_safety).

This is a *static source* check — it reads ``guardrail.py`` and inspects its imports.
It needs neither ``transformers``/``torch`` installed nor any model downloaded, so
it runs in CI. If anyone later wires a remote model client, the build fails here.
Same spirit as tests/test_architecture.py.
"""
import pathlib
import re

import agentforge.guardrail as guardrail_module
from agentforge.config import AGENT_GUARDRAIL_MODEL_ID

_SRC = pathlib.Path(guardrail_module.__file__).read_text(encoding="utf-8")
# Top-level module of every `import X` / `from X import ...` (including the lazy,
# indented imports inside functions). Prose/docstring mentions don't count — only
# real import statements, which is what would actually load a client.
_IMPORTED_MODULES = {
    m.split(".")[0]
    for m in re.findall(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", _SRC, re.M)
}

# Clients that would send text to a remote LLM / make network calls of their own.
_FORBIDDEN_IMPORTS = {"openai", "together", "requests", "httpx", "aiohttp", "urllib3"}


def test_guardrail_imports_no_remote_llm_client():
    leaked = _FORBIDDEN_IMPORTS & _IMPORTED_MODULES
    assert not leaked, (
        f"guardrail.py must not import a remote-LLM/network client — found {leaked}. "
        "Prompt Guard 2 runs locally via transformers; nothing should call out."
    )


def test_guardrail_loads_the_local_transformers_model():
    # The local inference path must be present (not a remote API call).
    assert "AutoModelForSequenceClassification" in _SRC
    assert "transformers" in _IMPORTED_MODULES


def test_default_model_is_a_local_repo_id_not_a_remote_endpoint():
    # A HuggingFace repo id ("org/model", downloaded once and run locally), never a
    # remote URL/endpoint that would imply calling out for inference.
    assert "://" not in AGENT_GUARDRAIL_MODEL_ID
    assert "/" in AGENT_GUARDRAIL_MODEL_ID
    # Sanity: the vetted local injection classifiers we support by default.
    assert AGENT_GUARDRAIL_MODEL_ID in (
        "ProtectAI/deberta-v3-base-prompt-injection-v2",
        "meta-llama/Llama-Prompt-Guard-2-86M",
        "meta-llama/Llama-Prompt-Guard-2-22M",
    )
