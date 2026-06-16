"""Egress invariant for the OUTPUT guardrail (issue #22).

The output guardrail must redact PII **fully locally** (pure stdlib ``re``) and must
NEVER import a remote-LLM / network client. If anyone later swaps in a cloud PII API
(or a heavy framework that phones home), the build fails here. Same spirit as
tests/test_guardrail_egress.py and tests/test_architecture.py.
"""
import pathlib
import re

import agentforge.output_guardrail as output_guardrail_module

_SRC = pathlib.Path(output_guardrail_module.__file__).read_text(encoding="utf-8")
_IMPORTED_MODULES = {
    m.split(".")[0]
    for m in re.findall(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", _SRC, re.M)
}

_FORBIDDEN_IMPORTS = {"openai", "together", "requests", "httpx", "aiohttp", "urllib3"}


def test_output_guardrail_imports_no_remote_client():
    leaked = _FORBIDDEN_IMPORTS & _IMPORTED_MODULES
    assert not leaked, (
        f"output_guardrail.py must not import a remote/network client — found {leaked}. "
        "PII redaction is local (stdlib re); nothing should call out."
    )


def test_output_guardrail_is_pure_stdlib_regex():
    # The redaction engine is the stdlib `re` module — no model, no service.
    assert "re" in _IMPORTED_MODULES
