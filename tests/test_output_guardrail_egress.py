"""Egress invariant for the OUTPUT guardrail (issues #22, #24).

The output guardrail must redact PII **fully locally** — either via the stdlib-``re``
regex engine or the Presidio + GLiNER engine, both of which run on-device. It must NEVER
import a remote-LLM / network client or a CLOUD PII SDK (Azure AI Language, AWS
Comprehend, GCP DLP). If anyone later wires in a cloud recognizer that phones home, the
build fails here. Same spirit as tests/test_guardrail_egress.py and test_architecture.py.
"""
import pathlib
import re

import agentforge.output_guardrail as output_guardrail_module

_SRC = pathlib.Path(output_guardrail_module.__file__).read_text(encoding="utf-8")
_IMPORTED_MODULES = {
    m.split(".")[0]
    for m in re.findall(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", _SRC, re.M)
}

# Remote-LLM / network clients AND cloud PII SDKs. Presidio runs locally; a cloud
# recognizer would pull one of these into THIS file — which is exactly what we forbid.
_FORBIDDEN_IMPORTS = {
    "openai", "together", "requests", "httpx", "aiohttp", "urllib3",
    "boto3", "azure", "google",
}


def test_output_guardrail_imports_no_remote_client():
    leaked = _FORBIDDEN_IMPORTS & _IMPORTED_MODULES
    assert not leaked, (
        f"output_guardrail.py must not import a remote/network client — found {leaked}. "
        "PII redaction is local (stdlib re); nothing should call out."
    )


def test_output_guardrail_is_pure_stdlib_regex():
    # The redaction engine is the stdlib `re` module — no model, no service.
    assert "re" in _IMPORTED_MODULES
