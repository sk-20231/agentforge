"""
Pytest fixtures for agentforge tests.
Use a temporary directory for memory so tests don't touch real user data.
"""
import os
import pytest


@pytest.fixture(autouse=True)
def _redirect_runtime_paths(tmp_path, monkeypatch):
    """Point every runtime data path at ``tmp_path`` so the suite has no side effects (issue #43).

    Each of these config values is read once at import time and bound as a
    module-level name in the CONSUMING module (``from agentforge.config import X``),
    so patching ``agentforge.config`` or the environment variable has no effect
    here — the consumer's own attribute is what the code reads. Same set (and the
    same reason) as the save/restore dance in ``redteam_fullstack._isolated_state``,
    plus the corpus file that one misses.

    Without this, any test reaching a code path that logs — ``main.resume_agent``
    emits spans that ``tests/test_approval.py`` does not patch — appends fake trace
    records to the developer's real ``agent_logs.jsonl``. That polluted the log with
    fixture trace ids and sent a real Step 21b.1 debugging session down a dead end.

    Tests that need their own path still win: they patch the same attributes from
    inside the test body, which runs after this fixture.
    """
    # Deliberately NOT "memory" — the opt-in ``temp_memory_dir`` fixture below claims
    # that name under the same tmp_path, and both creating it collides on Windows.
    memory_dir = tmp_path / "agent_memory"
    memory_dir.mkdir(exist_ok=True)

    monkeypatch.setattr("agentforge.logger.AGENT_LOG_FILE", str(tmp_path / "agent_logs.jsonl"))
    monkeypatch.setattr("agentforge.memory.semantic.MEMORY_DIR", str(memory_dir))
    monkeypatch.setattr("agentforge.mcp_client.AGENT_TOOL_PINS_FILE", str(tmp_path / "tool_pins.json"))
    monkeypatch.setattr("agentforge.rag.document_store.AGENT_CORPUS_FILE", str(tmp_path / "corpus.json"))


@pytest.fixture(autouse=True)
def _stub_mcp_catalog(monkeypatch):
    """Seed the classifier tool catalog so tests never spawn real MCP servers.

    ``tool_catalog_for_classifier`` lazily primes the catalog via MCP discovery
    (which spawns subprocesses). Pre-seeding ``_TOOL_CATALOG_CACHE`` keeps every
    test hermetic and deterministic. Tests that need a different catalog can
    monkeypatch the same attribute themselves (this fixture just sets a default).
    """
    import agentforge.tools as tools_mod
    monkeypatch.setattr(tools_mod, "_TOOL_CATALOG_CACHE", [
        {"name": "search_wikipedia", "description": "Look up a topic on Wikipedia and return a short summary"},
        {"name": "get_weather", "description": "Get current weather (temperature, conditions, wind) for a city by name"},
        {"name": "get_top_news", "description": "Search HackerNews for recent top stories on a topic"},
    ])


@pytest.fixture(autouse=True)
def _guardrail_off_by_default(monkeypatch):
    """Keep the gap-E content guardrail OUT of unrelated tests (hermeticity).

    The guardrail runs inside ``gw.call()`` on untrusted tool output. With the
    ungated default model installed, ANY test that dispatches an untrusted call
    would otherwise download + run the real injection classifier — slow, non-
    deterministic, and it can change a sibling test's result (e.g. an injection-
    shaped spotlight payload gets blocked). So we disable the guardrail by default;
    the tests that actually target it (tests/test_guardrail.py) re-enable it, and
    the live contract test calls the engine directly (not via the gateway flag).
    """
    monkeypatch.setattr("agentforge.mcp_client.AGENT_GUARDRAIL_ENABLED", False)


@pytest.fixture
def temp_memory_dir(tmp_path, monkeypatch):
    """Use a temporary directory for MEMORY_DIR so tests don't modify real memory files."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(exist_ok=True)
    monkeypatch.setattr("agentforge.memory.semantic.MEMORY_DIR", str(memory_dir))
    monkeypatch.setattr("agentforge.memory.semantic.os.makedirs", lambda *a, **k: None)
    return memory_dir
