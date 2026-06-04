"""
Pytest fixtures for agentforge tests.
Use a temporary directory for memory so tests don't touch real user data.
"""
import os
import pytest


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


@pytest.fixture
def temp_memory_dir(tmp_path, monkeypatch):
    """Use a temporary directory for MEMORY_DIR so tests don't modify real memory files."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    monkeypatch.setattr("agentforge.memory.semantic.MEMORY_DIR", str(memory_dir))
    monkeypatch.setattr("agentforge.memory.semantic.os.makedirs", lambda *a, **k: None)
    return memory_dir
