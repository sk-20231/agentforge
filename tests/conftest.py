"""
Pytest fixtures for simple-agent tests.
Use a temporary directory for memory so tests don't touch real user data.
"""
import os
import pytest


@pytest.fixture
def temp_memory_dir(tmp_path, monkeypatch):
    """Use a temporary directory for MEMORY_DIR so tests don't modify real memory files."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    monkeypatch.setattr("agentforge.memory.semantic.MEMORY_DIR", str(memory_dir))
    monkeypatch.setattr("agentforge.memory.semantic.os.makedirs", lambda *a, **k: None)
    return memory_dir
