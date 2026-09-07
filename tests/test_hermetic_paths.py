"""Guard the suite's hermeticity: tests must not touch real runtime data (issue #43).

Before this, running the suite appended fixture trace records to the developer's
actual ``agent_logs.jsonl``. The autouse ``_redirect_runtime_paths`` fixture in
conftest.py redirects every runtime path into ``tmp_path``; these tests fail if
that fixture stops covering one of them, which is the failure mode that would
otherwise go unnoticed until the real log is polluted again.
"""
import json
import os

import agentforge.logger as logger_mod
import agentforge.mcp_client as mcp_client
import agentforge.memory.semantic as semantic
import agentforge.rag.document_store as document_store
from agentforge.config import (
    AGENT_CORPUS_FILE,
    AGENT_LOG_FILE,
    AGENT_TOOL_PINS_FILE,
)


class TestRuntimePathsRedirected:
    """Every runtime path the agent writes to points inside tmp_path, not the repo."""

    def test_log_file_is_redirected(self, tmp_path):
        assert logger_mod.AGENT_LOG_FILE == str(tmp_path / "agent_logs.jsonl")

    def test_memory_dir_is_redirected(self, tmp_path):
        assert semantic.MEMORY_DIR == str(tmp_path / "agent_memory")

    def test_memory_dir_coexists_with_temp_memory_dir_fixture(self, tmp_path, temp_memory_dir):
        """The opt-in fixture still wins, and the two no longer collide on mkdir."""
        assert semantic.MEMORY_DIR == str(temp_memory_dir)
        assert temp_memory_dir == tmp_path / "memory"
        assert (tmp_path / "agent_memory").is_dir()

    def test_tool_pins_file_is_redirected(self, tmp_path):
        assert mcp_client.AGENT_TOOL_PINS_FILE == str(tmp_path / "tool_pins.json")

    def test_corpus_file_is_redirected(self, tmp_path):
        assert document_store.AGENT_CORPUS_FILE == str(tmp_path / "corpus.json")

    def test_redirected_paths_differ_from_configured_defaults(self):
        """The redirect is real, not a coincidence of matching default names."""
        assert logger_mod.AGENT_LOG_FILE != AGENT_LOG_FILE
        assert mcp_client.AGENT_TOOL_PINS_FILE != AGENT_TOOL_PINS_FILE
        assert document_store.AGENT_CORPUS_FILE != AGENT_CORPUS_FILE


class TestLoggingIsContained:
    """log_event / Span write to the temp log, leaving the real one untouched."""

    def test_log_event_writes_to_temp_log(self, tmp_path):
        logger_mod.log_event("hermeticity_probe", {"marker": "issue-43"}, trace_id="tid-probe")

        log_file = tmp_path / "agent_logs.jsonl"
        assert log_file.exists(), "log_event did not write to the redirected path"

        records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert any(r["event"] == "hermeticity_probe" for r in records)

    def test_span_writes_to_temp_log(self, tmp_path):
        with logger_mod.Span("hermeticity_span", trace_id="tid-probe") as s:
            s.payload = {"marker": "issue-43"}

        records = [
            json.loads(line)
            for line in (tmp_path / "agent_logs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        span_records = [r for r in records if r["event"] == "hermeticity_span"]
        assert len(span_records) == 1
        assert "duration_ms" in span_records[0]

    def test_real_log_file_is_not_appended_to(self, tmp_path):
        """The repo-root log must not grow — the concrete symptom reported in #43."""
        real_log = AGENT_LOG_FILE
        before = os.path.getsize(real_log) if os.path.exists(real_log) else None

        logger_mod.log_event("hermeticity_probe", {"marker": "issue-43"})

        after = os.path.getsize(real_log) if os.path.exists(real_log) else None
        assert after == before, f"the suite wrote to the real log at {real_log}"


class TestPerTestPatchesStillWin:
    """A test patching a path itself must override the autouse fixture."""

    def test_test_level_patch_overrides_fixture(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom_logs.jsonl"
        monkeypatch.setattr("agentforge.logger.AGENT_LOG_FILE", str(custom))

        logger_mod.log_event("hermeticity_probe", {"marker": "override"})

        assert custom.exists()
        assert not (tmp_path / "agent_logs.jsonl").exists()
