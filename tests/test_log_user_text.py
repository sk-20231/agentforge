"""User text stays out of the trace log unless AGENT_LOG_USER_TEXT is on.

The trace log (agent_logs.jsonl) is shipped to a log service once the app is
deployed, so by default it must hold only the LENGTH of what the user typed,
never the text. These tests read the real log file (conftest points it at
tmp_path) and check that a marker string the user "typed" never appears in it.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

import agentforge.logger as logger_mod
from agentforge.logger import user_text_fields

SECRET = "jane.doe@example.com"
USER_TEXT = f"my email is {SECRET} and I have asthma"


def _log_text() -> str:
    try:
        with open(logger_mod.AGENT_LOG_FILE, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _events(name: str) -> list[dict]:
    return [r["payload"] for r in map(json.loads, _log_text().splitlines())
            if r.get("event") == name]


@pytest.fixture
def log_text_on(monkeypatch):
    monkeypatch.setattr("agentforge.logger.AGENT_LOG_USER_TEXT", True)


# ---------- the helper ----------

class TestUserTextFields:
    def test_off_by_default_logs_only_the_length(self):
        assert user_text_fields("user_input", USER_TEXT) == {"user_input_chars": len(USER_TEXT)}

    def test_off_handles_none_and_non_strings(self):
        assert user_text_fields("text", None) == {"text_chars": 0}
        assert user_text_fields("text", 12345) == {"text_chars": 5}

    def test_on_logs_the_text_with_pii_redacted(self, log_text_on):
        fields = user_text_fields("user_input", USER_TEXT)
        assert SECRET not in fields["user_input"]
        assert "asthma" in fields["user_input"]   # the regex floor does not catch free text
        assert "EMAIL" in " ".join(fields["user_input_redacted"]).upper()

    def test_on_clean_text_has_no_extra_flags(self, log_text_on):
        assert user_text_fields("query", "what is RAG?") == {"query": "what is RAG?"}

    def test_on_long_text_is_capped_and_flagged(self, log_text_on):
        fields = user_text_fields("text", "x" * (logger_mod.USER_TEXT_MAX_CHARS + 50))
        assert len(fields["text"]) <= logger_mod.USER_TEXT_MAX_CHARS + 20  # cap + marker
        assert fields["text_truncated"] is True

    def test_on_scrub_failure_falls_back_to_length(self, log_text_on):
        with patch("agentforge.output_guardrail.scan_structured",
                   side_effect=RuntimeError("boom")):
            fields = user_text_fields("text", USER_TEXT)
        assert fields == {"text_chars": len(USER_TEXT), "text_scan_failed": True}


# ---------- the call sites, through the real log file ----------

class TestNoUserTextInTheLog:
    def test_trace_start(self):
        from agentforge.main import run_agent
        with patch("agentforge.main._input_guardrail_block", return_value="refused"):
            run_agent("u1", "s1", USER_TEXT)
        (start,) = _events("trace_start")
        assert start["user_input_chars"] == len(USER_TEXT)
        assert start["user_id"] == "u1"
        assert SECRET not in _log_text()

    def test_memory_write(self):
        from agentforge.memory.semantic import store_memory
        with patch("agentforge.memory.semantic.get_embedding", return_value=[0.1, 0.2]):
            store_memory("u1", USER_TEXT)
        (write,) = _events("memory_write")
        assert write == {"text_chars": len(USER_TEXT)}
        assert SECRET not in _log_text()

    def test_memory_write_on(self, log_text_on):
        from agentforge.memory.semantic import store_memory
        with patch("agentforge.memory.semantic.get_embedding", return_value=[0.1, 0.2]):
            store_memory("u1", USER_TEXT)
        (write,) = _events("memory_write")
        assert "asthma" in write["text"]
        assert SECRET not in _log_text()          # still scrubbed when on

    @pytest.mark.parametrize("stream", [False, True])
    @patch("agentforge.rag.qa._client")
    @patch("agentforge.rag.qa.search_docs")
    def test_docs_qa(self, mock_search, mock_client, stream):
        from agentforge.rag.qa import answer_from_docs
        mock_search.return_value = [
            {"id": "c0", "text": "Policy text.", "source": "doc.txt", "score": 0.9}]
        if stream:
            chunk = MagicMock(usage=None)
            chunk.choices = [MagicMock(delta=MagicMock(content="Answer [c0]."))]
            mock_client.chat.completions.create.return_value = iter([chunk])
            "".join(answer_from_docs(USER_TEXT, stream=True))
        else:
            msg = MagicMock(content="Answer [c0].")
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=msg)])
            answer_from_docs(USER_TEXT)
        assert SECRET not in _log_text()
        assert "asthma" not in _log_text()


# ---------- --from-trace asks for the task when the log did not keep it ----------

def test_review_asks_for_the_task_when_the_trace_has_none():
    from agentforge.evaluation import review_draft_interactive
    draft = {"id": "traj_001", "task": "", "expected_tools": ["get_weather"],
             "expected_outcome": "", "expected_substrings": [],
             "difficulty": "unknown", "_source_trace_id": "tid"}
    # id, difficulty, task (blank -> re-asked), task, tools, substrings, outcome, confirm
    answers = iter(["", "", "", "weather in Paris?", "", "", "a weather report", "y"])
    entry = review_draft_interactive(draft, input_fn=lambda _p: next(answers))
    assert entry["task"] == "weather in Paris?"
