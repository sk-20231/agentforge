"""
Unit tests for Step 19 — agent-trajectory evaluation.

Hermetic: the two pure scorers (tool-selection, loop-health) and the trajectory
reader need no API. The completion judge is exercised twice — the deterministic
substring path (no API) and the LLM-judge path (mocked OpenAI), matching the
mocking style in test_evaluation.py.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from agentforge.logger import log_event, reconstruct_trajectory
from agentforge.evaluation import (
    load_trajectory_dataset,
    score_tool_selection,
    score_loop_health,
    score_task_completion,
)


# ---------- score_tool_selection (pure) ----------

class TestScoreToolSelection:
    def test_exact_match_is_correct(self):
        r = score_tool_selection(["get_weather"], ["get_weather"])
        assert r["recall"] == 1.0
        assert r["correct"] is True
        assert r["missing"] == [] and r["unexpected"] == []

    def test_order_does_not_matter(self):
        r = score_tool_selection(["get_weather", "get_top_news"],
                                 ["get_top_news", "get_weather"])
        assert r["recall"] == 1.0
        assert r["correct"] is True

    def test_extra_call_is_flagged_but_recall_full(self):
        r = score_tool_selection(["get_weather"], ["get_weather", "search_wikipedia"])
        assert r["recall"] == 1.0          # all expected tools were called
        assert r["unexpected"] == ["search_wikipedia"]
        assert r["correct"] is False       # strict view: an extra tool was used

    def test_missing_tool_lowers_recall(self):
        r = score_tool_selection(["get_weather", "get_top_news"], ["get_weather"])
        assert r["recall"] == pytest.approx(0.5)
        assert r["missing"] == ["get_top_news"]
        assert r["correct"] is False

    def test_duplicate_calls_dont_inflate(self):
        r = score_tool_selection(["get_weather"], ["get_weather", "get_weather"])
        assert r["recall"] == 1.0
        assert r["correct"] is True        # duplicates of an expected tool are fine

    def test_no_tool_expected_and_none_called(self):
        r = score_tool_selection([], [])
        assert r["recall"] == 1.0
        assert r["correct"] is True

    def test_no_tool_expected_but_one_called(self):
        r = score_tool_selection([], ["get_weather"])
        assert r["recall"] == 1.0          # nothing was required
        assert r["unexpected"] == ["get_weather"]
        assert r["correct"] is False       # but it used a tool it shouldn't have


# ---------- score_loop_health (pure) ----------

class TestScoreLoopHealth:
    def test_clean_termination_is_healthy(self):
        traj = {"terminated_cleanly": True, "steps_taken": 2}
        r = score_loop_health(traj, max_steps=5)
        assert r["healthy"] is True
        assert r["hit_ceiling"] is False
        assert r["steps_taken"] == 2

    def test_hit_ceiling_is_unhealthy(self):
        traj = {"terminated_cleanly": False, "steps_taken": 5}
        r = score_loop_health(traj, max_steps=5)
        assert r["healthy"] is False
        assert r["hit_ceiling"] is True

    def test_missing_fields_default_to_unhealthy(self):
        r = score_loop_health({}, max_steps=5)
        assert r["healthy"] is False
        assert r["steps_taken"] == 0


# ---------- score_task_completion (deterministic path, no API) ----------

class TestScoreTaskCompletionSubstring:
    def test_all_substrings_present(self):
        r = score_task_completion(
            "add 12 and 30", "The answer is 42.", "the number 42",
            expected_substrings=["42"], use_judge=False)
        assert r["completed"] is True
        assert r["method"] == "substring"

    def test_missing_substring(self):
        r = score_task_completion(
            "add 12 and 30", "I am not sure.", "the number 42",
            expected_substrings=["42"], use_judge=False)
        assert r["completed"] is False
        assert "42" in r["reason"]

    def test_case_insensitive(self):
        r = score_task_completion(
            "capital of Japan", "It is TOKYO.", "Tokyo",
            expected_substrings=["tokyo"], use_judge=False)
        assert r["completed"] is True

    def test_deterministic_without_substrings_is_incomplete(self):
        r = score_task_completion(
            "q", "a", "outcome", expected_substrings=None, use_judge=False)
        assert r["completed"] is False
        assert "expected_substrings" in r["reason"]


# ---------- score_task_completion (LLM-judge path, mocked) ----------

class TestScoreTaskCompletionJudge:
    @patch("openai.OpenAI")
    def test_completed_verdict(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(
                content='{"completed": true, "reason": "Answer reports Tokyo weather."}'))
        ]
        r = score_task_completion("weather in Tokyo", "Tokyo is 20C and sunny.",
                                  "current Tokyo weather")
        assert r["completed"] is True
        assert r["method"] == "judge"

    @patch("openai.OpenAI")
    def test_incomplete_verdict(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(
                content='{"completed": false, "reason": "Did not address the task."}'))
        ]
        r = score_task_completion("weather in Tokyo", "I like cats.",
                                  "current Tokyo weather")
        assert r["completed"] is False

    @patch("openai.OpenAI")
    def test_api_error_is_incomplete(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API down")
        r = score_task_completion("q", "a", "outcome")
        assert r["completed"] is False
        assert "Judge error" in r["reason"]


# ---------- reconstruct_trajectory (reads the trace log) ----------

class TestReconstructTrajectory:
    def _write_events(self, tmp_path, records):
        path = tmp_path / "logs.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return str(path)

    def _step(self, tid, step, action_type, tool_name=None):
        return {"event": "react_step", "trace_id": tid,
                "payload": {"step": step, "action_type": action_type,
                            "tool_name": tool_name}}

    def test_reconstructs_tool_sequence_and_clean_end(self, tmp_path):
        tid = "abc123"
        path = self._write_events(tmp_path, [
            self._step(tid, 1, "tool", "search_wikipedia"),
            self._step(tid, 2, "tool", "get_weather"),
            self._step(tid, 3, "final"),
            {"event": "react_end", "trace_id": tid,
             "payload": {"steps_taken": 3, "reply_length": 55}},
        ])
        t = reconstruct_trajectory(tid, log_path=path)
        assert t["found"] is True
        assert t["tools_called"] == ["search_wikipedia", "get_weather"]
        assert t["steps_taken"] == 3
        assert t["terminated_cleanly"] is True
        assert t["final_reply_length"] == 55

    def test_hit_ceiling_is_not_clean(self, tmp_path):
        tid = "spin1"
        path = self._write_events(tmp_path, [
            self._step(tid, 1, "tool", "get_weather"),
            {"event": "react_end", "trace_id": tid,
             "payload": {"steps_taken": 5, "reply_length": 0, "stopped": "max_steps"}},
        ])
        t = reconstruct_trajectory(tid, log_path=path)
        assert t["terminated_cleanly"] is False
        assert t["steps_taken"] == 5

    def test_ignores_other_traces(self, tmp_path):
        path = self._write_events(tmp_path, [
            self._step("mine", 1, "tool", "get_weather"),
            self._step("other", 1, "tool", "search_wikipedia"),
            {"event": "react_end", "trace_id": "mine",
             "payload": {"steps_taken": 1, "reply_length": 10}},
        ])
        t = reconstruct_trajectory("mine", log_path=path)
        assert t["tools_called"] == ["get_weather"]

    def test_missing_trace_returns_not_found(self, tmp_path):
        path = self._write_events(tmp_path, [
            self._step("someone", 1, "tool", "get_weather"),
        ])
        t = reconstruct_trajectory("nope", log_path=path)
        assert t["found"] is False
        assert t["tools_called"] == []

    def test_missing_file_returns_not_found(self, tmp_path):
        t = reconstruct_trajectory("x", log_path=str(tmp_path / "nofile.jsonl"))
        assert t["found"] is False


# ---------- load_trajectory_dataset (validation) ----------

class TestLoadTrajectoryDataset:
    def _write(self, tmp_path, data):
        path = tmp_path / "traj.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def _entry(self, id="traj_001"):
        return {"id": id, "task": "do a thing",
                "expected_tools": ["get_weather"], "expected_outcome": "a result"}

    def test_loads_valid(self, tmp_path):
        path = self._write(tmp_path, [self._entry()])
        result = load_trajectory_dataset(path)
        assert len(result) == 1

    def test_allows_empty_expected_tools(self, tmp_path):
        e = self._entry()
        e["expected_tools"] = []
        path = self._write(tmp_path, [e])
        assert load_trajectory_dataset(path)[0]["expected_tools"] == []

    def test_rejects_empty_array(self, tmp_path):
        path = self._write(tmp_path, [])
        with pytest.raises(ValueError, match="non-empty"):
            load_trajectory_dataset(path)

    def test_rejects_missing_field(self, tmp_path):
        e = self._entry()
        del e["expected_outcome"]
        path = self._write(tmp_path, [e])
        with pytest.raises(ValueError, match="missing fields"):
            load_trajectory_dataset(path)

    def test_rejects_duplicate_ids(self, tmp_path):
        path = self._write(tmp_path, [self._entry("dup"), self._entry("dup")])
        with pytest.raises(ValueError, match="Duplicate"):
            load_trajectory_dataset(path)

    def test_rejects_non_list_expected_tools(self, tmp_path):
        e = self._entry()
        e["expected_tools"] = "get_weather"
        path = self._write(tmp_path, [e])
        with pytest.raises(ValueError, match="expected_tools"):
            load_trajectory_dataset(path)

    def test_loads_the_real_dataset(self):
        # The shipped dataset must always be valid.
        result = load_trajectory_dataset()
        assert len(result) >= 1
        assert all("task" in e for e in result)
