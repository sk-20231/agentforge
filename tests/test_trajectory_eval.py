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
    trace_to_eval_case,
    append_eval_case,
    review_draft_interactive,
    _next_trajectory_id,
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

    def test_recovers_task_from_react_start(self, tmp_path):
        tid = "t1"
        path = self._write_events(tmp_path, [
            {"event": "react_start", "trace_id": tid,
             "payload": {"user_input": "weather in Paris?", "max_steps": 5}},
            self._step(tid, 1, "tool", "get_weather"),
            {"event": "react_end", "trace_id": tid,
             "payload": {"steps_taken": 2, "reply_length": 30}},
        ])
        t = reconstruct_trajectory(tid, log_path=path)
        assert t["found"] is True
        assert t["task"] == "weather in Paris?"

    def test_task_defaults_empty_when_no_start(self, tmp_path):
        # A trace with only steps (no react_start) still reconstructs; task is "".
        path = self._write_events(tmp_path, [self._step("t2", 1, "tool", "get_weather")])
        t = reconstruct_trajectory("t2", log_path=path)
        assert t["task"] == ""


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


# ---------- Step 19 fast-follow: trace -> eval case ----------

class TestNextTrajectoryId:
    def test_empty_dataset_starts_at_001(self):
        assert _next_trajectory_id([]) == "traj_001"

    def test_increments_past_highest(self):
        ds = [{"id": "traj_001"}, {"id": "traj_003"}, {"id": "traj_002"}]
        assert _next_trajectory_id(ds) == "traj_004"

    def test_ignores_non_matching_ids(self):
        ds = [{"id": "custom_case"}, {"id": "traj_007"}]
        assert _next_trajectory_id(ds) == "traj_008"


class TestTraceToEvalCase:
    def _write_log(self, tmp_path, records):
        path = tmp_path / "logs.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return str(path)

    def _trace(self, tid, task, tools):
        records = [{"event": "react_start", "trace_id": tid,
                    "payload": {"user_input": task, "max_steps": 5}}]
        for i, name in enumerate(tools, start=1):
            records.append({"event": "react_step", "trace_id": tid,
                            "payload": {"step": i, "action_type": "tool",
                                        "tool_name": name}})
        records.append({"event": "react_end", "trace_id": tid,
                        "payload": {"steps_taken": len(tools) + 1, "reply_length": 40}})
        return records

    def test_draft_recovers_task_and_observed_tools(self, tmp_path):
        log = self._write_log(tmp_path, self._trace(
            "tid", "weather in Paris and Paris news?", ["get_weather", "get_top_news"]))
        draft = trace_to_eval_case(
            "tid", log_path=log, dataset_path=str(tmp_path / "none.json"))
        assert draft["task"] == "weather in Paris and Paris news?"
        assert draft["expected_tools"] == ["get_weather", "get_top_news"]
        assert draft["expected_outcome"] == ""          # human must fill
        assert draft["id"] == "traj_001"                # empty dataset
        assert draft["_source_trace_id"] == "tid"

    def test_draft_dedupes_repeated_tools_in_order(self, tmp_path):
        log = self._write_log(tmp_path, self._trace(
            "tid", "t", ["get_weather", "get_weather", "get_top_news"]))
        draft = trace_to_eval_case(
            "tid", log_path=log, dataset_path=str(tmp_path / "none.json"))
        assert draft["expected_tools"] == ["get_weather", "get_top_news"]

    def test_missing_trace_returns_none(self, tmp_path):
        log = self._write_log(tmp_path, self._trace("other", "t", ["get_weather"]))
        assert trace_to_eval_case(
            "absent", log_path=log, dataset_path=str(tmp_path / "none.json")) is None

    def test_id_avoids_clash_with_existing_dataset(self, tmp_path):
        ds = tmp_path / "traj.json"
        ds.write_text(json.dumps([
            {"id": "traj_009", "task": "x", "expected_tools": [],
             "expected_outcome": "y"}]), encoding="utf-8")
        log = self._write_log(tmp_path, self._trace("tid", "t", ["get_weather"]))
        draft = trace_to_eval_case("tid", log_path=log, dataset_path=str(ds))
        assert draft["id"] == "traj_010"


class TestAppendEvalCase:
    def _draft(self, **over):
        d = {"id": "traj_050", "task": "do a thing",
             "expected_tools": ["get_weather"], "expected_outcome": "a weather report",
             "expected_substrings": [], "difficulty": "easy"}
        d.update(over)
        return d

    def test_rejects_empty_outcome(self, tmp_path):
        path = str(tmp_path / "traj.json")
        with pytest.raises(ValueError, match="expected_outcome"):
            append_eval_case(self._draft(expected_outcome="  "), path=path)

    def test_rejects_missing_required_field(self, tmp_path):
        d = self._draft()
        del d["expected_tools"]
        with pytest.raises(ValueError, match="required fields"):
            append_eval_case(d, path=str(tmp_path / "traj.json"))

    def test_rejects_duplicate_id(self, tmp_path):
        path = tmp_path / "traj.json"
        path.write_text(json.dumps([self._draft(id="traj_050")]), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate id"):
            append_eval_case(self._draft(id="traj_050"), path=str(path))

    def test_appended_entry_is_loadable(self, tmp_path):
        # Round-trip: append into a fresh file, then the shipped validator accepts it.
        path = str(tmp_path / "traj.json")
        append_eval_case(self._draft(id="traj_100"), path=path)
        loaded = load_trajectory_dataset(path)
        assert [e["id"] for e in loaded] == ["traj_100"]

    def test_appends_to_existing_dataset(self, tmp_path):
        path = tmp_path / "traj.json"
        path.write_text(json.dumps([self._draft(id="traj_001")]), encoding="utf-8")
        append_eval_case(self._draft(id="traj_002"), path=str(path))
        assert [e["id"] for e in load_trajectory_dataset(str(path))] == \
            ["traj_001", "traj_002"]


class TestReviewDraftInteractive:
    def _draft(self):
        return {"id": "traj_001", "task": "weather in Paris?",
                "expected_tools": ["get_weather"], "expected_outcome": "",
                "expected_substrings": [], "difficulty": "unknown",
                "_source_trace_id": "tid"}

    def _feed(self, answers):
        it = iter(answers)
        return lambda _prompt: next(it)

    def test_happy_path_fills_outcome_and_confirms(self):
        # id, difficulty, tools, substrings, outcome, confirm
        fn = self._feed(["", "medium", "", "", "a Paris weather report", "y"])
        entry = review_draft_interactive(self._draft(), input_fn=fn)
        assert entry is not None
        assert entry["difficulty"] == "medium"
        assert entry["expected_outcome"] == "a Paris weather report"
        assert entry["expected_tools"] == ["get_weather"]   # kept the default

    def test_reasks_until_outcome_non_empty(self):
        # first outcome blank -> re-asked -> second non-empty
        fn = self._feed(["", "", "", "", "", "finally an outcome", "y"])
        entry = review_draft_interactive(self._draft(), input_fn=fn)
        assert entry["expected_outcome"] == "finally an outcome"

    def test_abort_when_not_confirmed(self):
        fn = self._feed(["", "", "", "", "an outcome", "n"])
        assert review_draft_interactive(self._draft(), input_fn=fn) is None

    def test_edit_tools_are_parsed(self):
        fn = self._feed(["", "hard", "get_weather, get_top_news", "Paris",
                         "both weather and news", "y"])
        entry = review_draft_interactive(self._draft(), input_fn=fn)
        assert entry["expected_tools"] == ["get_weather", "get_top_news"]
        assert entry["expected_substrings"] == ["Paris"]


# ------------- reconstruct_trajectory: tool ARGS (Step 21b.1) -------------

class TestReconstructTrajectoryToolCalls:
    """``tool_calls`` carries name AND arguments; ``tools_called`` stays as the
    names-only view, DERIVED from it so the two can never disagree."""

    def _write_events(self, tmp_path, records):
        path = tmp_path / "logs.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return str(path)

    def _tool_step(self, tid, step, tool_name, tool_input=None, **extra):
        payload = {"step": step, "action_type": "tool", "tool_name": tool_name}
        if tool_input is not None:
            payload["tool_input"] = tool_input
        payload.update(extra)
        return {"event": "react_step", "trace_id": tid, "payload": payload}

    def test_tool_calls_carry_name_and_args_in_order(self, tmp_path):
        tid = "args1"
        path = self._write_events(tmp_path, [
            self._tool_step(tid, 1, "get_weather", {"city": "Paris"}),
            self._tool_step(tid, 2, "get_top_news", {"topic": "Paris"}),
            {"event": "react_end", "trace_id": tid,
             "payload": {"steps_taken": 3, "reply_length": 20}},
        ])
        t = reconstruct_trajectory(tid, log_path=path)
        assert t["tool_calls"] == [
            {"name": "get_weather", "args": {"city": "Paris"}},
            {"name": "get_top_news", "args": {"topic": "Paris"}},
        ]

    def test_tools_called_is_derived_from_tool_calls(self, tmp_path):
        tid = "args2"
        path = self._write_events(tmp_path, [
            self._tool_step(tid, 1, "get_weather", {"city": "Paris"}),
            self._tool_step(tid, 2, "get_top_news", {"topic": "Paris"}),
        ])
        t = reconstruct_trajectory(tid, log_path=path)
        assert t["tools_called"] == [c["name"] for c in t["tool_calls"]]
        assert t["tools_called"] == ["get_weather", "get_top_news"]

    def test_old_trace_without_tool_input_still_reconstructs(self, tmp_path):
        """Backward compat: traces written before 21b.1 have no `tool_input`.
        They must still read back, just without argument detail."""
        tid = "legacy"
        path = self._write_events(tmp_path, [
            self._tool_step(tid, 1, "get_weather"),   # no tool_input key
            {"event": "react_end", "trace_id": tid,
             "payload": {"steps_taken": 2, "reply_length": 9}},
        ])
        t = reconstruct_trajectory(tid, log_path=path)
        assert t["tools_called"] == ["get_weather"]
        assert t["tool_calls"] == [{"name": "get_weather", "args": {}}]

    def test_no_tool_steps_gives_empty_tool_calls(self, tmp_path):
        tid = "notools"
        path = self._write_events(tmp_path, [
            {"event": "react_step", "trace_id": tid,
             "payload": {"step": 1, "action_type": "final", "tool_name": None}},
            {"event": "react_end", "trace_id": tid,
             "payload": {"steps_taken": 1, "reply_length": 5}},
        ])
        t = reconstruct_trajectory(tid, log_path=path)
        assert t["tool_calls"] == []
        assert t["tools_called"] == []

    def test_repeated_tool_keeps_both_calls_with_their_own_args(self, tmp_path):
        """The distinguishing case for arg logging: same tool, different args.
        Names alone make these two steps look identical."""
        tid = "repeat"
        path = self._write_events(tmp_path, [
            self._tool_step(tid, 1, "get_weather", {"city": "Paris"}),
            self._tool_step(tid, 2, "get_weather", {"city": "Berlin"}),
        ])
        t = reconstruct_trajectory(tid, log_path=path)
        assert t["tools_called"] == ["get_weather", "get_weather"]
        assert [c["args"]["city"] for c in t["tool_calls"]] == ["Paris", "Berlin"]

    def test_truncation_flag_does_not_leak_into_tool_calls(self, tmp_path):
        """The flags are step-level metadata, not part of the call record."""
        tid = "trunc"
        path = self._write_events(tmp_path, [
            self._tool_step(tid, 1, "get_weather", {"city": "Par...[truncated]"},
                            tool_input_truncated=True),
        ])
        t = reconstruct_trajectory(tid, log_path=path)
        assert set(t["tool_calls"][0].keys()) == {"name", "args"}
