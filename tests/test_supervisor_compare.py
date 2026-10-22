"""
Unit tests for Step 21.2 part 2b — the supervisor ON vs OFF comparison.

Hermetic: a fake agent stands in for ``main.run_agent`` (it reads the same
module flag the real one reads, and logs token usage the way real LLM calls
do), and a fake judge replaces the LLM judge. No test calls the API.
"""
import json
import pytest

from agentforge import logger as logger_mod
from agentforge import main
from agentforge import supervisor_compare as sc
from agentforge.evaluation import load_planner_dataset
from agentforge.logger import log_event
from agentforge.memory import semantic
from agentforge.rag import document_store


def _case(cid, message="q", outcome="the answer", group="g", history=None):
    return {"id": cid, "message": message, "accept": [["answer"]], "group": group,
            "history": history or [], "expected_outcome": outcome}


class FakeAgent:
    """Records each call with the arm and memory folder it ran under."""

    def __init__(self, replies=None, tokens=None, raise_on=None):
        self.calls = []
        self.replies = replies or {}          # (message, arm) -> reply
        self.tokens = tokens or {"off": 100, "on": 150}
        self.raise_on = raise_on

    def __call__(self, user_id, session_id, message, history=None, stream=False,
                 trace_id=None, approval_handler=None):
        arm = "on" if main.AGENT_SUPERVISOR_ENABLED else "off"
        self.calls.append({"message": message, "arm": arm, "memory": semantic.MEMORY_DIR,
                           "history": history, "stream": stream, "session": session_id,
                           "approves": approval_handler(None)})
        if message == self.raise_on:
            raise RuntimeError("boom")
        log_event("token_usage", {"prompt_tokens": self.tokens[arm], "completion_tokens": 10,
                                  "cost_usd": 0.001, "operation": "fake"}, trace_id=trace_id)
        return self.replies.get((message, arm), "ok")

    @property
    def measured(self):
        """The calls without the unrecorded warm-up turn."""
        return [c for c in self.calls if c["session"] != "compare-warm-up"]


def judge_ok(task, reply, expected):
    return {"completed": "ok" in reply, "reason": f"judged {reply!r}", "method": "fake"}


class TestCompare:
    def test_summary_per_arm(self, tmp_path):
        agent = FakeAgent(replies={("b", "off"): "wrong"})
        r = sc.compare([_case("p1", "a"), _case("p2", "b")], agent=agent, judge=judge_ok,
                       tmpdir=str(tmp_path), verbose=False)
        assert r["arms"]["off"]["completed"] == 1
        assert r["arms"]["on"]["completed"] == 2
        assert r["arms"]["on"]["completion_rate"] == 1.0
        # Tokens come from the trace's token_usage events in the isolated log.
        assert r["arms"]["off"]["tokens_mean"] == 110
        assert r["arms"]["on"]["tokens_mean"] == 160
        assert r["arms"]["on"]["llm_calls_mean"] == 1.0
        assert r["arms"]["off"]["cost_usd"] == pytest.approx(0.002)

    def test_result_records_the_settings(self, tmp_path):
        r = sc.compare([_case("p1")], agent=FakeAgent(), judge=judge_ok, tmpdir=str(tmp_path),
                       verbose=False)
        assert "OPENAI_MODEL" in r["config"]
        assert "AGENT_OUTPUT_GUARDRAIL_ENGINE" in r["config"]
        assert r["config"]["JUDGE_TEMPERATURE"] == 0
        json.dumps(r)                                   # --out must be able to save it

    def test_default_judge_runs_at_temperature_zero(self, tmp_path, monkeypatch):
        seen = {}

        def fake_judge(task, reply, expected, temperature=None):
            seen["temperature"] = temperature
            return {"completed": True, "reason": "", "method": "fake"}

        monkeypatch.setattr(sc, "score_task_completion", fake_judge)
        sc.compare([_case("p1")], agent=FakeAgent(), tmpdir=str(tmp_path), verbose=False)
        assert seen["temperature"] == 0

    def test_disagreements_name_the_winner(self, tmp_path):
        agent = FakeAgent(replies={("b", "off"): "wrong"})
        r = sc.compare([_case("p1", "a"), _case("p2", "b")], agent=agent, judge=judge_ok,
                       tmpdir=str(tmp_path), verbose=False)
        assert [(d["id"], d["winner"]) for d in r["disagreements"]] == [("p2", "on")]
        assert r["disagreements"][0]["off"]["reply"] == "wrong"

    def test_arms_alternate_which_runs_first(self, tmp_path):
        agent = FakeAgent()
        sc.compare([_case("p1", "a"), _case("p2", "b"), _case("p3", "c")], agent=agent,
                   judge=judge_ok, tmpdir=str(tmp_path), verbose=False)
        assert [(c["message"], c["arm"]) for c in agent.measured] == [
            ("a", "off"), ("a", "on"), ("b", "on"), ("b", "off"), ("c", "off"), ("c", "on")]

    def test_each_arm_has_its_own_memory_per_run(self, tmp_path):
        agent = FakeAgent()
        sc.compare([_case("p1", "a")], runs=2, agent=agent, judge=judge_ok,
                   tmpdir=str(tmp_path), verbose=False)
        folders = {(c["arm"], c["memory"]) for c in agent.measured}
        assert len(folders) == 4                       # 2 arms x 2 runs
        assert all(str(tmp_path) in m for _, m in folders)

    def test_agent_gets_history_no_stream_and_an_approver(self, tmp_path):
        history = [{"role": "user", "content": "weather in Berlin?"}]
        agent = FakeAgent()
        sc.compare([_case("p1", "a", history=history)], agent=agent, judge=judge_ok,
                   tmpdir=str(tmp_path), verbose=False)
        assert agent.measured[0]["history"] == history
        assert agent.measured[0]["stream"] is False
        assert agent.measured[0]["approves"] is True

    def test_warm_up_runs_first_and_is_not_recorded(self, tmp_path):
        agent = FakeAgent()
        r = sc.compare([_case("p1", "a")], agent=agent, judge=judge_ok,
                       tmpdir=str(tmp_path), verbose=False)
        first = agent.calls[0]
        assert first["session"] == "compare-warm-up" and first["message"] == sc.WARM_UP_MESSAGE
        assert first["arm"] == "off"
        # Its own memory folder: the warm-up must not store anything an arm reads.
        assert first["memory"] not in {c["memory"] for c in agent.measured}
        assert [t["id"] for t in r["per_turn"]] == ["p1", "p1"]
        assert r["arms"]["off"]["tokens"] == 110       # the warm-up's tokens are not counted

    def test_failed_warm_up_does_not_stop_the_run(self, tmp_path):
        r = sc.compare([_case("p1", "a")], agent=FakeAgent(raise_on=sc.WARM_UP_MESSAGE),
                       judge=judge_ok, tmpdir=str(tmp_path), verbose=False)
        assert r["arms"]["on"]["completed"] == 1

    def test_warm_up_can_be_turned_off(self, tmp_path):
        agent = FakeAgent()
        sc.compare([_case("p1", "a")], agent=agent, judge=judge_ok, tmpdir=str(tmp_path),
                   verbose=False, warm_up=False)
        assert agent.calls == agent.measured

    def test_judge_sees_the_history(self, tmp_path):
        seen = []

        def judge(task, reply, expected):
            seen.append((task, expected))
            return {"completed": True, "reason": "", "method": "fake"}

        history = [{"role": "user", "content": "weather in Berlin?"},
                   {"role": "assistant", "content": "12 C in Berlin."}]
        sc.compare([_case("p1", "And news there?", outcome="Berlin news", history=history)],
                   agent=FakeAgent(), judge=judge, tmpdir=str(tmp_path), verbose=False)
        task, expected = seen[0]
        assert "weather in Berlin?" in task and "And news there?" in task
        assert expected == "Berlin news"

    def test_crashing_agent_is_recorded_not_raised(self, tmp_path):
        judged = []

        def judge(task, reply, expected):
            judged.append(reply)
            return judge_ok(task, reply, expected)

        r = sc.compare([_case("p1", "bad"), _case("p2", "a")], agent=FakeAgent(raise_on="bad"),
                       judge=judge, tmpdir=str(tmp_path), verbose=False)
        assert r["arms"]["off"]["errors"] == 1 and r["arms"]["on"]["errors"] == 1
        crashed = [t for t in r["per_turn"] if t["id"] == "p1"]
        assert all(not t["completed"] and "boom" in t["error"] for t in crashed)
        assert len(judged) == 2                        # the judge never sees a crashed turn
        assert r["arms"]["on"]["completed"] == 1       # the run went on

    def test_by_group(self, tmp_path):
        r = sc.compare([_case("p1", "a", group="single"), _case("p2", "b", group="multi")],
                       agent=FakeAgent(replies={("b", "on"): "no"}), judge=judge_ok,
                       tmpdir=str(tmp_path), verbose=False)
        assert r["by_group"]["multi"]["off"]["completed"] == 1
        assert r["by_group"]["multi"]["on"]["completed"] == 0
        assert r["by_group"]["single"]["on"]["total"] == 1

    def test_missing_expected_outcome_is_rejected(self, tmp_path):
        case = _case("p1")
        del case["expected_outcome"]
        with pytest.raises(ValueError, match="p1"):
            sc.compare([case], agent=FakeAgent(), judge=judge_ok, tmpdir=str(tmp_path),
                       verbose=False)

    def test_state_is_restored_and_fixture_untouched(self, tmp_path):
        before = (logger_mod.AGENT_LOG_FILE, document_store.AGENT_CORPUS_FILE,
                  semantic.MEMORY_DIR, main.AGENT_SUPERVISOR_ENABLED)
        fixture = sc._TEST_CORPUS.read_bytes()
        seen = {}

        def agent(*args, **kwargs):
            seen["corpus"] = document_store.AGENT_CORPUS_FILE
            seen["log"] = logger_mod.AGENT_LOG_FILE
            return "ok"

        sc.compare([_case("p1")], agent=agent, judge=judge_ok, tmpdir=str(tmp_path),
                   verbose=False)
        assert str(tmp_path) in seen["corpus"] and str(tmp_path) in seen["log"]
        assert json.loads(open(seen["corpus"], encoding="utf-8").read())["chunks"]
        after = (logger_mod.AGENT_LOG_FILE, document_store.AGENT_CORPUS_FILE,
                 semantic.MEMORY_DIR, main.AGENT_SUPERVISOR_ENABLED)
        assert after == before
        assert sc._TEST_CORPUS.read_bytes() == fixture

    def test_state_is_restored_after_an_exception(self, tmp_path):
        before = (logger_mod.AGENT_LOG_FILE, main.AGENT_SUPERVISOR_ENABLED)

        def judge(*args):
            raise KeyboardInterrupt  # not caught by _run_turn: must still restore

        with pytest.raises(KeyboardInterrupt):
            sc.compare([_case("p1")], agent=FakeAgent(), judge=judge, tmpdir=str(tmp_path),
                       verbose=False)
        assert (logger_mod.AGENT_LOG_FILE, main.AGENT_SUPERVISOR_ENABLED) == before


class TestCli:
    def test_unknown_case_id(self, capsys):
        assert sc.main(["--cases", "plan_999"]) == 2
        assert "plan_999" in capsys.readouterr().out


class TestShippedDataset:
    def test_every_case_has_an_expected_outcome(self):
        for entry in load_planner_dataset():
            assert entry.get("expected_outcome", "").strip(), entry["id"]

    def test_test_corpus_exists(self):
        assert sc._TEST_CORPUS.is_file()
