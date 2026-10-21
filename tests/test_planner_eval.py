"""
Unit tests for Step 21.2 part 2 — the planner eval.

Hermetic: ``score_plan`` is pure, and ``run_planner_eval`` is driven by a fake
planner, so no test calls the API. The shipped dataset is checked for shape
and for worker names the supervisor actually has.
"""
import json
import pytest

from agentforge.evaluation import (
    PLANNER_OUTCOMES,
    load_planner_dataset,
    run_planner_eval,
    score_plan,
)
from agentforge.prompts import SUPERVISOR_WORKER_NAMES
from agentforge.supervisor import Plan


def _tasks(*pairs):
    return [{"worker": w, "brief": b} for w, b in pairs]


def _entry(accept, keywords=(), dependent=False):
    return {"id": "plan_x", "message": "m", "accept": accept,
            "keywords": list(keywords), "dependent": dependent}


# ---------- score_plan (pure) ----------

class TestScorePlan:
    def test_accepted_plan_passes(self):
        entry = _entry([["read_user_documents", "research"]], keywords=["parental"])
        tasks = _tasks(("research", "news on parental leave laws"),
                       ("read_user_documents", "handbook parental leave policy"))
        assert score_plan(entry, tasks)["outcome"] == "pass"

    def test_direct_reply_passes_when_no_task_expected(self):
        assert score_plan(_entry([[]]), [])["outcome"] == "pass"

    def test_none_is_invalid(self):
        assert score_plan(_entry([["research"]]), None)["outcome"] == "invalid"

    def test_lost_keyword_is_harmful_even_with_right_workers(self):
        entry = _entry([["research"]], keywords=["berlin"])
        r = score_plan(entry, _tasks(("research", "latest news there")))
        assert r["outcome"] == "harmful"
        assert "berlin" in r["reason"]

    def test_keywords_are_case_insensitive(self):
        entry = _entry([["research"]], keywords=["Berlin"])
        assert score_plan(entry, _tasks(("research", "news in BERLIN")))["outcome"] == "pass"

    def test_dropped_part_is_harmful(self):
        # "I love hiking. What does the handbook say about volunteer days?"
        entry = _entry([["read_user_documents", "remember"]], keywords=["volunteer"])
        r = score_plan(entry, _tasks(("read_user_documents", "volunteer days")))
        assert r["outcome"] == "harmful"

    def test_swap_in_multi_task_plan_is_harmless(self):
        # Accepted: docs + answer. Planned: docs + research. In a 2+-task plan
        # answer runs on REACT, so both plans run the same workers.
        entry = _entry([["answer", "read_user_documents"]])
        tasks = _tasks(("read_user_documents", "remote work"), ("research", "what a VPN is"))
        r = score_plan(entry, tasks)
        assert r["outcome"] == "harmless"
        assert "swap" in r["reason"]

    def test_answer_for_a_lookup_in_a_one_task_plan_is_harmful(self):
        # One task: answer keeps its own pipeline, which has no tools.
        entry = _entry([["research"]], keywords=["tokyo"])
        assert score_plan(entry, _tasks(("answer", "weather in Tokyo")))["outcome"] == "harmful"

    def test_dependent_split_is_its_own_outcome(self):
        entry = _entry([["research"]], dependent=True)
        tasks = _tasks(("research", "who is the CEO of OpenAI and where were they born"),
                       ("research", "weather in that city"))
        assert score_plan(entry, tasks)["outcome"] == "dependent_split"

    def test_dependent_split_with_answer_counts_as_split(self):
        # "Plan my weekend in Berlin around the weather" -> weather + plan.
        entry = _entry([["research"]], keywords=["berlin"], dependent=True)
        tasks = _tasks(("research", "weather in Berlin"), ("answer", "plan a weekend in Berlin"))
        assert score_plan(entry, tasks)["outcome"] == "dependent_split"

    def test_extra_research_task_is_harmful_not_a_split(self):
        # Live run 1: "And what's the latest news there?" after a Berlin weather
        # turn -> the planner added the weather lookup again, unasked.
        entry = _entry([["research"]], keywords=["berlin"], dependent=False)
        tasks = _tasks(("research", "current weather in Berlin"),
                       ("research", "latest news about Berlin"))
        r = score_plan(entry, tasks)
        assert r["outcome"] == "harmful"
        assert "extra research" in r["reason"]

    def test_merged_research_tasks_are_harmless(self):
        entry = _entry([["research", "research"]], keywords=["london"])
        tasks = _tasks(("research", "weather in London and UK tech news"))
        r = score_plan(entry, tasks)
        assert r["outcome"] == "harmless"
        assert "merged" in r["reason"]

    def test_merge_counts_against_the_largest_accepted_plan(self):
        # Accepted: 4 or 2 research tasks. 3 is fewer than 4 -> merged.
        entry = _entry([["research"] * 4, ["research"] * 2])
        tasks = _tasks(("research", "a"), ("research", "b"), ("research", "c"))
        assert score_plan(entry, tasks)["outcome"] == "harmless"

    def test_merge_keeps_other_workers(self):
        # Merging research must not hide a dropped remember part.
        entry = _entry([["remember", "research", "research"]])
        assert score_plan(entry, _tasks(("research", "x")))["outcome"] == "harmful"

    def test_merging_a_knowledge_part_into_research_is_harmful(self):
        # The merge rule only uses accepted plans with no answer part: a
        # knowledge part folded into a lookup brief could be silently dropped.
        entry = _entry([["answer", "research"]], keywords=["rome"])
        assert score_plan(entry, _tasks(("research", "weather in Rome")))["outcome"] == "harmful"

    def test_docs_instead_of_research_is_harmful(self):
        entry = _entry([["research"]], keywords=["paris"])
        r = score_plan(entry, _tasks(("read_user_documents", "weather in Paris")))
        assert r["outcome"] == "harmful"

    def test_every_outcome_is_a_known_outcome(self):
        entry = _entry([["research"]], dependent=True)
        for tasks in (None, [], _tasks(("research", "x")),
                      _tasks(("research", "a"), ("research", "b"))):
            assert score_plan(entry, tasks)["outcome"] in PLANNER_OUTCOMES


# ---------- run_planner_eval (fake planner, no API) ----------

class TestRunPlannerEval:
    DATASET = [
        {"id": "p1", "message": "weather in Tokyo", "accept": [["research"]],
         "keywords": ["tokyo"], "group": "single"},
        {"id": "p2", "message": "hi", "accept": [[]], "group": "single"},
        {"id": "p3", "message": "And there?", "accept": [["research"]],
         "keywords": ["berlin"], "group": "history",
         "history": [{"role": "user", "content": "weather in Berlin?"}]},
    ]

    def _planner(self, calls):
        def planner(message, history):
            calls.append((message, history))
            if message == "hi":
                return Plan(tasks=[], direct_reply="Hello!")
            if message == "And there?":
                return None                       # invalid plan
            return Plan(tasks=_tasks(("research", "weather in Tokyo")))
        return planner

    def test_counts_rates_and_groups(self):
        calls = []
        r = run_planner_eval(self.DATASET, planner=self._planner(calls), verbose=False)
        assert r["total"] == 3
        assert r["counts"] == {"pass": 2, "harmless": 0, "dependent_split": 0,
                               "invalid": 1, "harmful": 0}
        assert r["rates"]["pass"] == pytest.approx(2 / 3)
        assert r["by_group"]["single"]["pass"] == 2
        assert r["by_group"]["history"]["invalid"] == 1
        assert r["by_group"]["history"]["total"] == 1

    def test_history_is_passed_to_the_planner(self):
        calls = []
        run_planner_eval(self.DATASET, planner=self._planner(calls), verbose=False)
        assert calls[0] == ("weather in Tokyo", [])
        assert calls[2][1] == [{"role": "user", "content": "weather in Berlin?"}]

    def test_runs_repeat_every_case(self):
        calls = []
        r = run_planner_eval(self.DATASET, runs=2, planner=self._planner(calls), verbose=False)
        assert r["total"] == 6 and len(calls) == 6
        assert [c["run"] for c in r["per_case"]] == [0, 0, 0, 1, 1, 1]

    def test_crashing_planner_is_recorded_not_raised(self):
        def planner(message, history):
            raise RuntimeError("boom")
        r = run_planner_eval(self.DATASET[:1], planner=planner, verbose=False)
        assert r["counts"]["invalid"] == 1
        assert r["per_case"][0]["error"] == "boom"

    def test_reports_cost_fields(self):
        r = run_planner_eval(self.DATASET, planner=self._planner([]), verbose=False)
        assert r["tokens"] == {"total": 0, "mean": 0.0}   # fake planner logs no usage
        assert r["latency_ms"]["max"] >= r["latency_ms"]["p50"] >= 0
        assert all("latency_ms" in c and "tokens" in c for c in r["per_case"])


# ---------- load_planner_dataset ----------

class TestLoadPlannerDataset:
    def _write(self, tmp_path, data):
        p = tmp_path / "planner.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    def test_rejects_empty(self, tmp_path):
        with pytest.raises(ValueError):
            load_planner_dataset(self._write(tmp_path, []))

    def test_rejects_missing_field(self, tmp_path):
        with pytest.raises(ValueError, match="missing"):
            load_planner_dataset(self._write(tmp_path, [{"id": "a", "message": "m"}]))

    def test_rejects_duplicate_id(self, tmp_path):
        e = {"id": "a", "message": "m", "accept": [[]]}
        with pytest.raises(ValueError, match="Duplicate"):
            load_planner_dataset(self._write(tmp_path, [e, dict(e)]))

    def test_rejects_bad_accept(self, tmp_path):
        with pytest.raises(ValueError, match="accept"):
            load_planner_dataset(self._write(
                tmp_path, [{"id": "a", "message": "m", "accept": ["research"]}]))


class TestShippedDataset:
    def test_loads(self):
        assert len(load_planner_dataset()) >= 40

    def test_only_real_worker_names(self):
        for entry in load_planner_dataset():
            for combo in entry["accept"]:
                assert set(combo) <= set(SUPERVISOR_WORKER_NAMES), entry["id"]

    def test_dependent_cases_accept_one_research_task(self):
        # The dependent_split outcome only makes sense if the right plan is
        # ONE research task (REACT chains the steps itself).
        dependent = [e for e in load_planner_dataset() if e.get("dependent")]
        assert dependent
        for entry in dependent:
            assert entry["accept"] == [["research"]], entry["id"]
