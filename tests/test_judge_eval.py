"""Hermetic tests for the judge-agreement eval (issue #95).

The eval itself is live (real judge model). These check the harness: the
metric maths against hand-calculated values, that a judge failure stays in the
denominator, the gold-set validation, the labelling session, the run loop with
a fake judge, the exit codes, and the shipped gold set's shape.
"""
import json

import pytest

from agentforge import judge_eval
from agentforge.judge_eval import (
    FAITHFUL as F,
    JUDGE_ERROR as E,
    UNFAITHFUL as U,
    agreement_report,
    cohens_kappa,
    confusion_matrix,
    kappa_interval,
)


def _case(cid="g1", label=None, split="dev", source="authored", category="added_fact",
          by="human"):
    case = {
        "id": cid, "question": "What is the budget?",
        "chunks": [{"id": "c0", "text": "The budget is $100."}],
        "answer": "The budget is $100.", "split": split, "source": source,
        "category": category,
    }
    if label is not None:
        case["reference_label"] = label
        case["labelled_by"] = by
    return case


def _write(tmp_path, cases):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return str(path)


def _pairs(ff, fu, uf, uu):
    """Label lists from the four cell counts (human first, judge second)."""
    human = [F] * (ff + fu) + [U] * (uf + uu)
    judge = [F] * ff + [U] * fu + [F] * uf + [U] * uu
    return human, judge


# ------------------------------- kappa -------------------------------

class TestCohensKappa:
    def test_textbook_value(self):
        # 20 + 15 of 50 agree -> observed 0.70. Chance: 0.5*0.6 + 0.5*0.4 = 0.50.
        # kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40.
        human, judge = _pairs(ff=20, fu=5, uf=10, uu=15)
        assert cohens_kappa(human, judge) == pytest.approx(0.40)

    def test_perfect_agreement_is_one(self):
        assert cohens_kappa([F, U, F, U], [F, U, F, U]) == pytest.approx(1.0)

    def test_total_disagreement_is_negative(self):
        assert cohens_kappa([F, U, F, U], [U, F, U, F]) == pytest.approx(-1.0)

    def test_high_raw_agreement_can_be_zero_kappa(self):
        # Both sides say "faithful" 90% of the time, independently: they agree
        # on 82% of cases, and that is exactly the chance level.
        human, judge = _pairs(ff=81, fu=9, uf=9, uu=1)
        assert agreement_report(human, judge)["agreement"] == pytest.approx(0.82)
        assert cohens_kappa(human, judge) == pytest.approx(0.0)

    def test_one_label_on_both_sides_is_undefined_not_one(self):
        assert cohens_kappa([F, F, F], [F, F, F]) is None

    def test_no_cases_is_undefined(self):
        assert cohens_kappa([], []) is None

    def test_judge_failure_stays_in_the_denominator(self):
        # Without the failed case the judge is perfect (kappa 1.0). With it:
        # observed 3/4; chance 0.5*0.25 + 0.5*0.5 = 0.375; kappa = 0.6.
        assert cohens_kappa([F, F, U, U], [F, E, U, U]) == pytest.approx(0.6)
        assert cohens_kappa([F, U, U], [F, U, U]) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa([F], [F, U])


class TestKappaInterval:
    def test_contains_kappa_and_is_repeatable(self):
        human, judge = _pairs(ff=20, fu=5, uf=10, uu=15)
        low, high = kappa_interval(human, judge)
        assert low < 0.40 < high
        assert kappa_interval(human, judge) == (low, high)

    def test_undefined_kappa_has_no_interval(self):
        assert kappa_interval([F, F], [F, F]) is None


# ------------------------- matrix and report -------------------------

class TestConfusionMatrix:
    def test_counts_every_cell(self):
        m = confusion_matrix([F, F, F, U, U, U], [F, U, E, F, U, U])
        assert m == {F: {F: 1, U: 1, E: 1}, U: {F: 1, U: 2, E: 0}}

    def test_reference_label_cannot_be_error(self):
        with pytest.raises(ValueError):
            confusion_matrix([E], [F])

    def test_unknown_judge_label_raises(self):
        with pytest.raises(ValueError):
            confusion_matrix([F], ["maybe"])


class TestAgreementReport:
    def test_both_error_rates(self):
        human, judge = _pairs(ff=20, fu=5, uf=10, uu=15)
        r = agreement_report(human, judge)
        assert r["n"] == 50
        assert r["missed_unfaithful_rate"] == pytest.approx(10 / 25)
        assert r["wrongly_flagged_rate"] == pytest.approx(5 / 25)
        assert r["judge_errors"] == 0
        assert r["kappa"] == pytest.approx(0.40)

    def test_a_failure_is_neither_a_miss_nor_a_flag(self):
        r = agreement_report([F, U], [E, E])
        assert r["judge_errors"] == 2
        assert r["missed_unfaithful_rate"] == 0.0
        assert r["wrongly_flagged_rate"] == 0.0
        assert r["agreement"] == 0.0

    def test_rate_is_none_when_the_class_is_empty(self):
        r = agreement_report([F, F], [F, U])
        assert r["missed_unfaithful_rate"] is None
        assert r["wrongly_flagged_rate"] == pytest.approx(0.5)


# ----------------------------- gold set -----------------------------

class TestLoadGoldDataset:
    def test_loads_labelled_and_unlabelled(self, tmp_path):
        path = _write(tmp_path, [_case("a", F), _case("b"), _case("c", None)])
        assert [c["id"] for c in judge_eval.load_gold_dataset(path)] == ["a", "b", "c"]

    @pytest.mark.parametrize("change, message", [
        ({"split": "train"}, "split"),
        ({"source": "web"}, "source"),
        ({"reference_label": "maybe"}, "reference_label"),
        ({"reference_label": "error"}, "reference_label"),
        ({"chunks": []}, "chunks"),
        ({"chunks": [{"id": "c0"}]}, "chunks"),
        ({"answer": "  "}, "answer"),
    ])
    def test_rejects_a_malformed_case(self, tmp_path, change, message):
        path = _write(tmp_path, [_case() | change])
        with pytest.raises(ValueError, match=message):
            judge_eval.load_gold_dataset(path)

    @pytest.mark.parametrize("by", [None, "", "model", "claude"])
    def test_rejects_a_label_without_a_known_labeller(self, tmp_path, by):
        case = _case("a", F)
        case["labelled_by"] = by
        with pytest.raises(ValueError, match="labelled_by"):
            judge_eval.load_gold_dataset(_write(tmp_path, [case]))

    def test_accepts_a_label_given_by_an_ai(self, tmp_path):
        path = _write(tmp_path, [_case("a", U, by="ai")])
        assert judge_eval.load_gold_dataset(path)[0]["labelled_by"] == "ai"

    def test_rejects_a_missing_field(self, tmp_path):
        case = _case()
        del case["category"]
        with pytest.raises(ValueError, match="category"):
            judge_eval.load_gold_dataset(_write(tmp_path, [case]))

    def test_rejects_duplicate_ids(self, tmp_path):
        with pytest.raises(ValueError, match="Duplicate"):
            judge_eval.load_gold_dataset(_write(tmp_path, [_case("a"), _case("a")]))

    def test_rejects_an_empty_set(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty"):
            judge_eval.load_gold_dataset(_write(tmp_path, []))


class TestShippedGoldSet:
    """The file in the repo must stay usable for the live eval."""

    def test_is_valid_and_both_halves_are_filled(self):
        cases = judge_eval.load_gold_dataset()
        by_split = {s: [c for c in cases if c["split"] == s] for s in judge_eval.SPLITS}
        assert all(len(v) >= 10 for v in by_split.values())
        for half in by_split.values():
            assert {c["source"] for c in half} == set(judge_eval.SOURCES)

    def test_ids_do_not_give_away_the_category(self):
        for case in judge_eval.load_gold_dataset():
            assert case["category"] not in case["id"]
            assert case["source"] not in case["id"]


# ----------------------------- labelling -----------------------------

class _Session:
    """Scripted keyboard input; records everything printed."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []
        self.shown = []

    def input(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0)

    def print(self, text=""):
        self.shown.append(str(text))


class TestLabelGoldInteractive:
    def _run(self, tmp_path, cases, answers, **kwargs):
        path = _write(tmp_path, cases)
        session = _Session(answers)
        out = judge_eval.label_gold_interactive(
            path, input_fn=session.input, print_fn=session.print, **kwargs)
        saved = {c["id"]: c for c in json.loads(open(path, encoding="utf-8").read())}
        return out, saved, session

    def test_labels_are_saved_with_an_optional_note(self, tmp_path):
        out, saved, _ = self._run(tmp_path, [_case("a")], ["u", "number changed"])
        assert out == {"labelled": 1, "skipped": 0, "remaining": 0}
        assert saved["a"]["reference_label"] == U
        assert saved["a"]["note"] == "number changed"
        assert saved["a"]["labelled_by"] == "human"

    def test_relabelling_an_ai_label_makes_it_a_human_label(self, tmp_path):
        _, saved, _ = self._run(tmp_path, [_case("a", F, by="ai")], ["u", ""], relabel=True)
        assert saved["a"]["reference_label"] == U
        assert saved["a"]["labelled_by"] == "human"

    def test_skip_leaves_the_case_without_a_label(self, tmp_path):
        out, saved, _ = self._run(tmp_path, [_case("a")], ["s"])
        assert out == {"labelled": 0, "skipped": 1, "remaining": 1}
        assert saved["a"].get("reference_label") is None

    def test_quit_keeps_what_was_already_saved(self, tmp_path):
        out, saved, _ = self._run(
            tmp_path, [_case("a"), _case("b"), _case("c")], ["f", "", "q"])
        assert out["labelled"] == 1 and out["remaining"] == 2
        assert sum(1 for c in saved.values() if c.get("reference_label")) == 1

    def test_an_unknown_key_asks_again(self, tmp_path):
        out, saved, session = self._run(tmp_path, [_case("a")], ["x", "", "f", ""])
        assert saved["a"]["reference_label"] == F
        assert len(session.prompts) == 4

    def test_labelled_cases_are_not_shown_again_unless_relabel(self, tmp_path):
        cases = [_case("a", F), _case("b")]
        out, saved, _ = self._run(tmp_path, cases, ["u", ""])
        assert out["labelled"] == 1 and saved["a"]["reference_label"] == F
        out, saved, _ = self._run(tmp_path, cases, ["u", "", "u", ""], relabel=True)
        assert out["labelled"] == 2 and saved["a"]["reference_label"] == U

    def test_nothing_to_label(self, tmp_path):
        out, _, session = self._run(tmp_path, [_case("a", F)], [])
        assert out == {"labelled": 0, "skipped": 0, "remaining": 0}
        assert session.prompts == []

    def test_source_and_category_are_never_shown(self, tmp_path):
        case = _case("a", source="authored", category="changed_number")
        _, _, session = self._run(tmp_path, [case], ["f", ""])
        screen = "\n".join(session.shown + session.prompts)
        assert "changed_number" not in screen and "authored" not in screen
        assert "What is the budget?" in screen and "[c0]" in screen

    def test_the_rules_are_shown_before_the_first_case(self, tmp_path):
        _, _, session = self._run(tmp_path, [_case("a")], ["s"])
        assert session.shown[0] == judge_eval.LABELLING_RULES


# ------------------------------ run loop ------------------------------

def _fake_judge(verdicts):
    """A judge that answers by case question; records how it was called."""
    calls = []

    def judge(question, answer, chunks, model=None, temperature=None):
        calls.append({"question": question, "model": model, "temperature": temperature})
        verdict = verdicts[question]
        if isinstance(verdict, list):          # a different verdict per run
            verdict = verdict.pop(0)
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    judge.calls = calls
    return judge


def _ok(faithful):
    return {"faithful": faithful, "reason": "because", "error": None}


class TestRunJudgeAgreement:
    def _cases(self):
        cases = [_case("a", F), _case("b", U), _case("c", F, source="pipeline",
                                                     category="real"), _case("d")]
        for c in cases:
            c["question"] = c["id"]
        return cases

    def test_perfect_judge_passes_the_gate(self):
        judge = _fake_judge({"a": _ok(True), "b": _ok(False), "c": _ok(True)})
        out = judge_eval.run_judge_agreement(self._cases(), runs=2, judge=judge, verbose=False)
        assert out["labelled"] == 3 and out["skipped_unlabelled"] == 1
        assert out["kappa_mean"] == pytest.approx(1.0)
        assert out["consistency"] == 1.0
        assert out["gate_passed"] is True and out["disagreements"] == []
        assert len(judge.calls) == 6

    def test_judge_is_pinned_to_temperature_zero_and_the_given_model(self):
        judge = _fake_judge({"a": _ok(True), "b": _ok(False), "c": _ok(True)})
        out = judge_eval.run_judge_agreement(
            self._cases(), runs=1, model="judge-model", judge=judge, verbose=False)
        assert {c["temperature"] for c in judge.calls} == {0}
        assert {c["model"] for c in judge.calls} == {"judge-model"}
        assert out["settings"]["model"] == "judge-model"
        assert len(out["settings"]["prompt_sha256_12"]) == 12

    def test_a_judge_that_says_faithful_to_everything_fails_the_gate(self):
        judge = _fake_judge({"a": _ok(True), "b": _ok(True), "c": _ok(True)})
        out = judge_eval.run_judge_agreement(self._cases(), runs=1, judge=judge, verbose=False)
        assert out["kappa_mean"] == pytest.approx(0.0)
        assert out["gate_passed"] is False
        assert out["per_run"][0]["missed_unfaithful_rate"] == 1.0
        assert [r["id"] for r in out["disagreements"]] == ["b"]

    def test_judge_error_field_and_raised_exception_both_count_as_error(self):
        failed = {"faithful": False, "reason": "Judge error", "error": "APIError"}
        judge = _fake_judge({"a": failed, "b": _ok(False), "c": RuntimeError("boom")})
        out = judge_eval.run_judge_agreement(self._cases(), runs=1, judge=judge, verbose=False)
        labels = {r["id"]: r["judge_labels"] for r in out["per_case"]}
        assert labels == {"a": [E], "b": [U], "c": [E]}
        assert out["per_run"][0]["judge_errors"] == 2
        assert out["per_run"][0]["n"] == 3

    def test_a_verdict_that_flips_between_runs_is_not_consistent(self):
        judge = _fake_judge({"a": [_ok(True), _ok(False)], "b": _ok(False), "c": _ok(True)})
        out = judge_eval.run_judge_agreement(self._cases(), runs=2, judge=judge, verbose=False)
        assert out["consistency"] == pytest.approx(2 / 3)
        assert out["per_run"][0]["kappa"] == pytest.approx(1.0)
        assert out["per_run"][1]["kappa"] < 1.0
        assert out["by_category"]["added_fact"] == {"cases": 2, "agree": 3, "verdicts": 4}
        assert out["by_source"]["pipeline"] == {"cases": 1, "agree": 2, "verdicts": 2}

    def test_one_label_only_gives_undefined_kappa_and_a_failed_gate(self):
        cases = [c for c in self._cases() if c.get("reference_label") == F]
        judge = _fake_judge({"a": _ok(True), "c": _ok(True)})
        out = judge_eval.run_judge_agreement(cases, runs=1, judge=judge, verbose=False)
        assert out["kappa_mean"] is None and out["gate_passed"] is False

    def test_result_counts_who_gave_the_labels(self):
        cases = self._cases()
        cases[1]["labelled_by"] = "ai"
        judge = _fake_judge({"a": _ok(True), "b": _ok(False), "c": _ok(True)})
        out = judge_eval.run_judge_agreement(cases, runs=1, judge=judge, verbose=False)
        assert out["labelled_by"] == {"human": 2, "ai": 1}

    def test_runs_must_be_positive(self):
        with pytest.raises(ValueError):
            judge_eval.run_judge_agreement(self._cases(), runs=0, judge=_fake_judge({}))

    def test_result_is_json_serialisable(self):
        judge = _fake_judge({"a": _ok(True), "b": _ok(True), "c": _ok(True)})
        out = judge_eval.run_judge_agreement(self._cases(), runs=1, judge=judge, verbose=False)
        json.dumps(out)


# -------------------------------- CLI --------------------------------

class TestMain:
    def _gold(self, tmp_path, by="human"):
        cases = [_case("a", F, split="dev"), _case("b", U, split="dev"),
                 _case("c", F, split="test"), _case("d", U, split="test"),
                 _case("e", split="test")]
        for c in cases:
            c["question"] = c["id"]
            if "labelled_by" in c:
                c["labelled_by"] = by
        return _write(tmp_path, cases)

    def _patch_judge(self, monkeypatch, verdicts):
        from agentforge import evaluation
        judge = _fake_judge(verdicts)
        monkeypatch.setattr(evaluation, "score_faithfulness", judge)
        return judge

    def test_pass_returns_zero_and_writes_the_run(self, tmp_path, monkeypatch, capsys):
        self._patch_judge(monkeypatch, {"a": _ok(True), "b": _ok(False),
                                        "c": _ok(True), "d": _ok(False)})
        out_file = tmp_path / "run.json"
        code = judge_eval.main(["--judge-agreement", "--runs=1",
                                f"--gold={self._gold(tmp_path)}", f"--out={out_file}"])
        assert code == 0
        saved = json.loads(out_file.read_text(encoding="utf-8"))
        assert saved["settings"]["split"] == "all" and saved["labelled"] == 4
        assert "PASS" in capsys.readouterr().out

    def test_below_target_returns_one_and_lists_the_disagreement(
            self, tmp_path, monkeypatch, capsys):
        self._patch_judge(monkeypatch, {"a": _ok(True), "b": _ok(True),
                                        "c": _ok(True), "d": _ok(True)})
        code = judge_eval.main(["--judge-agreement", "--runs=1",
                                f"--gold={self._gold(tmp_path)}"])
        printed = capsys.readouterr().out
        assert code == 1 and "FAIL" in printed
        assert "reference=unfaithful" in printed

    @pytest.mark.parametrize("by, warned", [("human", False), ("ai", True)])
    def test_report_says_when_the_labels_are_not_from_a_human(
            self, tmp_path, monkeypatch, capsys, by, warned):
        self._patch_judge(monkeypatch, {"a": _ok(True), "b": _ok(False),
                                        "c": _ok(True), "d": _ok(False)})
        judge_eval.main(["--judge-agreement", "--runs=1",
                         f"--gold={self._gold(tmp_path, by=by)}"])
        printed = capsys.readouterr().out
        assert f"Reference labels by: {by} 4" in printed
        assert ("not agreement with people" in printed) is warned

    def test_split_limits_the_cases(self, tmp_path, monkeypatch):
        judge = self._patch_judge(monkeypatch, {"c": _ok(True), "d": _ok(False)})
        code = judge_eval.main(["--judge-agreement", "--runs=1", "--split=test",
                                f"--gold={self._gold(tmp_path)}"])
        assert code == 0
        assert sorted(c["question"] for c in judge.calls) == ["c", "d"]

    def test_nothing_labelled_returns_two_without_calling_the_judge(
            self, tmp_path, monkeypatch):
        judge = self._patch_judge(monkeypatch, {})
        path = _write(tmp_path, [_case("a"), _case("b")])
        assert judge_eval.main(["--judge-agreement", f"--gold={path}"]) == 2
        assert judge.calls == []

    def test_unknown_split_returns_two(self, tmp_path):
        assert judge_eval.main(["--judge-agreement", "--split=train",
                                f"--gold={self._gold(tmp_path)}"]) == 2
