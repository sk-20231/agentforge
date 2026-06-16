"""Tests for the forced-fetch isolation pass + harder attacks (issue #25).

Hermetic: classify_forced is a pure function over (fetch audit outcomes, sink URLs), so
it's tested directly; evaluate_forced's summary math is tested with run_forced_fetch
monkeypatched (no subprocess, no network). The real-gateway forced fetch is exercised
by the optional live run, mirroring the rest of the harness's hermetic test style.
"""
from agentforge import redteam_fullstack as rf
from agentforge.redteam_fullstack import (
    BEHAVIORAL,
    TRIFECTA,
    Attack,
    classify_forced,
    evaluate_forced,
)


def _trifecta(target_kind, url):
    return Attack(id="z", name="t", attack_class=TRIFECTA, user_prompt="look it up",
                  payload="cover", canary="C", target_kind=target_kind, forced_url=url)


class TestClassifyForced:
    def test_ssrf_blocked_by_guard(self):
        a = _trifecta("ssrf", "http://169.254.169.254/latest/meta-data/")
        r = classify_forced(a, ["url_blocked"], [])
        assert r["blocked_by"] == rf.L_SSRF
        assert r["dispatched"] is False

    def test_exfil_dispatched_after_hitl(self):
        a = _trifecta("exfil_public", "http://example.com/c?n=X")
        # public URL: SSRF guard allows -> HITL asked -> auto-approve -> dispatched.
        r = classify_forced(a, ["approval_requested", "ok"], ["http://example.com/c?n=X"])
        assert r["blocked_by"] is None
        assert r["dispatched"] is True
        assert r["reached_hitl"] is True

    def test_hitl_denied(self):
        a = _trifecta("exfil_public", "http://example.com/c?n=X")
        r = classify_forced(a, ["approval_requested", "denied"], [])
        assert r["blocked_by"] == rf.L_HITL
        assert r["dispatched"] is False

    def test_ssrf_leak_is_a_failure_signal(self):
        # If an SSRF URL somehow reaches the sink, the guard FAILED — dispatched True.
        a = _trifecta("ssrf", "http://127.0.0.1:6379/")
        r = classify_forced(a, ["ok"], ["http://127.0.0.1:6379/"])
        assert r["dispatched"] is True
        assert r["blocked_by"] is None  # not blocked -> shows up as a leak in the report


class TestEvaluateForcedSummary:
    def test_summary_math(self, monkeypatch):
        # Fake the per-attack run: SSRF always blocked by the guard; exfil always
        # dispatched past auto-approved HITL. Tests the aggregation, not the gateway.
        def fake_run(attack, log_path, tmpdir):
            if attack.target_kind == "ssrf":
                return classify_forced(attack, ["url_blocked"], [])
            return classify_forced(attack, ["approval_requested", "ok"], [attack.forced_url])

        monkeypatch.setattr(rf, "run_forced_fetch", fake_run)
        report = evaluate_forced(rf.ATTACKS)

        assert report["ssrf"]["n"] == 4
        assert report["ssrf"]["blocked_by_guard"] == 4
        assert report["ssrf"]["leaked_ids"] == []          # no SSRF leak -> guard healthy
        assert report["exfil_public"]["n"] == 6
        assert report["exfil_public"]["reached_hitl"] == 6
        assert report["exfil_public"]["dispatched_under_auto_approve"] == 6


class TestAttackSetInvariants:
    def test_every_trifecta_has_a_forced_url(self):
        for a in rf.ATTACKS:
            if a.attack_class == TRIFECTA:
                assert a.forced_url, f"{a.id} (trifecta) needs a forced_url"

    def test_behavioral_attacks_have_no_forced_url(self):
        for a in rf.ATTACKS:
            if a.attack_class == BEHAVIORAL:
                assert a.forced_url is None

    def test_harder_attacks_present(self):
        ids = {a.id for a in rf.ATTACKS}
        assert {"h1", "h2", "h3", "h4", "h5", "h6"} <= ids
        for a in rf.ATTACKS:
            if a.id.startswith("h"):
                assert a.attack_class == BEHAVIORAL
                assert a.canary  # each carries the token the injection tries to emit
                # (h2 deliberately encodes its canary in base64, so it is NOT literal
                #  in the payload — that obfuscation is the point of the harder set.)
