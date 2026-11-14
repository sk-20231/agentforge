"""load_local_first: load guardrail models from the local cache with no request
to huggingface.co, and download only when a model is not cached (issue #60).

A fake ``huggingface_hub.constants`` stands in for the real one, so these tests
need no optional dependency and run the same in CI (which does not install
huggingface_hub) and locally.
"""
import logging
import os
import sys
import types

import pytest

from agentforge import hf_local


@pytest.fixture
def hub(monkeypatch):
    """A fake huggingface_hub package whose offline flag starts OFF."""
    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HUB_OFFLINE = False
    package = types.ModuleType("huggingface_hub")
    package.constants = constants
    monkeypatch.setitem(sys.modules, "huggingface_hub", package)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    return constants


def test_cached_model_loads_offline_with_no_download(hub):
    calls = []

    def load(local_files_only):
        calls.append((local_files_only, hub.HF_HUB_OFFLINE))
        return "model"

    assert hf_local.load_local_first(load, "org/model") == "model"
    assert calls == [(True, True)]      # one call: cache only, offline mode on
    assert hub.HF_HUB_OFFLINE is False  # restored afterwards


def test_uncached_model_downloads_once_with_a_warning(hub, caplog):
    calls = []

    def load(local_files_only):
        calls.append((local_files_only, hub.HF_HUB_OFFLINE))
        if local_files_only:
            raise OSError("not in the local cache")
        return "downloaded"

    with caplog.at_level(logging.WARNING, logger="agentforge.hf_local"):
        assert hf_local.load_local_first(load, "org/model") == "downloaded"
    # First from the cache in offline mode, then ONE normal load with offline off.
    assert calls == [(True, True), (False, False)]
    assert "org/model is not in the local cache" in caplog.text
    assert "downloading it once" in caplog.text
    assert hub.HF_HUB_OFFLINE is False


def test_no_download_when_the_user_set_offline(hub, caplog):
    hub.HF_HUB_OFFLINE = True           # the user's HF_HUB_OFFLINE=1
    calls = []

    def load(local_files_only):
        calls.append(local_files_only)
        raise OSError("not in the local cache")

    with caplog.at_level(logging.WARNING, logger="agentforge.hf_local"):
        with pytest.raises(OSError):
            hf_local.load_local_first(load, "org/model")
    assert calls == [True]              # never tried to download
    assert "HF_HUB_OFFLINE is set" in caplog.text
    assert hub.HF_HUB_OFFLINE is True   # the user's setting is kept


def test_missing_library_is_not_treated_as_a_missing_model(hub):
    calls = []

    def load(local_files_only):
        calls.append(local_files_only)
        raise ImportError("No module named 'transformers'")

    with pytest.raises(ImportError):
        hf_local.load_local_first(load, "org/model")
    assert calls == [True]
    assert hub.HF_HUB_OFFLINE is False


def test_telemetry_is_turned_off(hub, monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    hf_local.load_local_first(lambda local_files_only: None, "org/model")
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


# ---------- the pinned revisions ----------

def test_default_models_are_pinned_to_the_reviewed_commits():
    from agentforge import config
    assert config._GUARDRAIL_MODEL_PIN == "90c9989b1a342275dd0d1a95aad283c04e075671"
    assert config._OUTPUT_GUARDRAIL_MODEL_PIN == "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d"


@pytest.mark.parametrize("env, model_id, expected", [
    (None, "org/default", "sha1"),       # default model, nothing set -> the pin
    (None, "org/other", None),           # another model -> latest, not our pin
    ("abc123", "org/default", "abc123"), # set -> that commit
    ("abc123", "org/other", "abc123"),
    ("", "org/default", None),           # set to empty -> latest
    ("  ", "org/default", None),
])
def test_pinned_revision_rule(monkeypatch, env, model_id, expected):
    from agentforge.config import _pinned_revision
    if env is None:
        monkeypatch.delenv("X_TEST_REVISION", raising=False)
    else:
        monkeypatch.setenv("X_TEST_REVISION", env)
    assert _pinned_revision("X_TEST_REVISION", model_id, "org/default", "sha1") == expected


# ---------- the guardrail passes the pin and the cache flag on ----------

def test_injection_classifier_loads_through_load_local_first(monkeypatch):
    """_build_scanner must load via load_local_first, passing local_files_only
    and the pinned revision to BOTH from_pretrained calls."""
    from agentforge import guardrail

    seen = []

    class _Fake:
        def __init__(self, kind):
            self.kind = kind

        def from_pretrained(self, model_id, **kw):
            seen.append((self.kind, model_id, kw))
            if self.kind == "model":
                m = types.SimpleNamespace(eval=lambda: None,
                                          config=types.SimpleNamespace(id2label={0: "SAFE", 1: "INJECTION"}))
                return m
            return object()

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = _Fake("tokenizer")
    fake_tf.AutoModelForSequenceClassification = _Fake("model")
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    via = []
    monkeypatch.setattr(guardrail, "load_local_first",
                        lambda load, model_id: via.append(model_id) or load(True))
    monkeypatch.setattr(guardrail, "AGENT_GUARDRAIL_MODEL_REVISION", "pinned-sha")

    guardrail._build_scanner()

    assert via == [guardrail.AGENT_GUARDRAIL_MODEL_ID]
    assert [k for k, _, _ in seen] == ["tokenizer", "model"]
    for _, _, kw in seen:
        assert kw == {"revision": "pinned-sha", "local_files_only": True}
