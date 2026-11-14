"""Load a local Hugging Face model without contacting huggingface.co (issue #60).

WHY: ``transformers`` and ``gliner`` check huggingface.co for a newer version of
every model file on each load, even when the files are already in the local
cache. That sends the machine's IP address, start times and model names to
Hugging Face once per process, against the local-only egress posture of the
guardrails, and it is slow on a broken network (100.7 s vs 13.8 s measured).
A deployed container starts many times, so it would do this on every start.

HOW: ``load_local_first`` first loads from the cache with the hub library in
offline mode, so no request is made at all. ``local_files_only=True`` alone is
not enough: gliner does not pass it on when it loads its base model's config
(``gliner/modeling/encoder.py``), which still made one request. Offline mode
blocks that one too (measured with the network blocked: 7 attempts → 0).

Only when the model is NOT cached yet does it load the normal way, which
downloads it once — with a WARNING, so a download is never silent. If the user
has set ``HF_HUB_OFFLINE=1``, there is no download: the error is raised and the
guardrail reports UNAVAILABLE, as before.

Egress: this module only reads the local cache, and the fallback is the same
download ``from_pretrained`` did before. It imports no network client itself.
"""
import contextlib
import logging
import os
import threading
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Offline mode is a process-wide flag in huggingface_hub. One lock around each
# whole load, so one load's offline window can never break another thread's
# download.
_LOCK = threading.Lock()


@contextlib.contextmanager
def _hub_offline():
    """Turn the hub's offline mode on for the block, and always restore it."""
    import huggingface_hub.constants as hub_constants  # lazy: optional dependency
    previous = hub_constants.HF_HUB_OFFLINE
    hub_constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        hub_constants.HF_HUB_OFFLINE = previous


def _user_set_offline() -> bool:
    import huggingface_hub.constants as hub_constants
    return bool(hub_constants.HF_HUB_OFFLINE)


def load_local_first(load: Callable[[bool], T], model_id: str) -> T:
    """Return ``load(local_files_only)``, from the local cache when possible.

    ``load`` takes one argument, ``local_files_only``, and passes it to every
    ``from_pretrained`` call it makes. It is called with ``True`` first, inside
    offline mode. If that fails, it is called once more with ``False`` (a normal
    load that downloads the model) — unless the user set ``HF_HUB_OFFLINE=1``.

    ImportError is re-raised at once: a missing library is not a missing model.
    """
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    with _LOCK:
        try:
            with _hub_offline():
                result = load(True)
            logger.debug("hf_local: %s loaded from the local cache.", model_id)
            return result
        except ImportError:
            raise
        except Exception as exc:  # not cached (LocalEntryNotFoundError, OSError, ...)
            if _user_set_offline():
                logger.warning("hf_local: %s is not available offline (%s) and "
                               "HF_HUB_OFFLINE is set, so it is not downloaded.",
                               model_id, type(exc).__name__)
                raise
            logger.warning("hf_local: %s is not in the local cache (%s); "
                           "downloading it once from huggingface.co.",
                           model_id, type(exc).__name__)
        return load(False)
