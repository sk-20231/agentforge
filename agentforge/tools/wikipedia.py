"""Wikipedia lookup tool.

Fetches the introductory summary of a Wikipedia article via the public
REST API. No API key required; the API requires a User-Agent header.
"""
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from agentforge.tools._safety import sanitize_text

logger = logging.getLogger(__name__)

_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
# Full-text search, not title-prefix search: "current population of Canada"
# has no title match at all, but full-text search ranks "Population of Canada"
# first (checked live 2026-10-23, issue #65).
_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page?{}"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "AgentForge/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _summary(title: str) -> Optional[dict]:
    """The page summary for an exact title, or None when no page has that title."""
    try:
        return _get_json(_SUMMARY_URL.format(urllib.parse.quote(title, safe="")))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise ValueError(f"Wikipedia API error (HTTP {e.code}) for '{title}'") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Could not reach Wikipedia: {e.reason}") from e


def _best_match(query: str) -> Optional[str]:
    """The title of the top full-text search hit for ``query``, or None.

    A failed search counts as "no match": the exact-title lookup before it
    already proved Wikipedia is reachable, so the caller reports not-found.
    """
    url = _SEARCH_URL.format(urllib.parse.urlencode({"q": query, "limit": 1}))
    try:
        data = _get_json(url)
    except (urllib.error.URLError, ValueError) as e:  # HTTPError is a URLError
        logger.warning("Wikipedia search failed for %r: %s", query, e)
        return None
    pages = data.get("pages") if isinstance(data, dict) else None
    if not isinstance(pages, list) or not pages or not isinstance(pages[0], dict):
        return None
    # `key` is the URL form of the title (underscores); either works for the summary.
    title = pages[0].get("key") or pages[0].get("title")
    return title if isinstance(title, str) and title.strip() else None


def wikipedia_lookup(topic: str) -> str:
    """Fetch a short summary of a Wikipedia article via the REST API.

    The topic is tried as an exact page title first. When no page has that
    title, the top full-text search hit is used instead (issue #65), so a
    near-miss like "current population of Canada" still finds "Population of
    Canada". The returned title is the article actually used.

    Returns raw, *sanitized* text on success. On any failure it RAISES
    (``ValueError``) so the MCP server maps it to ``isError: true`` (an
    LLM-recoverable tool error). It no longer wraps the result — the MCP gateway
    is the single place that wraps tool output as untrusted data, with the turn's
    nonce (Step 17e). Sanitization stays here as defence-in-depth on the
    machine-level tricks (control/HTML/zero-width chars).
    """
    logger.info("Wikipedia lookup invoked for topic: %s", topic)
    if not topic or not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string")
    topic = topic.strip()

    data = _summary(topic)
    if data is None:
        match = _best_match(topic)
        if match is None:
            raise ValueError(f"No Wikipedia article found for '{topic}'")
        logger.info("Wikipedia: no page titled %r, using top search hit %r", topic, match)
        data = _summary(match)
        if data is None:
            raise ValueError(f"No Wikipedia article found for '{topic}'")

    title = data.get("title", topic)
    extract = data.get("extract", "")
    if not extract:
        raise ValueError(f"No summary available for '{topic}'")

    clean_title = sanitize_text(title, 200)
    clean_extract = sanitize_text(extract, 1500)
    return f"{clean_title}: {clean_extract}"


TOOL_FUNCTION = wikipedia_lookup

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wikipedia_lookup",
        "description": "Look up a topic on Wikipedia and return a short summary",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to search for on Wikipedia",
                }
            },
            "required": ["topic"],
        },
    },
}
