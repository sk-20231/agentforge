"""Wikipedia lookup tool.

Fetches the introductory summary of a Wikipedia article via the public
REST API. No API key required; the API requires a User-Agent header.
"""
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from agentforge.tools._safety import sanitize_text

logger = logging.getLogger(__name__)


def wikipedia_lookup(topic: str) -> str:
    """Fetch a short summary of a Wikipedia article via the REST API.

    Returns raw, *sanitized* text on success. On any failure it RAISES
    (``ValueError``) so the MCP server maps it to ``isError: true`` (an
    LLM-recoverable tool error). It no longer wraps the result — the MCP gateway
    is the single place that wraps tool output as untrusted data, with the turn's
    nonce (Step 17e). Sanitization stays here as defence-in-depth on the
    machine-level tricks (control/HTML/zero-width chars).
    """
    logger.info("Wikipedia lookup invoked for topic: %s", topic)
    if not topic or not isinstance(topic, str):
        raise ValueError("topic must be a non-empty string")

    encoded = urllib.parse.quote(topic.strip(), safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    req = urllib.request.Request(url, headers={"User-Agent": "AgentForge/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(f"No Wikipedia article found for '{topic}'") from e
        raise ValueError(f"Wikipedia API error (HTTP {e.code}) for '{topic}'") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Could not reach Wikipedia: {e.reason}") from e

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
