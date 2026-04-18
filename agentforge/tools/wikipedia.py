"""Wikipedia lookup tool.

Fetches the introductory summary of a Wikipedia article via the public
REST API. No API key required; the API requires a User-Agent header.
"""
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from agentforge.tools._safety import sanitize_text, wrap_untrusted

logger = logging.getLogger(__name__)


def wikipedia_lookup(topic: str) -> str:
    """Fetch a short summary of a Wikipedia article via the REST API."""
    logger.info("Wikipedia lookup invoked for topic: %s", topic)
    try:
        if not topic or not isinstance(topic, str):
            return "Error: topic must be a non-empty string"

        encoded = urllib.parse.quote(topic.strip(), safe="")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

        req = urllib.request.Request(url, headers={"User-Agent": "AgentForge/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        title = data.get("title", topic)
        extract = data.get("extract", "")

        if not extract:
            return f"No summary available for '{topic}'"

        # Wikipedia content is user-editable — sanitize + wrap before passing
        # to the LLM so any injection attempt in the article is treated as data.
        clean_title = sanitize_text(title, 200)
        clean_extract = sanitize_text(extract, 1500)
        return wrap_untrusted(f"{clean_title}: {clean_extract}", source="Wikipedia")

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"No Wikipedia article found for '{topic}'"
        return f"Wikipedia API error (HTTP {e.code}) for '{topic}'"
    except urllib.error.URLError as e:
        return f"Could not reach Wikipedia: {e.reason}"
    except Exception as e:
        return f"Error looking up '{topic}': {e}"


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
