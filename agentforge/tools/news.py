"""HackerNews top-stories tool.

Uses the Algolia HackerNews Search API to find recent high-quality
stories on a topic. Output is sanitized and wrapped in untrusted-data
delimiters so the LLM treats story titles as data, not instruction.
"""
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from agentforge.tools._safety import extract_domain, sanitize_text, wrap_untrusted

logger = logging.getLogger(__name__)

# search_by_date sorts newest first; the numericFilter keeps out brand-new
# low-quality posts. Result: recent stories with at least some traction.
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
DEFAULT_MIN_POINTS = 20
DEFAULT_HITS_PER_PAGE = 5


def get_top_news(topic: str) -> str:
    """Return top HN stories for a topic as a compact, sanitized list."""
    logger.info("HN news lookup invoked for topic: %s", topic)
    try:
        if not topic or not isinstance(topic, str):
            return "Error: topic must be a non-empty string"

        params = urllib.parse.urlencode({
            "query": topic.strip(),
            "tags": "story",
            "numericFilters": f"points>{DEFAULT_MIN_POINTS}",
            "hitsPerPage": DEFAULT_HITS_PER_PAGE,
        })
        req = urllib.request.Request(
            f"{HN_SEARCH_URL}?{params}",
            headers={"User-Agent": "AgentForge/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        hits = data.get("hits") or []
        if not hits:
            return f"No recent HN stories found for '{topic}'"

        clean_topic = sanitize_text(topic, 60)
        lines = [f"Top HN stories for '{clean_topic}':"]
        for i, hit in enumerate(hits, start=1):
            title = sanitize_text(hit.get("title") or "", 200)
            if not title:
                continue
            points = hit.get("points") or 0
            domain = extract_domain(hit.get("url") or "")
            domain_suffix = f" — {domain}" if domain else ""
            lines.append(f"{i}. {title} ({points} pts){domain_suffix}")

        return wrap_untrusted("\n".join(lines), source="HackerNews")

    except urllib.error.HTTPError as e:
        return f"HN API error (HTTP {e.code}) for '{topic}'"
    except urllib.error.URLError as e:
        return f"Could not reach HN search: {e.reason}"
    except Exception as e:
        return f"Error looking up news for '{topic}': {e}"


TOOL_FUNCTION = get_top_news

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_top_news",
        "description": (
            "Search HackerNews for recent top stories on a topic. "
            "Returns up to 5 stories with titles, points, and source domains. "
            "Use for tech news and developer-focused topics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Keyword or topic to search for, e.g. 'OpenAI', 'Rust', 'quantum computing'",
                }
            },
            "required": ["topic"],
        },
    },
}
